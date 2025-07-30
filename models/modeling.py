import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn

from .module_CAttention import CAM
from .module_clip import CLIP, convert_weights, _PT_NAME
from .module_cross import Transformer as TransformerClip
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL

allgather = AllGather.apply
allgather2 = AllGather2.apply

class ResidualLinear(nn.Module):
    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()

        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int), )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x

class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()

        self.config = config

        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if os.path.exists(model_path):
            FileNotFoundError
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            convert_weights(self.clip)

        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn(config)
        self.loss_kl = KL(config)

        self.apply(self.init_weights)  # random init must before loading pretrain
        self.clip.load_state_dict(state_dict, strict=False)

        self.alpha, self.beta = self.config.alpha, self.config.beta
        embed_dim = state_dict["text_projection"].shape[1]
        self.max_words, self.max_frames = self.config.max_words, self.config.max_frames
        self.cf_c_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.cf_f_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1), )
        self.cam_sf = CAM(embed_dim=embed_dim, dropout=0.3)
        self.cam_sc = CAM(embed_dim=embed_dim, dropout=0.3)

        ## ===> Initialization trick [HARD CODE]
        new_state_dict = OrderedDict()

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        self.load_state_dict(new_state_dict, strict=False)

    def forward(self, text, text_mask, text_list, text_mask_list, video, video_mask, idx=None, global_step=0):

        text_mask = text_mask.view(-1, text_mask.shape[-1])
        text = text.view(-1, text.shape[-1])
        text_mask_list = [text_mask.view(-1, text_mask.shape[-1]) for text_mask in text_mask_list]
        text_list = [text.view(-1, text.shape[-1]) for text in text_list]
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = torch.as_tensor(video).float()
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.view(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)

        s_feat = self.get_text_feat(text, text_mask)
        s_feat_list = map(list, zip(*[
            self.get_text_feat(text, text_mask)
            for text, text_mask in zip(text_list, text_mask_list)
        ]))
        c_feat = torch.stack(s_feat_list, dim=1)
        f_feat = self.get_video_feat(video, video_mask)

        s_feat = s_feat.contiguous()  # [a, d]
        c_feat = c_feat.contiguous()  # [b, c, d]
        f_feat = f_feat.contiguous()  # [b, f, d]

        if self.training:
            s_feat = allgather(s_feat, self.config)
            c_feat = allgather(c_feat, self.config)
            f_feat = allgather(f_feat, self.config)
            torch.distributed.barrier()

        logit_scale = self.clip.logit_scale.exp()

        # init params:
        a, d    = s_feat.size(0), s_feat.size(1)
        b, c, d = c_feat.size(0), c_feat.size(1), c_feat.size(2)
        b, f, d = f_feat.size(0), f_feat.size(1), f_feat.size(2),

        ########## Step-I: Sort ##########
        # sims_sf = torch.einsum("ad,bfd->abf", [self.norm(s_feat), self.norm(f_feat)])
        # sims_sf = sims_sf.diagonal(dim1=0, dim2=1).transpose(0, 1)
        # _, f_new_idx = torch.topk(sims_sf, k=f, dim=-1, largest=True)
        # f_feat = f_feat[torch.arange(b)[:, None], f_new_idx, :]
        # sims_sc = torch.einsum("ad,bcd->abc", [self.norm(s_feat), self.norm(c_feat)])
        # sims_sc = sims_sc.diagonal(dim1=0, dim2=1).transpose(0, 1)
        # _, c_new_idx = torch.topk(sims_sc, k=c, dim=-1, largest=True)
        # c_feat = c_feat[torch.arange(b)[:, None], c_new_idx, :]

        ########## Step-II: Interaction ##########
        # <c_feat, f_feat>
        c_w = torch.softmax(self.cf_c_feat_w(c_feat).squeeze(-1), dim=-1)
        f_w = torch.softmax(self.cf_f_feat_w(f_feat).squeeze(-1), dim=-1)
        sims_cf = torch.einsum("acd,bfd->abcf", [self.norm(c_feat), self.norm(f_feat)])
        sims_c2f, _ = sims_cf.max(dim=-1)
        sims_c2f = torch.einsum('abc,ac->ab', [sims_c2f, c_w])
        sims_f2c, _ = sims_cf.max(dim=-2)
        sims_f2c = torch.einsum('abf,bf->ab', [sims_f2c, f_w])
        sims_cf = (sims_c2f + sims_f2c) / 2.0
        loss_cf = (self.loss_fct(sims_cf * logit_scale) + self.loss_fct(sims_cf.T * logit_scale)) / 2.0
        # <s_feat, f_feat>
        f_feat_agg = self.cam_sf(s_feat, f_feat)
        sims_sf = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(f_feat_agg)])
        loss_sf = (self.loss_fct(sims_sf * logit_scale) + self.loss_fct(sims_sf.T * logit_scale)) / 2.0
        # <s_feat, c_feat>
        c_feat_agg = self.cam_sc(s_feat, c_feat)
        sims_sc = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(c_feat_agg)])
        loss_sc = (self.loss_fct(sims_sc * logit_scale) + self.loss_fct(sims_sc.T * logit_scale)) / 2.0

        ########## Step-III: KL Loss ##########
        loss_kl_sf = (self.loss_kl(sims_sf, sims_cf) + self.loss_kl(sims_sf, sims_cf.T)) / 2.0
        loss_kl_sc = (self.loss_kl(sims_sc, sims_cf) + self.loss_kl(sims_sc, sims_cf.T)) / 2.0

        total_loss = loss_cf + loss_sf + loss_sc + (loss_kl_sf + loss_kl_sc)

        if self.training:
            return total_loss
        else:
            return None

    def norm(self, feat):
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def get_text_feat(self, text_ids, text_mask):
        text_ids = text_ids.view(-1, text_ids.shape[-1])
        text_mask = text_mask.view(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        s_feat = self.clip.encode_text(text_ids, return_hidden=False, mask=text_mask)
        s_feat = s_feat.float().view(bs_pair, s_feat.size(-1))
        return s_feat

    def get_video_feat(self, video, video_mask):

        if not self.training:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.view(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.view(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat = self.clip.encode_image(video, return_hidden=False, mask=video_mask)
        f_feat = f_feat.float().view(bs_pair, -1, f_feat.size(-1))
        return f_feat

    def get_similarity_logits(self, s_feat, c_feat, f_feat):

        # OBJECT: (s_feat) -->> (c_feat, f_feat)

        f_feat_agg = self.cam_sf(s_feat, f_feat)
        sims_sf = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(f_feat_agg)])

        c_feat_agg = self.cam_sc(s_feat, c_feat)
        sims_sc = torch.einsum("ad,bad->ab", [self.norm(s_feat), self.norm(c_feat_agg)])

        sims = sims_sf + sims_sc

        return sims

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples
            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
