import torch
from nle import nethack
from torch import nn
from torch.nn import functional as F

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sf_examples.nethack.models.utils import _step_to_range, Crop
from sf_examples.nethack.models.chaotic_dwarf import MessageEncoder, BLStatsEncoder






        

class SymbolicGlyphNet(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.obs_space = obs_space
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        self.use_prev_action = cfg.use_prev_action

        glyphs_shape = obs_space["glyphs"].shape
        
        self.H = glyphs_shape[0]
        self.W = glyphs_shape[1]
        self.crop_dim = cfg.crop_dim
        self.edim = cfg.glyph_edim
        self.k_dim = 2 * self.edim

        self.encoder_out_size = 0
        
        self.glyph_embed = nn.Embedding(nethack.MAX_GLYPH, self.edim)
        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.crop_conv = nn.Sequential(
            nn.Conv2d(self.edim, self.k_dim, kernel_size=(3, 3), stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(self.k_dim, 2 * self.k_dim, kernel_size=(3, 3), stride=2),
            nn.ELU(inplace=True),
        )
                
        self.topline_encoder = torch.jit.script(MessageEncoder())
        self.bottomline_encoder = torch.jit.script(BLStatsEncoder())        
        topline_shape = obs_space["message"].shape
        bottomline_shape = obs_space["blstats"].shape
               
        if self.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.num_actions = None
            self.prev_actions_dim = 0


        self.encoder_out_size = sum(
            [
                calc_num_elements(self.topline_encoder, topline_shape),
                calc_num_elements(self.bottomline_encoder, bottomline_shape),
                calc_num_elements(self.crop_conv, (self.edim, self.crop_dim, self.crop_dim)),
                self.prev_actions_dim,
            ]
        )

        if self.cfg.with_sol:
            self.num_policies = self.obs_space['rewards'].shape[0]
            self.policy_encoder = nn.Linear(self.num_policies, self.edim)
            self.encoder_out_size += self.edim
        


    def _select(self, embed, x, max_dim=None):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        if max_dim is None:
            out = embed.weight.index_select(0, x.reshape(-1))
        else:
            out = embed.weight[:, :max_dim].index_select(0, x.reshape(-1))
        try:
            return out.reshape(x.shape + (-1,))
        except Exception as e:
            raise ValueError("Invalid size") from e
        

    def forward(self, obs_dict):

        topline = obs_dict["message"]
        bottom_line = obs_dict["blstats"]
        glyphs = obs_dict["glyphs"].int()
        B, H, W = glyphs.shape

        encodings = []
        
        topline_embed = self.topline_encoder(topline.float(memory_format=torch.contiguous_format).view(B, -1))
        encodings.append(topline_embed)
        
        bottomline_embed = self.bottomline_encoder(bottom_line.float(memory_format=torch.contiguous_format).view(B, -1))
        encodings.append(bottomline_embed)

        coordinates = bottom_line[:, :2].int()
        crop_glyphs = self.crop(glyphs, coordinates)
        crop_embed = self._select(self.glyph_embed, crop_glyphs).permute(0, 3, 1, 2)
        if self.cfg.with_sol:
            policy_embed = self.policy_encoder(F.one_hot(obs_dict['current_policy'].long(), self.num_policies).float().view(B, -1))
            encodings.append(policy_embed)
            crop_embed = crop_embed + policy_embed.unsqueeze(-1).unsqueeze(-1)
        
        crop_embed = self.crop_conv(crop_embed)
        encodings.append(crop_embed.float(memory_format=torch.contiguous_format).view(B, -1))

        if self.use_prev_action:
            prev_actions = obs_dict["prev_actions"].long().view(B)
            encodings.append(torch.nn.functional.one_hot(prev_actions, self.num_actions))

        return torch.cat(encodings, dim=1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


