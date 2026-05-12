import torch
from nle import nethack
from torch import nn
from torch.nn import functional as F

import numpy as np

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sf_examples.nethack.models.utils import _step_to_range, Crop
from sf_examples.nethack.models.chaotic_dwarf import MessageEncoder, BLStatsEncoder
from sf_examples.nethack.utils.nle_tokenizer.tokenizer import NLE_TOKENIZER






class SymbolicGlyphTokenNetEmbeddingBag(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.obs_space = obs_space
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        self.use_prev_action = cfg.use_prev_action
        self.use_glyph_directions = cfg.use_glyph_directions

        self.encoder_out_size = 0

        # glyph image encoder
        glyphs_shape = obs_space["glyphs"].shape

        self.H = glyphs_shape[0]
        self.W = glyphs_shape[1]
        self.crop_dim = cfg.crop_dim
        self.edim = cfg.glyph_edim
        self.k_dim = 2 * self.edim

        self.glyph_embed = nn.Embedding(nethack.MAX_GLYPH, self.edim)
        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.crop_conv = nn.Sequential(
            nn.Conv2d(self.edim, self.k_dim, kernel_size=(3, 3), stride=2),
            nn.ELU(inplace=True),
            nn.Conv2d(self.k_dim, 2 * self.k_dim, kernel_size=(3, 3), stride=2),
            nn.ELU(inplace=True),
        )

        # embedding matrix for NLE tokens
        max_token = np.max(list(NLE_TOKENIZER.values()))
        self.token_embed = nn.EmbeddingBag(max_token + 1, self.k_dim, mode='sum', padding_idx=0)
        torch.nn.init.normal_(self.token_embed.weight, mean=0, std=self.cfg.token_embed_std)

        # learned embeddings to mark menu choices
        self.menu_pos_embed = nn.Parameter(torch.randn(1, cfg.max_menu_items, self.k_dim))

        # blstats encoder
        self.bottomline_encoder = torch.jit.script(BLStatsEncoder())
        bottomline_shape = obs_space["blstats"].shape

        if self.use_prev_action:
            self.num_actions = obs_space["prev_actions"].n
            self.prev_actions_dim = self.num_actions
        else:
            self.num_actions = None
            self.prev_actions_dim = 0

        if self.use_glyph_directions:
            self.glyph_directions_dim = self.obs_space['glyph_directions'].shape[0]
        else:
            self.glyph_directions_dim = 0



        self.encoder_out_size = sum(
            [
                calc_num_elements(self.bottomline_encoder, bottomline_shape),
                calc_num_elements(self.crop_conv, (self.edim, self.crop_dim, self.crop_dim)),
                self.prev_actions_dim,
                self.glyph_directions_dim,
            ]
        )

        # for message
        self.encoder_out_size += self.k_dim
        # for menu
        self.encoder_out_size += self.k_dim

        if self.cfg.use_prev_reward:
            self.encoder_out_size += 1

        if self.cfg.inv_encoder_type == 'bow':
            self.encoder_out_size += self.k_dim
        elif self.cfg.inv_encoder_type == 'att':
            # we will encode the inventory with a simple attention layer, which works better than BoW
            # one could imagine more sophisticated versions where queries depend on the other observations
            self.num_query_heads = self.cfg.inv_query_heads
            self.queries = nn.Parameter(torch.randn(1, self.num_query_heads, self.k_dim) * self.cfg.inv_query_std)
            self.encoder_out_size += self.num_query_heads * self.k_dim

        if self.cfg.use_spellcasting_wrapper:
            self.encoder_out_size += self.cfg.max_spells * self.k_dim

        if self.cfg.use_dungeon_overview_wrapper:
            proj_dim = int(self.k_dim / 4)
            self.dungeon_overview_proj = nn.Linear(self.k_dim, proj_dim, bias=False)
            self.encoder_out_size += self.cfg.max_dungeon_overview_levels * proj_dim


        if self.cfg.with_sol:
            self.num_policies = self.obs_space['rewards'].shape[0]
            self.policy_encoder = nn.Linear(self.num_policies, self.edim)
            self.encoder_out_size += self.edim


        if self.cfg.use_attributes_wrapper:
            self.encoder_out_size += 13  # 13 roles
            self.encoder_out_size += 5   # 5 races
            self.encoder_out_size += 3   # 3 alignments





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

        bottomline_embed = self.bottomline_encoder(bottom_line.float(memory_format=torch.contiguous_format).view(B, -1))
        encodings.append(bottomline_embed)

        coordinates = bottom_line[:, :2].int()
        crop_glyphs = self.crop(glyphs, coordinates)
        crop_embed = self._select(self.glyph_embed, crop_glyphs).permute(0, 3, 1, 2)
        if self.cfg.with_sol:
            policy_embed = self.policy_encoder(obs_dict["current_policy_vec"].float().view(B, -1))
            encodings.append(policy_embed)
            crop_embed = crop_embed + policy_embed.unsqueeze(-1).unsqueeze(-1)

        crop_embed = self.crop_conv(crop_embed)
        encodings.append(crop_embed.float(memory_format=torch.contiguous_format).view(B, -1))

        if self.use_prev_action:
            prev_actions = obs_dict["prev_actions"].long().view(B)
            encodings.append(torch.nn.functional.one_hot(prev_actions, self.num_actions))

        if self.cfg.use_prev_reward:
            prev_rewards = obs_dict["prev_rewards"]
            encodings.append(prev_rewards * self.cfg.reward_scale)

        if self.use_glyph_directions:
            encodings.append(obs_dict["glyph_directions"])

        tokens = obs_dict['msg_tok'].long()
        msg_embed = self.token_embed(tokens)
        encodings.append(msg_embed)


        # embed the inventory
        tokens = obs_dict['inv_tok'].long()
        _, inv_rows, inv_cols = tokens.shape
        inv_embed = self.token_embed(tokens.view(B * inv_rows, inv_cols)).view(B, inv_rows, -1)


        if self.cfg.inv_encoder_type == 'bow':
            # BoW over inventory items
            inv_embed = inv_embed.sum(1)
        elif self.cfg.inv_encoder_type == 'att':
            # attention over inventory items with learnable queries
            inv_embed = F.scaled_dot_product_attention(self.queries.repeat(B, 1, 1), inv_embed, inv_embed)
            inv_embed = inv_embed.view(B, -1)

        encodings.append(inv_embed)

        if self.cfg.use_menu_selection_wrapper:
            tokens = obs_dict['menu_tok'].long()
            _, menu_rows, menu_cols = tokens.shape
            menu_embed = self.token_embed(tokens.view(B * menu_rows, menu_cols)).view(B, menu_rows, -1)
            # this is essentially the encoder from: https://openreview.net/pdf?id=rJTKKKqeg (Section 2.1)
            menu_embed = torch.sum(menu_embed * self.menu_pos_embed, dim=1)
            encodings.append(menu_embed)

        if self.cfg.use_spellcasting_wrapper:
            tokens = obs_dict['spells_tok'].long()
            _, spells_rows, spells_cols = tokens.shape
            spells_embed = self.token_embed(tokens.view(B * spells_rows, spells_cols)).view(B, spells_rows, -1)
            encodings.append(spells_embed.view(B, -1))

        if self.cfg.use_dungeon_overview_wrapper:
            tokens = obs_dict['overview_tok'].long()
            _, rows, cols = tokens.shape
            overview_embed = self.token_embed(tokens.view(B * rows, cols)).view(B, rows, -1)
            overview_embed = self.dungeon_overview_proj(overview_embed)
            encodings.append(overview_embed.view(B, -1))

        if self.cfg.use_attributes_wrapper:
            encodings.append(obs_dict['role'])
            encodings.append(obs_dict['race'])
            encodings.append(obs_dict['align'])


        return torch.cat(encodings, dim=1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


