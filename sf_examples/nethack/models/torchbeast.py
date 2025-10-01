import torch.nn as nn
import torch
from nle import nethack
from torch import nn
from torch.nn import functional as F

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sf_examples.nethack.models.utils import _step_to_range


NUM_CHARS = 256
NUM_COLORS = 16
PAD_CHAR = 0


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)



class TorchBeastMessageEncoder(Encoder):

    def __init__(self, cfg: Config):
        super().__init__(cfg)
    
        # Define network
        out_dim = 0

        self.msg_model = "lt_cnn"
        self.h_dim = 512
        
        if self.msg_model == "lt_cnn_small":
            self.msg_hdim = 32
            self.msg_edim = 16
            self.char_lt = nn.Embedding(NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR)
            self.conv1 = nn.Conv1d(self.msg_edim, self.msg_hdim, kernel_size=7)
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(3 * self.msg_hdim, self.msg_hdim),
                nn.ReLU(),
                nn.Linear(self.msg_hdim, self.msg_hdim),
                nn.ReLU(),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += self.msg_hdim

        elif self.msg_model == "lt_cnn":
            self.msg_hdim = 64
            self.msg_edim = 32
            self.char_lt = nn.Embedding(NUM_CHARS, self.msg_edim, padding_idx=PAD_CHAR)
            self.conv1 = nn.Conv1d(self.msg_edim, self.msg_hdim, kernel_size=7)
            # remaining convolutions, relus, pools, and a small FC network
            self.conv2_6_fc = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv2
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=7),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # conv3
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv4
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv5
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                # conv6
                nn.Conv1d(self.msg_hdim, self.msg_hdim, kernel_size=3),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=3),
                # fc receives -- [ B x h_dim x 5 ]
                Flatten(),
                nn.Linear(5 * self.msg_hdim, 2 * self.msg_hdim),
                nn.ReLU(),
                # nn.Linear(2 * self.msg_hdim, self.msg_hdim),
                # nn.ReLU(),
            )  # final output -- [ B x h_dim x 5 ]
            out_dim += 2 * self.msg_hdim

        #log.debug(f"Encoder latent dimensionality: {out_dim}")

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim // 2),
            nn.ReLU(),
            nn.Linear(self.h_dim // 2, self.h_dim),
        )

        self.encoder_out_size = self.h_dim
        log.debug("Policy head output size: %r", self.encoder_out_size)

    def _select(self, embed, x):
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        try:
            return out.reshape(x.shape + (-1,))
        except Exception as e:
            raise ValueError("Invalid size") from e

    def forward(self, obs_dict):

        messages = obs_dict["message"].long()
        batch_size = messages.shape[0]

        reps = []

        if self.msg_model != "none":
            if self.msg_model == "lt_cnn_small":
                messages = messages[:, :128]  # most messages are much shorter than 256
            char_emb = self.char_lt(messages).transpose(1, 2)
            char_rep = self.conv2_6_fc(self.conv1(char_emb))
            reps.append(char_rep)

        # -- [batch size x K]
        reps = torch.cat(reps, dim=1)
        st = self.fc(reps)

        return st
