import os
import sys
import torch

import wandb

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.utils import log
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.encoder import Encoder
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.wandb_utils import init_wandb
from sf_examples.nethack.models import MODELS_LOOKUP
from sf_examples.nethack.nethack_env import NETHACK_ENVS, make_nethack_env
from sf_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_model,
    add_extra_params_nethack_env,
    add_extra_params_rewards,
    nethack_override_defaults,
)




def register_nethack_envs():
    for env in NETHACK_ENVS:
        register_env(env.name, make_nethack_env)


def make_nethack_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    try:
        model_cls = MODELS_LOOKUP[cfg.model]
    except KeyError:
        raise NotImplementedError("model=%s" % cfg.model) from None

    return model_cls(cfg, obs_space)


def register_nethack_components():
    register_nethack_envs()
    global_model_factory().register_encoder_factory(make_nethack_encoder)


def parse_nethack_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_general(parser)
    add_extra_params_rewards(parser)
    nethack_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)

    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()
    cfg = parse_nethack_args()

    if cfg.use_ddp and cfg.with_wandb:
        rank = int(os.environ.get("RANK"))
        if rank == 0:
            log.debug(f"[DDP] Weights and Biases integration enabled for rank {rank} process")
        else:
            log.debug(f"[DDP] Weights and Biases integration disabled for rank {rank} process")
            cfg.with_wandb = False


    if cfg.train_dir == "wandb_auto":
        init_wandb(cfg)  # should be done before writers are initialized
        cfg.train_dir = f"{wandb.run.dir}/train_dir"
        wandb.config.update({"local_train_dir": cfg.train_dir}, allow_val_change=True)

    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
