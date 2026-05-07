import os
# Use 1st GPU for RL, the rest will be used for Craftium env rendering which is expensive
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl
from sf_examples.craftium.craftium_params import add_extra_params_craftium_env
from sf_examples.craftium.craftium_env import CRAFTIUM_ENVS, make_craftium_env


def register_craftium_envs():
    for env in CRAFTIUM_ENVS:
        register_env(env.name, make_craftium_env)


def register_craftium_components():
    register_craftium_envs()


def parse_craftium_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_craftium_env(parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():  # pragma: no cover
    """Script entry point."""
    register_craftium_components()
    cfg = parse_craftium_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
