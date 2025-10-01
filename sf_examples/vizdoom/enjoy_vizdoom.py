import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sf_examples.vizdoom.doom.doom_params import add_doom_env_args, add_doom_env_eval_args, doom_override_defaults
from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

from sample_factory.cfg.arguments import (
    maybe_load_from_checkpoint,
)



def main():
    """Script entry point."""
    register_vizdoom_components()
    parser, cfg = parse_sf_args(evaluation=True)
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)

    cfg = maybe_load_from_checkpoint(cfg)
    cfg.save_video=True
    cfg.max_num_episodes=10

    
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
