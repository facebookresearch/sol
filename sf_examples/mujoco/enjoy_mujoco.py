import sys

from sample_factory.enjoy import enjoy
from sf_examples.mujoco.train_mujoco import parse_mujoco_cfg, register_mujoco_components

from sample_factory.cfg.arguments import (
    maybe_load_from_checkpoint,
)


def main():
    """Script entry point."""
    register_mujoco_components()
    cfg = parse_mujoco_cfg(evaluation=True)
    cfg = maybe_load_from_checkpoint(cfg)
    cfg.save_video=True
    cfg.max_num_episodes=2
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
