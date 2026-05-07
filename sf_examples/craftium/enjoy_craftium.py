import sys

from sample_factory.cfg.arguments import (
    parse_full_cfg,
    parse_sf_args,
    maybe_load_from_checkpoint,
)

from sample_factory.enjoy import enjoy
from sf_examples.craftium.train_craftium import parse_craftium_args, register_craftium_components


def main():
    """Script entry point."""
    register_craftium_components()
    cfg = parse_craftium_args(evaluation=True)
    cfg = maybe_load_from_checkpoint(cfg)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
