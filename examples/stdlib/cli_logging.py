"""Parse immutable CLI config and log without touching global handlers via tinax.stdlib."""

import argparse
import logging
import sys
from collections.abc import Mapping
from dataclasses import dataclass

from tinax.stdlib import make_stream_logger, parse_config


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Immutable configuration produced from explicit command arguments."""

    steps: int
    label: str


def _config(values: Mapping[str, object]) -> RunConfig:
    steps, label = values["steps"], values["label"]
    if not isinstance(steps, int) or isinstance(steps, bool) or not isinstance(label, str):
        raise TypeError("parsed steps and label have unexpected types")
    return RunConfig(steps=steps, label=label)


def main() -> None:
    """Parse a fixed argument list into config, then emit one isolated log record."""
    parser = argparse.ArgumentParser(prog="tinax-stdlib-example")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--label", required=True)
    config = parse_config(parser, ["--steps", "3", "--label", "warmup"], _config)

    logger = make_stream_logger(
        "tinax.example", level=logging.INFO, stream=sys.stdout, format_string="%(levelname)s %(message)s"
    )
    logger.info("label=%s steps=%d", config.label, config.steps)


if __name__ == "__main__":
    main()
