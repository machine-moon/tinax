"""Parse immutable application config and log without touching global handlers."""

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO

from tinax.stdlib import make_stream_logger, parse_config


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Configuration produced from explicit command arguments."""

    steps: int
    label: str


def _config(values: Mapping[str, object]) -> RunConfig:
    steps = values["steps"]
    label = values["label"]
    if not isinstance(steps, int) or isinstance(steps, bool) or not isinstance(label, str):
        raise TypeError("parsed steps and label have unexpected types")
    return RunConfig(steps=steps, label=label)


def run(argv: Sequence[str], stream: TextIO) -> RunConfig:
    """Parse explicit arguments, emit one isolated log record, and return immutable config."""
    parser = argparse.ArgumentParser(prog="tinax-stdlib-example")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--label", required=True)
    config = parse_config(parser, argv, _config)
    logger = make_stream_logger(
        "tinax.example",
        level=logging.INFO,
        stream=stream,
        format_string="%(levelname)s %(message)s",
    )
    logger.info("label=%s steps=%d", config.label, config.steps)
    return config
