import argparse
import io
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TextIO, cast

import pytest

from tinax.stdlib import make_stream_logger, parse_config


@dataclass(frozen=True, slots=True)
class Config:
    steps: int
    label: str


def _config(values: Mapping[str, object]) -> Config:
    steps = values["steps"]
    label = values["label"]
    if not isinstance(steps, int) or isinstance(steps, bool) or not isinstance(label, str):
        raise TypeError("unexpected parsed value types")
    return Config(steps, label)


def test_parse_config_uses_explicit_arguments_and_an_immutable_mapping() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--label", required=True)
    mapping_is_immutable = False

    def factory(values: Mapping[str, object]) -> Config:
        nonlocal mapping_is_immutable
        try:
            cast(dict[str, object], values)["extra"] = True
        except TypeError:
            mapping_is_immutable = True
        return _config(values)

    config = parse_config(parser, ["--steps", "4", "--label", "run"], factory)

    assert config == Config(4, "run")
    assert mapping_is_immutable


def test_parse_config_preserves_argparse_errors() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, required=True)

    with pytest.raises(SystemExit) as error:
        parse_config(parser, [], _config)
    assert error.value.code == 2


@pytest.mark.parametrize("argv", ["--flag", b"--flag", [1], object()])
def test_parse_config_rejects_invalid_argument_sequences(argv: object) -> None:
    with pytest.raises(TypeError, match="sequence of strings|only strings"):
        parse_config(argparse.ArgumentParser(), cast(Sequence[str], argv), _config)


def test_parse_config_validates_parser_and_factory() -> None:
    with pytest.raises(TypeError, match="ArgumentParser"):
        parse_config(cast(argparse.ArgumentParser, object()), [], _config)
    with pytest.raises(TypeError, match="callable"):
        parse_config(argparse.ArgumentParser(), [], cast(Callable[[Mapping[str, object]], object], None))


def test_stream_logger_is_isolated_from_root_and_registry() -> None:
    root = logging.getLogger()
    root_handlers = tuple(root.handlers)
    root_level = root.level
    stream = io.StringIO()

    logger = make_stream_logger(
        "isolated",
        level=logging.INFO,
        stream=stream,
        format_string="%(levelname)s:%(message)s",
    )
    logger.info("step=%d", 3)

    assert stream.getvalue() == "INFO:step=3\n"
    assert not logger.propagate
    assert len(logger.handlers) == 1
    assert logging.getLogger("isolated") is not logger
    assert tuple(root.handlers) == root_handlers
    assert root.level == root_level


def test_stream_loggers_do_not_accumulate_handlers_by_name() -> None:
    first = make_stream_logger("same", level=logging.INFO, stream=io.StringIO(), format_string="%(message)s")
    second = make_stream_logger("same", level=logging.INFO, stream=io.StringIO(), format_string="%(message)s")

    assert first is not second
    assert len(first.handlers) == 1
    assert len(second.handlers) == 1


@pytest.mark.parametrize(
    ("name", "level", "stream", "format_string", "error", "message"),
    [
        (1, logging.INFO, io.StringIO(), "%(message)s", TypeError, "name"),
        ("", logging.INFO, io.StringIO(), "%(message)s", ValueError, "empty"),
        ("x", True, io.StringIO(), "%(message)s", TypeError, "level"),
        ("x", logging.INFO, object(), "%(message)s", TypeError, "stream"),
        ("x", logging.INFO, io.StringIO(), 1, TypeError, "format_string"),
    ],
)
def test_stream_logger_validates_public_boundaries(
    name: object,
    level: object,
    stream: object,
    format_string: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        make_stream_logger(
            cast(str, name),
            level=cast(int, level),
            stream=cast(TextIO, stream),
            format_string=cast(str, format_string),
        )
