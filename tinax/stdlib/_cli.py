"""Explicit argparse conversion into caller-owned configuration."""

from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType


def parse_config[T](
    parser: ArgumentParser,
    argv: Sequence[str],
    factory: Callable[[Mapping[str, object]], T],
) -> T:
    """Parse explicit arguments and pass a shallow read-only value mapping to a configuration factory.

    Args:
        parser: Configured ``argparse.ArgumentParser``. ``sys.argv`` is never read.
        argv: Argument strings excluding the executable name (an Abseil entry point
            passes ``argv[1:]``).
        factory: Callable invoked with a shallow read-only mapping of parsed values;
            its return value is passed through unchanged.

    Returns:
        The object produced by ``factory``.

    Raises:
        TypeError: If ``parser`` is not an ``ArgumentParser``, ``argv`` is not a
            sequence of strings, or ``factory`` is not callable.
    """
    if not isinstance(parser, ArgumentParser):
        raise TypeError("parser must be an argparse.ArgumentParser")
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of strings")
    arguments = tuple(argv)
    if any(not isinstance(argument, str) for argument in arguments):
        raise TypeError("argv must contain only strings")
    if not callable(factory):
        raise TypeError("factory must be callable")
    values = MappingProxyType(vars(parser.parse_args(arguments)).copy())
    return factory(values)
