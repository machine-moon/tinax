"""Atomic destination handling for weight files."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp


@contextmanager
def atomic_write_path(destination: Path, *, overwrite: bool) -> Iterator[Path]:
    """Yield a sibling temporary path and atomically publish it on success."""
    destination = _validate_destination(destination, overwrite=overwrite)
    descriptor, temporary_name = mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        with temporary.open("rb+") as temporary_file:
            os.fsync(temporary_file.fileno())
        _publish(temporary, destination, overwrite=overwrite)
        _sync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_destination(destination: Path, *, overwrite: bool) -> Path:
    if destination.is_symlink():
        raise ValueError(f"destination must not be a symbolic link: {destination}")
    if destination.exists():
        if destination.is_dir():
            raise IsADirectoryError(destination)
        if not destination.is_file():
            raise ValueError(f"destination must be a regular file or absent: {destination}")
        if not overwrite:
            raise FileExistsError(f"destination already exists: {destination}")
    if not destination.parent.exists():
        raise FileNotFoundError(f"destination directory does not exist: {destination.parent}")
    if not destination.parent.is_dir():
        raise NotADirectoryError(destination.parent)
    return destination


def _publish(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.is_symlink():
        raise ValueError(f"destination must not be a symbolic link: {destination}")
    if overwrite:
        if destination.exists() and not destination.is_file():
            if destination.is_dir():
                raise IsADirectoryError(destination)
            raise ValueError(f"destination must be a regular file or absent: {destination}")
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination, follow_symlinks=False)
    except FileExistsError as error:
        if destination.is_symlink():
            raise ValueError(f"destination must not be a symbolic link: {destination}") from error
        raise FileExistsError(f"destination already exists: {destination}") from error


def _sync_directory(directory: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
