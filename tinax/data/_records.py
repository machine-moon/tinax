"""Read and write ArrayRecord files as lazy JSON-decoded Grain datasets."""

import importlib as _importlib
import json as _json
from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from os import PathLike as _PathLike
from pathlib import Path as _Path
from typing import Any as _Any
from typing import cast as _cast

import grain as _grain


def read_array_record(path: str | _PathLike[str]) -> _grain.MapDataset:
    """Load one ArrayRecord file's JSON-encoded rows into a lazy, randomly-accessible dataset.

    Args:
        path: Path to an existing ArrayRecord file.

    Returns:
        A ``grain.MapDataset`` decoding each row from JSON on access.

    Raises:
        TypeError: If ``path`` is not a string or ``os.PathLike``.
        FileNotFoundError: If no file exists at ``path``.
    """
    if not isinstance(path, (str, _PathLike)):
        raise TypeError("path must be a string or os.PathLike")
    resolved = _Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"No ArrayRecord file at '{resolved}'")

    def decode(raw: bytes) -> _Any:
        return _json.loads(raw.decode("utf-8"))

    source = _grain.sources.ArrayRecordDataSource(str(resolved))
    validated_source = _cast(_grain.sources.RandomAccessDataSource, source)
    return _grain.MapDataset.source(validated_source).map(decode)


def write_array_record(
    path: str | _PathLike[str],
    records: _Iterable[_Mapping[str, _Any]],
    *,
    group_size: int = 1,
) -> None:
    """Serialize an iterable of JSON-encodable mapping rows into an ArrayRecord file.

    Args:
        path: Destination file path.
        records: Iterable of JSON-encodable mappings, one per row.
        group_size: Number of records grouped per compressed chunk. Must be a positive
            integer.

    Raises:
        TypeError: If ``path`` is not a string or ``os.PathLike``, or ``group_size`` is
            not an integer.
        ValueError: If ``group_size`` is not positive.
        ImportError: If ``array-record`` is not installed.
    """
    if not isinstance(path, (str, _PathLike)):
        raise TypeError("path must be a string or os.PathLike")
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        raise TypeError("group_size must be an integer")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    try:
        array_record = _importlib.import_module("array_record.python.array_record_module")
    except ImportError as error:
        raise ImportError("ArrayRecord support needs the array-record package to be installed.") from error

    def encode(record: _Mapping[str, _Any]) -> bytes:
        return _json.dumps(record).encode("utf-8")

    writer = array_record.ArrayRecordWriter(str(path), f"group_size:{group_size}")
    try:
        for record in records:
            writer.write(encode(record))
    finally:
        writer.close()
