"""Read and write Parquet files as lazy, randomly-accessible Grain datasets."""

from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from os import PathLike as _PathLike
from pathlib import Path as _Path
from typing import Any as _Any

import grain as _grain
import pyarrow as _pa
import pyarrow.parquet as _pq


def read_parquet_record(path: str | _PathLike[str]) -> _grain.MapDataset:
    """Load one Parquet file's rows into a lazy, randomly-accessible dataset.

    Args:
        path: Path to an existing Parquet file.

    Returns:
        A ``grain.MapDataset`` over the file's rows, each a ``dict[str, Any]``.

    Raises:
        TypeError: If ``path`` is not a string or ``os.PathLike``.
        FileNotFoundError: If no file exists at ``path``.
    """
    if not isinstance(path, (str, _PathLike)):
        raise TypeError("path must be a string or os.PathLike")
    resolved = _Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"No Parquet file at '{resolved}'")

    rows = _pq.read_table(str(resolved)).to_pylist()
    return _grain.MapDataset.source(rows)


def write_parquet_record(
    path: str | _PathLike[str],
    records: _Sequence[_Mapping[str, _Any]],
) -> None:
    """Serialize a sequence of mapping rows into a Parquet file.

    Unlike ``write_array_record``, ``records`` must be a finite, in-memory sequence:
    Parquet's columnar layout requires the full column set up front and cannot stream
    row by row.

    Args:
        path: Destination file path. Overwritten if it already exists.
        records: Non-empty sequence of mapping rows sharing one column schema.

    Raises:
        TypeError: If ``path`` is not a string or ``os.PathLike``, or ``records`` is not
            a sequence of mappings.
        ValueError: If ``records`` is empty.
    """
    if not isinstance(path, (str, _PathLike)):
        raise TypeError("path must be a string or os.PathLike")
    if not isinstance(records, _Sequence) or isinstance(records, (str, bytes)):
        raise TypeError("records must be a sequence of mappings")
    row_tuple = tuple(records)
    if not row_tuple:
        raise ValueError("records must not be empty")
    if any(not isinstance(row, _Mapping) for row in row_tuple):
        raise TypeError("records must be a sequence of mappings")

    table = _pa.Table.from_pylist(list(row_tuple))
    _pq.write_table(table, str(path))
