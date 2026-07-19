from typing import cast

import pytest

from tinax.data import read_parquet_record, write_parquet_record


def test_parquet_record_round_trips_rows(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    rows = [{"id": 0, "label": 1}, {"id": 1, "label": 0}]

    write_parquet_record(path, rows)
    loaded = read_parquet_record(path)

    assert list(loaded) == rows
    assert len(loaded) == 2


def test_parquet_record_write_rejects_an_empty_sequence(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    with pytest.raises(ValueError, match="not be empty"):
        write_parquet_record(path, [])


def test_parquet_record_write_rejects_non_sequence_or_non_mapping_records(tmp_path) -> None:
    path = tmp_path / "rows.parquet"
    with pytest.raises(TypeError, match="sequence of mappings"):
        write_parquet_record(path, cast(list, iter([{"id": 0}])))
    with pytest.raises(TypeError, match="sequence of mappings"):
        write_parquet_record(path, [cast(dict, ["not", "a", "mapping"])])


def test_parquet_record_read_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No Parquet file"):
        read_parquet_record(tmp_path / "absent.parquet")


def test_parquet_record_rejects_non_path_arguments(tmp_path) -> None:
    with pytest.raises(TypeError, match="path must be"):
        read_parquet_record(cast(str, 7))
    with pytest.raises(TypeError, match="path must be"):
        write_parquet_record(cast(str, 7), [{"id": 0}])
