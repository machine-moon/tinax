from typing import cast

import pytest

from tinax.data import read_array_record, write_array_record


def test_array_record_round_trips_json_rows(tmp_path) -> None:
    path = tmp_path / "rows.arrayrecord"
    rows = [{"id": 0, "label": 1}, {"id": 1, "label": 0}]

    write_array_record(path, rows)
    loaded = read_array_record(path)

    assert list(loaded) == rows


def test_array_record_write_rejects_non_positive_group_size(tmp_path) -> None:
    path = tmp_path / "rows.arrayrecord"
    with pytest.raises(ValueError, match="group_size"):
        write_array_record(path, [{"id": 0}], group_size=0)


def test_array_record_write_rejects_boolean_group_size(tmp_path) -> None:
    path = tmp_path / "rows.arrayrecord"
    with pytest.raises(TypeError, match="group_size must be an integer"):
        write_array_record(path, [{"id": 0}], group_size=cast(int, True))


def test_array_record_read_reports_a_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No ArrayRecord file"):
        read_array_record(tmp_path / "absent.arrayrecord")


def test_array_record_rejects_non_path_arguments(tmp_path) -> None:
    with pytest.raises(TypeError, match="path must be"):
        read_array_record(cast(str, 7))
    with pytest.raises(TypeError, match="path must be"):
        write_array_record(cast(str, 7), [{"id": 0}])
