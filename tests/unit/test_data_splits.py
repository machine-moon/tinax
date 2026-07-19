from typing import cast

import grain
import pytest

from tinax.data import split_by_column, split_random


def _rows(*splits: str) -> grain.MapDataset[dict[str, object]]:
    records: list[dict[str, object]] = [{"id": index, "split": split} for index, split in enumerate(splits)]
    return grain.MapDataset.source(records).map(lambda row: row)


def test_split_by_column_partitions_and_drops_the_column() -> None:
    dataset = _rows("train", "train", "test")

    train, test = split_by_column(dataset, "split")

    assert list(train) == [{"id": 0}, {"id": 1}]
    assert list(test) == [{"id": 2}]


def test_split_by_column_partitions_have_correct_length_and_indexing() -> None:
    dataset = _rows("train", "test", "train", "test")

    train, test = split_by_column(dataset, "split")

    assert len(train) == 2
    assert len(test) == 2
    assert train[0] == {"id": 0}
    assert train[1] == {"id": 2}
    assert test[1] == {"id": 3}


def test_split_by_column_can_keep_the_column() -> None:
    dataset = _rows("train", "test")

    train, test = split_by_column(dataset, "split", drop_column=False)

    assert list(train) == [{"id": 0, "split": "train"}]
    assert list(test) == [{"id": 1, "split": "test"}]


def test_split_by_column_rejects_a_row_outside_the_two_values() -> None:
    dataset = _rows("train", "val")

    with pytest.raises(ValueError, match=r"outside \('train', 'test'\)"):
        split_by_column(dataset, "split")


def test_split_by_column_rejects_duplicate_values() -> None:
    dataset = _rows("train")

    with pytest.raises(ValueError, match="two distinct entries"):
        split_by_column(dataset, "split", values=("train", "train"))


def test_split_by_column_rejects_bad_types() -> None:
    dataset = _rows("train")

    with pytest.raises(TypeError, match="grain.MapDataset"):
        split_by_column(cast(grain.MapDataset, [1, 2]), "split")
    with pytest.raises(TypeError, match="column must be a string"):
        split_by_column(dataset, cast(str, 7))
    with pytest.raises(TypeError, match="drop_column must be a bool"):
        split_by_column(dataset, "split", drop_column=cast(bool, 1))


def _numbered(size: int) -> grain.MapDataset[dict[str, object]]:
    records: list[dict[str, object]] = [{"id": index} for index in range(size)]
    return grain.MapDataset.source(records).map(lambda row: row)


def test_split_random_partitions_by_ratio_and_is_deterministic() -> None:
    dataset = _numbered(10)

    first_a, second_a = split_random(dataset, 0.8, seed=0)
    first_b, second_b = split_random(dataset, 0.8, seed=0)

    assert len(first_a) == 8
    assert len(second_a) == 2
    assert list(first_a) == list(first_b)
    assert list(second_a) == list(second_b)
    assert {row["id"] for row in first_a} | {row["id"] for row in second_a} == set(range(10))


def test_split_random_never_produces_a_degenerate_empty_partition() -> None:
    dataset = _numbered(10)

    first, second = split_random(dataset, 0.999, seed=0)

    assert len(first) == 9
    assert len(second) == 1


def test_split_random_can_drop_a_column() -> None:
    dataset = _rows("train", "test", "train")

    first, second = split_random(dataset, 0.5, seed=1, drop_column="split")

    assert all("split" not in row for row in first)
    assert all("split" not in row for row in second)


def test_split_random_rejects_out_of_range_ratio() -> None:
    dataset = _numbered(4)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_random(dataset, 0.0, seed=0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_random(dataset, 1.0, seed=0)


def test_split_random_rejects_too_small_a_dataset() -> None:
    with pytest.raises(ValueError, match="at least two records"):
        split_random(_numbered(1), 0.5, seed=0)


def test_split_random_rejects_bad_types() -> None:
    dataset = _numbered(4)

    with pytest.raises(TypeError, match="train_ratio must be a float"):
        split_random(dataset, cast(float, True), seed=0)
    with pytest.raises(TypeError, match="seed must be an integer"):
        split_random(dataset, 0.5, seed=cast(int, True))
    with pytest.raises(TypeError, match="drop_column must be a string"):
        split_random(dataset, 0.5, seed=0, drop_column=cast(str, 7))
