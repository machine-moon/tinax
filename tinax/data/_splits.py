"""Validated dataset splitting by a discriminator column or a random ratio."""

from typing import Any as _Any

import grain as _grain


def _validate_dataset(dataset: object) -> None:
    if not isinstance(dataset, _grain.MapDataset):
        raise TypeError("dataset must be a grain.MapDataset")


def _drop_column(dataset: _grain.MapDataset[dict[str, _Any]], column: str) -> _grain.MapDataset[dict[str, _Any]]:
    return dataset.map(lambda row: {key: value for key, value in row.items() if key != column})


def split_by_column(
    dataset: _grain.MapDataset[dict[str, _Any]],
    column: str,
    *,
    values: tuple[str, str] = ("train", "test"),
    drop_column: bool = True,
) -> tuple[_grain.MapDataset[dict[str, _Any]], _grain.MapDataset[dict[str, _Any]]]:
    """Partition ``dataset`` by an exact two-value discriminator column.

    Every row's ``column`` value must be one of ``values``; a row holding anything
    else would otherwise be silently dropped by a naive filter-based split. The returned
    partitions are re-indexed sources with correct ``len`` and indexing, so they compose
    with the length-based ``training_batches`` and ``evaluation_batches`` pipelines.

    Args:
        dataset: Finite ``grain.MapDataset`` of mapping rows.
        column: Name of the column carrying the discriminator value.
        values: The (first, second) partition values.
        drop_column: Whether to drop ``column`` from both returned partitions.

    Returns:
        The two partitions of ``dataset``, in the order given by ``values``.

    Raises:
        TypeError: If ``dataset`` is not a ``grain.MapDataset``, ``column`` is not a
            string, or ``drop_column`` is not a bool.
        ValueError: If ``values`` does not hold two distinct entries, or any row's
            ``column`` value is not one of ``values``.
    """
    _validate_dataset(dataset)
    if not isinstance(column, str):
        raise TypeError("column must be a string")
    if not isinstance(drop_column, bool):
        raise TypeError("drop_column must be a bool")
    first_value, second_value = values
    if first_value == second_value:
        raise ValueError("values must contain two distinct entries")

    first_rows: list[dict[str, _Any]] = []
    second_rows: list[dict[str, _Any]] = []
    for row in dataset:
        value = row[column]
        if value == first_value:
            first_rows.append(row)
        elif value == second_value:
            second_rows.append(row)
        else:
            raise ValueError(f"column {column!r} has value {value!r} outside {values}")

    first = _grain.MapDataset.source(first_rows)
    second = _grain.MapDataset.source(second_rows)
    if drop_column:
        first = _drop_column(first, column)
        second = _drop_column(second, column)
    return first, second


def split_random(
    dataset: _grain.MapDataset[dict[str, _Any]],
    train_ratio: float,
    *,
    seed: int,
    drop_column: str | None = None,
) -> tuple[_grain.MapDataset[dict[str, _Any]], _grain.MapDataset[dict[str, _Any]]]:
    """Shuffle ``dataset`` and cut it into two partitions at ``train_ratio``.

    Args:
        dataset: Finite ``grain.MapDataset`` to split.
        train_ratio: Fraction of records in the first partition; must be strictly
            between 0 and 1 so neither partition is degenerately empty.
        seed: Integer seed for the shuffle.
        drop_column: Optional column name to drop from both partitions afterward.

    Returns:
        The (first, second) partitions.

    Raises:
        TypeError: If ``dataset`` is not a ``grain.MapDataset``, ``train_ratio`` is not
            a float, ``seed`` is not an integer, or ``drop_column`` is not a string or
            ``None``.
        ValueError: If ``train_ratio`` is not strictly between 0 and 1, or the dataset
            holds fewer than two records.
    """
    _validate_dataset(dataset)
    if isinstance(train_ratio, bool) or not isinstance(train_ratio, (int, float)):
        raise TypeError("train_ratio must be a float")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if drop_column is not None and not isinstance(drop_column, str):
        raise TypeError("drop_column must be a string or None")

    size = len(dataset)
    if size < 2:
        raise ValueError("dataset must hold at least two records to split")
    split_index = min(max(round(size * train_ratio), 1), size - 1)

    shuffled = dataset.shuffle(seed=seed)
    first = shuffled[:split_index]
    second = shuffled[split_index:]
    if drop_column is not None:
        first = _drop_column(first, drop_column)
        second = _drop_column(second, drop_column)
    return first, second
