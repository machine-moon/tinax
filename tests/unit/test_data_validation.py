from collections.abc import Callable, Sequence
from typing import cast

import grain
import pytest

from tinax.data import evaluation_batches, training_batches


def _int_batch(records: Sequence[int]) -> tuple[int, ...]:
    return tuple(records)


def _training(
    *,
    size: int = 4,
    options: object | None = None,
    seed: object = 0,
    shuffle: object = False,
    num_epochs: object = 1,
    batch_size: object = 2,
) -> grain.MapDataset[tuple[int, ...]]:
    if options is None:
        options = grain.sharding.ShardOptions(0, 1, drop_remainder=True)
    return training_batches(
        grain.MapDataset.range(size),
        shard_options=cast(grain.sharding.ShardOptions, options),
        seed=cast(int, seed),
        shuffle=cast(bool, shuffle),
        num_epochs=cast(int | None, num_epochs),
        batch_size=cast(int, batch_size),
        batch_fn=_int_batch,
    )


def _evaluation(
    *, size: int, index: int = 0, count: int = 1, batch_fn: object = _int_batch
) -> grain.MapDataset[tuple[int, ...]]:
    return evaluation_batches(
        grain.MapDataset.range(size),
        shard_options=grain.sharding.ShardOptions(index, count, drop_remainder=False),
        batch_size=2,
        batch_fn=cast(Callable[[Sequence[int]], tuple[int, ...]], batch_fn),
    )


def test_evaluation_is_ordered_complete_uneven_and_empty_safe() -> None:
    process_batches = [list(_evaluation(size=8, index=index, count=3)) for index in range(3)]

    assert process_batches == [[(0, 1), (2,)], [(3, 4), (5,)], [(6, 7)]]
    assert [len(batches) for batches in process_batches] == [2, 2, 1]
    assert [record for batches in process_batches for batch in batches for record in batch] == list(range(8))
    assert list(_evaluation(size=0)) == []


def test_training_rejects_empty_data_and_an_incomplete_global_batch() -> None:
    with pytest.raises(ValueError, match="at least one complete global batch"):
        _training(size=0)
    with pytest.raises(ValueError, match="at least one complete global batch"):
        _training(size=5, options=grain.sharding.ShardOptions(0, 3, drop_remainder=True))


def test_pipeline_remainder_policies_are_not_interchangeable() -> None:
    with pytest.raises(ValueError, match="drop_remainder must be True"):
        _training(options=grain.sharding.ShardOptions(0, 1, drop_remainder=False))
    with pytest.raises(ValueError, match="drop_remainder must be False"):
        evaluation_batches(
            grain.MapDataset.range(4),
            shard_options=grain.sharding.ShardOptions(0, 1, drop_remainder=True),
            batch_size=2,
            batch_fn=_int_batch,
        )


def test_training_rejects_bool_counts_and_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        _training(seed=True)
    with pytest.raises(TypeError, match="num_epochs must be an integer"):
        _training(num_epochs=True)
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        _training(batch_size=True)


def test_pipeline_validates_flags_coordinates_and_callable() -> None:
    with pytest.raises(TypeError, match="shard_options.shard_index must be an integer"):
        _training(options=grain.sharding.ShardOptions(True, 2, drop_remainder=True))
    with pytest.raises(TypeError, match="shuffle must be a bool"):
        _training(shuffle=1)
    with pytest.raises(TypeError, match="batch_fn must be callable"):
        _evaluation(size=4, batch_fn=None)


def test_training_delegates_seed_range_validation_and_rejects_invalid_epochs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _training(seed=-1, shuffle=True)
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        _training(num_epochs=0)


def test_pipelines_reject_already_infinite_input() -> None:
    infinite = grain.MapDataset.range(4).repeat()
    with pytest.raises(ValueError, match="one finite epoch"):
        training_batches(
            infinite,
            shard_options=grain.sharding.ShardOptions(0, 1, drop_remainder=True),
            seed=0,
            shuffle=False,
            num_epochs=1,
            batch_size=2,
            batch_fn=_int_batch,
        )
    with pytest.raises(ValueError, match="one finite epoch"):
        evaluation_batches(
            infinite,
            shard_options=grain.sharding.ShardOptions(0, 1, drop_remainder=False),
            batch_size=2,
            batch_fn=_int_batch,
        )
