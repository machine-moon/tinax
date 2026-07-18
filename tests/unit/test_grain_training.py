from collections.abc import Sequence
from itertools import islice

import grain

from tinax.grain import training_batches


def _int_batch(records: Sequence[int]) -> tuple[int, ...]:
    return tuple(records)


def _training(
    *,
    size: int,
    index: int = 0,
    count: int = 1,
    seed: int = 0,
    shuffle: bool = False,
    num_epochs: int | None = 1,
    batch_size: int = 2,
) -> grain.MapDataset[tuple[int, ...]]:
    return training_batches(
        grain.MapDataset.range(size),
        shard_options=grain.sharding.ShardOptions(index, count, drop_remainder=True),
        seed=seed,
        shuffle=shuffle,
        num_epochs=num_epochs,
        batch_size=batch_size,
        batch_fn=_int_batch,
    )


def test_training_epochs_are_deterministic_and_reseeded() -> None:
    first = list(_training(size=24, seed=2**64, shuffle=True, num_epochs=3, batch_size=4))
    second = list(_training(size=24, seed=2**64, shuffle=True, num_epochs=3, batch_size=4))
    epochs = [first[start : start + 6] for start in range(0, len(first), 6)]

    assert first == second
    assert len(epochs) == 3
    assert all(sorted(record for batch in epoch for record in batch) == list(range(24)) for epoch in epochs)
    assert epochs[0] != epochs[1]


def test_training_processes_are_exactly_disjoint_with_equal_steps() -> None:
    process_batches = [
        list(_training(size=17, index=index, count=3, seed=5, shuffle=True, num_epochs=3))
        for index in range(3)
    ]
    expected = list(
        grain.MapDataset.range(17).seed(5).shuffle().slice(slice(0, 12)).repeat(3, reseed_each_epoch=True)
    )

    assert [len(batches) for batches in process_batches] == [6, 6, 6]
    for epoch_index in range(3):
        expected_epoch = expected[epoch_index * 12 : (epoch_index + 1) * 12]
        for process_index, batches in enumerate(process_batches):
            local_batches = batches[epoch_index * 2 : (epoch_index + 1) * 2]
            local_records = [record for batch in local_batches for record in batch]
            assert local_records == expected_epoch[process_index * 4 : (process_index + 1) * 4]


def test_training_batches_never_cross_finite_epoch_boundaries() -> None:
    process_batches = [
        list(_training(size=11, index=index, count=2, num_epochs=2)) for index in range(2)
    ]

    assert process_batches == [
        [(0, 1), (2, 3), (0, 1), (2, 3)],
        [(4, 5), (6, 7), (4, 5), (6, 7)],
    ]


def test_infinite_training_repeat_is_consumed_safely_by_steps() -> None:
    iterator = iter(_training(size=8, seed=7, shuffle=True, num_epochs=None))
    try:
        sampled = list(islice(iterator, 12))
    finally:
        iterator.close()
    epochs = [sampled[start : start + 4] for start in range(0, len(sampled), 4)]

    assert len(epochs) == 3
    assert all(sorted(record for batch in epoch for record in batch) == list(range(8)) for epoch in epochs)
    assert epochs[0] != epochs[1]
