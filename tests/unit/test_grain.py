from collections.abc import Callable, Sequence
from typing import assert_type, cast

import grain
import pytest
from absl import flags

import tinax.grain as grain_api
from tinax.grain import (
    InMemorySource,
    evaluation_batches,
    open_multiprocessing_iterator,
    training_batches,
)
from tinax.grain._generic import BatchTransform


def _text_batch(records: Sequence[str]) -> tuple[str, ...]:
    return tuple(records)


def _training_options() -> grain.sharding.ShardOptions:
    return grain.sharding.ShardOptions(0, 1, drop_remainder=True)


def _evaluation_options() -> grain.sharding.ShardOptions:
    return grain.sharding.ShardOptions(0, 1, drop_remainder=False)


def test_package_exports_stable_grain_primitives() -> None:
    assert grain_api.InMemorySource is InMemorySource
    assert grain_api.training_batches is training_batches
    assert grain_api.evaluation_batches is evaluation_batches
    assert callable(grain_api.open_multiprocessing_iterator)
    assert not hasattr(grain_api, "snapshot_iterator_state")
    assert not hasattr(grain_api, "restore_iterator_state")


def test_in_memory_source_has_stable_identity_and_an_honest_shallow_snapshot() -> None:
    nested = {"values": [1]}
    records = [nested]
    source = InMemorySource(records, source_id="train-records-v1")

    records.append({"values": [9]})
    nested["values"].append(2)

    assert len(source) == 1
    assert source[0] is nested
    assert source[0] == {"values": [1, 2]}
    assert source.source_id == "train-records-v1"
    assert repr(source) == "InMemorySource(source_id='train-records-v1', size=1)"
    assert isinstance(source, grain.sources.RandomAccessDataSource)


def test_in_memory_source_identity_is_explicit_and_validated() -> None:
    with pytest.raises(TypeError, match="source_id must be a string"):
        InMemorySource([1], source_id=cast(str, 7))
    with pytest.raises(ValueError, match="source_id must not be empty"):
        InMemorySource([1], source_id="")


def test_source_and_batch_transforms_preserve_public_type_relationships() -> None:
    source = InMemorySource(["alpha", "beta"], source_id="words-v1")
    assert_type(source, InMemorySource[str])
    assert_type(source[0], str)
    dataset = grain.MapDataset.source(source)
    assert_type(dataset, grain.MapDataset[str])
    batch_fn: BatchTransform[str, tuple[str, ...]] = _text_batch

    training = training_batches(
        dataset,
        shard_options=_training_options(),
        seed=3,
        shuffle=False,
        num_epochs=1,
        batch_size=2,
        batch_fn=batch_fn,
    )
    evaluation = evaluation_batches(
        dataset,
        shard_options=_evaluation_options(),
        batch_size=2,
        batch_fn=batch_fn,
    )

    assert_type(training, grain.MapDataset[tuple[str, ...]])
    assert_type(evaluation, grain.MapDataset[tuple[str, ...]])
    assert list(training) == [("alpha", "beta")]
    assert list(evaluation) == [("alpha", "beta")]


def test_worker_options_are_forwarded_without_mutating_absl_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    before = flags.FLAGS.flag_values_dict()
    received: list[grain.MultiprocessingOptions] = []

    def capture_options(
        dataset: grain.IterDataset[int],
        options: grain.MultiprocessingOptions | None = None,
        worker_init_fn: Callable[[int, int], None] | None = None,
        sequential_slice: bool = False,
    ) -> grain.IterDataset[int]:
        del worker_init_fn, sequential_slice
        assert flags.FLAGS.flag_values_dict() == before
        assert options is not None
        received.append(options)
        return dataset

    monkeypatch.setattr(grain.IterDataset, "mp_prefetch", capture_options)
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    dataset = grain.MapDataset.range(2).to_iter_dataset(read_options)
    with open_multiprocessing_iterator(dataset, num_workers=0) as iterator:
        assert list(iterator) == [0, 1]
    assert received == [grain.MultiprocessingOptions(num_workers=0)]
