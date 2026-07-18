"""Build deterministic sharded Grain pipelines with managed workers."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import grain

from tinax.grain import evaluation_batches, open_multiprocessing_iterator, training_batches

type Batch = tuple[int, ...]
type ShardBatches = tuple[Batch, ...]


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Hold deterministic training, evaluation, and worker-consumed batches."""

    training_by_shard: tuple[ShardBatches, ...]
    evaluation_by_shard: tuple[ShardBatches, ...]
    worker_training_shard: ShardBatches


def _tuple_batch(records: Sequence[int]) -> Batch:
    return tuple(records)


def _training_pipeline(
    dataset: grain.MapDataset[int],
    *,
    shard_options: grain.sharding.ShardOptions,
    seed: int,
    shuffle: bool,
    num_epochs: int,
    batch_size: int,
    batch_fn: Callable[[Sequence[int]], Batch],
) -> grain.MapDataset[Batch]:
    return training_batches(
        dataset,
        shard_options=shard_options,
        seed=seed,
        shuffle=shuffle,
        num_epochs=num_epochs,
        batch_size=batch_size,
        batch_fn=batch_fn,
    )


def _evaluation_pipeline(
    dataset: grain.MapDataset[int],
    *,
    shard_options: grain.sharding.ShardOptions,
    batch_size: int,
    batch_fn: Callable[[Sequence[int]], Batch],
) -> grain.MapDataset[Batch]:
    return evaluation_batches(
        dataset,
        shard_options=shard_options,
        batch_size=batch_size,
        batch_fn=batch_fn,
    )


def _as_batches(dataset: grain.MapDataset[Batch]) -> ShardBatches:
    return tuple(dataset)


def _with_workers(
    dataset: grain.MapDataset[Batch],
    *,
    read_options: grain.ReadOptions,
    num_workers: int,
    per_worker_buffer_size: int,
    worker_init_fn: Callable[[int, int], None] | None,
    sequential_slice: bool,
) -> ShardBatches:
    iterable = dataset.to_iter_dataset(read_options)
    with open_multiprocessing_iterator(
        iterable,
        num_workers=num_workers,
        per_worker_buffer_size=per_worker_buffer_size,
        worker_init_fn=worker_init_fn,
        sequential_slice=sequential_slice,
    ) as iterator:
        return tuple(iterator)


def run_pipeline_example(
    *,
    dataset: grain.MapDataset[int],
    shard_count: int,
    worker_shard_index: int,
    seed: int,
    shuffle: bool,
    num_epochs: int,
    batch_size: int,
    batch_fn: Callable[[Sequence[int]], Batch],
    read_options: grain.ReadOptions,
    num_workers: int,
    per_worker_buffer_size: int,
    worker_init_fn: Callable[[int, int], None] | None,
    sequential_slice: bool,
) -> PipelineResult:
    """Run explicit process shards without discovering process topology."""
    training = tuple(
        _as_batches(
            _training_pipeline(
                dataset,
                shard_options=grain.sharding.ShardOptions(
                    shard_index=shard_index,
                    shard_count=shard_count,
                    drop_remainder=True,
                ),
                seed=seed,
                shuffle=shuffle,
                num_epochs=num_epochs,
                batch_size=batch_size,
                batch_fn=batch_fn,
            )
        )
        for shard_index in range(shard_count)
    )
    evaluation = tuple(
        _as_batches(
            _evaluation_pipeline(
                dataset,
                shard_options=grain.sharding.ShardOptions(
                    shard_index=shard_index,
                    shard_count=shard_count,
                    drop_remainder=False,
                ),
                batch_size=batch_size,
                batch_fn=batch_fn,
            )
        )
        for shard_index in range(shard_count)
    )
    worker_training = _with_workers(
        _training_pipeline(
            dataset,
            shard_options=grain.sharding.ShardOptions(
                shard_index=worker_shard_index,
                shard_count=shard_count,
                drop_remainder=True,
            ),
            seed=seed,
            shuffle=shuffle,
            num_epochs=num_epochs,
            batch_size=batch_size,
            batch_fn=batch_fn,
        ),
        read_options=read_options,
        num_workers=num_workers,
        per_worker_buffer_size=per_worker_buffer_size,
        worker_init_fn=worker_init_fn,
        sequential_slice=sequential_slice,
    )
    return PipelineResult(training, evaluation, worker_training)


def main(argv: Sequence[str]) -> None:
    """Run the deterministic pipeline recipe with one real worker."""
    del argv
    result = run_pipeline_example(
        dataset=grain.MapDataset.range(17),
        shard_count=3,
        worker_shard_index=0,
        seed=2027,
        shuffle=True,
        num_epochs=2,
        batch_size=2,
        batch_fn=_tuple_batch,
        read_options=grain.ReadOptions(num_threads=0, prefetch_buffer_size=0),
        num_workers=1,
        per_worker_buffer_size=1,
        worker_init_fn=None,
        sequential_slice=True,
    )
    steps = tuple(len(batches) for batches in result.training_by_shard)
    print(f"training steps by shard: {steps}")


if __name__ == "__main__":
    from absl import app

    app.run(main)
