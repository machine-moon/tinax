"""Deterministic process-aware Grain batch pipelines."""

import sys

import grain as _grain

from tinax.grain._generic import BatchTransform as _BatchTransform


def _positive_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _sharding_options(
    options: object, *, drop_remainder: bool
) -> _grain.sharding.ShardOptions:
    if not isinstance(options, _grain.sharding.ShardOptions):
        raise TypeError("shard_options must be a grain.sharding.ShardOptions")
    shard_index = options.shard_index
    shard_count = options.shard_count
    if not isinstance(shard_index, int) or isinstance(shard_index, bool):
        raise TypeError("shard_options.shard_index must be an integer")
    if not isinstance(shard_count, int) or isinstance(shard_count, bool):
        raise TypeError("shard_options.shard_count must be an integer")
    if shard_count < 1:
        raise ValueError("shard_options.shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_options.shard_index must be in [0, shard_count)")
    if not isinstance(options.drop_remainder, bool):
        raise TypeError("shard_options.drop_remainder must be a bool")
    if options.drop_remainder is not drop_remainder:
        required = "True" if drop_remainder else "False"
        raise ValueError(f"shard_options.drop_remainder must be {required}")
    return options


def _finite_length[T](dataset: _grain.MapDataset[T]) -> int:
    length = len(dataset)
    if length == sys.maxsize:
        raise ValueError("dataset must describe one finite epoch")
    return length


def _shard_dataset[T](
    dataset: _grain.MapDataset[T], options: _grain.sharding.ShardOptions
) -> _grain.MapDataset[T]:
    records_per_shard, remainder = divmod(len(dataset), options.shard_count)
    start = records_per_shard * options.shard_index
    local_size = records_per_shard
    if not options.drop_remainder:
        start += min(options.shard_index, remainder)
        local_size += options.shard_index < remainder
    return dataset.slice(slice(start, start + local_size))


def _validate_dataset(dataset: object) -> None:
    if not isinstance(dataset, _grain.MapDataset):
        raise TypeError("dataset must be a grain.MapDataset")


def _validate_batch_fn(batch_fn: object) -> None:
    if not callable(batch_fn):
        raise TypeError("batch_fn must be callable")


def training_batches[T, BatchT](
    dataset: _grain.MapDataset[T],
    *,
    shard_options: _grain.sharding.ShardOptions,
    seed: int,
    shuffle: bool,
    num_epochs: int | None,
    batch_size: int,
    batch_fn: _BatchTransform[T, BatchT],
) -> _grain.MapDataset[BatchT]:
    """Build deterministic full local batches with equal process step counts in every training epoch.

    Args:
        dataset: Finite ``grain.MapDataset`` describing one epoch of records.
        shard_options: Per-process shard selection. ``drop_remainder`` must be ``True``
            so every process yields the same number of steps.
        seed: Integer seed for deterministic ordering and shuffling.
        shuffle: Whether to shuffle records within each epoch.
        num_epochs: Number of epochs to repeat, or ``None`` for an unbounded stream.
        batch_size: Local (per-process) batch size. Must be positive.
        batch_fn: Callable collating a sequence of records into one batch.

    Returns:
        A ``grain.MapDataset`` of collated batches, sharded to this process and
        repeated for ``num_epochs`` with per-epoch reseeding.

    Raises:
        TypeError: If ``dataset`` is not a ``grain.MapDataset``, ``shard_options`` is
            not a ``grain.sharding.ShardOptions``, ``seed`` is not an integer,
            ``shuffle`` is not a bool, ``batch_fn`` is not callable, or a size is a
            boolean.
        ValueError: If ``shard_options.drop_remainder`` is not ``True``, a size is not
            positive, the dataset is not one finite epoch, or it holds fewer than one
            complete global batch.
    """
    _validate_dataset(dataset)
    options = _sharding_options(shard_options, drop_remainder=True)
    batch_size = _positive_integer("batch_size", batch_size)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a bool")
    if num_epochs is not None:
        num_epochs = _positive_integer("num_epochs", num_epochs)
    _validate_batch_fn(batch_fn)

    epoch_size = _finite_length(dataset)
    global_batch_size = batch_size * options.shard_count
    usable_size = epoch_size - epoch_size % global_batch_size
    if usable_size == 0:
        raise ValueError("dataset must contain at least one complete global batch")

    ordered = dataset.seed(seed)
    if shuffle:
        ordered = ordered.shuffle()
    truncated = ordered.slice(slice(0, usable_size))
    local = _shard_dataset(truncated, options)
    batches = local.batch(batch_size, drop_remainder=True, batch_fn=batch_fn)
    return batches.repeat(num_epochs, reseed_each_epoch=True)


def evaluation_batches[T, BatchT](
    dataset: _grain.MapDataset[T],
    *,
    shard_options: _grain.sharding.ShardOptions,
    batch_size: int,
    batch_fn: _BatchTransform[T, BatchT],
) -> _grain.MapDataset[BatchT]:
    """Build one ordered finite pass that keeps remainder records and may have unequal process step counts.

    Args:
        dataset: Finite ``grain.MapDataset`` describing the evaluation records.
        shard_options: Per-process shard selection. ``drop_remainder`` must be
            ``False`` so remainder records are kept.
        batch_size: Local (per-process) batch size. Must be positive.
        batch_fn: Callable collating a sequence of records into one batch.

    Returns:
        A ``grain.MapDataset`` of collated batches for this process's shard, in order
        and without dropping the final partial batch.

    Raises:
        TypeError: If ``dataset`` is not a ``grain.MapDataset``, ``shard_options`` is
            not a ``grain.sharding.ShardOptions``, ``batch_fn`` is not callable, or
            ``batch_size`` is a boolean.
        ValueError: If ``shard_options.drop_remainder`` is not ``False``, ``batch_size``
            is not positive, or the dataset is not one finite epoch.
    """
    _validate_dataset(dataset)
    options = _sharding_options(shard_options, drop_remainder=False)
    batch_size = _positive_integer("batch_size", batch_size)
    _validate_batch_fn(batch_fn)
    _finite_length(dataset)
    local = _shard_dataset(dataset, options)
    return local.batch(batch_size, drop_remainder=False, batch_fn=batch_fn)
