"""Explicit Grain multiprocessing iterator lifecycle."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import grain as _grain


def _nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@contextmanager
def open_multiprocessing_iterator[T](
    dataset: _grain.IterDataset[T],
    *,
    num_workers: int,
    per_worker_buffer_size: int = 1,
    worker_init_fn: Callable[[int, int], None] | None = None,
    sequential_slice: bool = False,
) -> Iterator[_grain.DatasetIterator[T]]:
    """Open a Grain multiprocessing iterator and close every worker when the context exits.

    A context manager yielding a dataset iterator backed by worker processes; on exit
    the iterator and all workers are closed.

    Args:
        dataset: ``grain.IterDataset`` to iterate with prefetching workers.
        num_workers: Number of worker processes. ``0`` runs in the main process.
        per_worker_buffer_size: Prefetch buffer size per worker. Must be positive.
        worker_init_fn: Optional callable run in each worker as
            ``worker_init_fn(worker_index, worker_count)``.
        sequential_slice: Whether workers slice the dataset sequentially.

    Yields:
        A ``grain.DatasetIterator`` over the prefetched dataset.

    Raises:
        TypeError: If ``dataset`` is not a ``grain.IterDataset``, an integer argument
            is a boolean, ``worker_init_fn`` is not callable or ``None``, or
            ``sequential_slice`` is not a bool.
        ValueError: If ``num_workers`` is negative or ``per_worker_buffer_size`` is not
            positive.
        RuntimeError: If ``num_workers`` is nonzero but Abseil flags are not parsed;
            enter through ``absl.app.run``.
    """
    if not isinstance(dataset, _grain.IterDataset):
        raise TypeError("dataset must be a grain.IterDataset")
    num_workers = _nonnegative_integer("num_workers", num_workers)
    per_worker_buffer_size = _nonnegative_integer(
        "per_worker_buffer_size", per_worker_buffer_size
    )
    if per_worker_buffer_size == 0:
        raise ValueError("per_worker_buffer_size must be positive")
    if worker_init_fn is not None and not callable(worker_init_fn):
        raise TypeError("worker_init_fn must be callable or None")
    if not isinstance(sequential_slice, bool):
        raise TypeError("sequential_slice must be a bool")
    if num_workers:
        from absl import flags as _flags

        if not _flags.FLAGS.is_parsed():
            raise RuntimeError("Grain multiprocessing requires parsed Abseil flags; enter through absl.app.run")

    options = _grain.MultiprocessingOptions(
        num_workers=num_workers,
        per_worker_buffer_size=per_worker_buffer_size,
    )
    prefetched = dataset.mp_prefetch(
        options,
        worker_init_fn=worker_init_fn,
        sequential_slice=sequential_slice,
    )
    iterator = iter(prefetched)
    try:
        yield iterator
    finally:
        iterator.close()
