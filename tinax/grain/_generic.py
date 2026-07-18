"""Type-preserving Grain primitives."""

from collections.abc import Callable, Iterable, Sequence

type BatchTransform[RecordT, BatchT] = Callable[[Sequence[RecordT]], BatchT]


class InMemorySource[T]:
    """Snapshot an outer iterable while preserving references to its possibly mutable records.

    A deterministic random-access source supporting ``len()`` and integer indexing.
    The outer iterable is copied into a tuple at construction, but individual records
    are held by reference and are not deep copied.

    Args:
        records: Iterable of records to snapshot into the source.
        source_id: Stable, non-empty identifier used in source representations.

    Raises:
        TypeError: If ``source_id`` is not a string.
        ValueError: If ``source_id`` is empty.
    """

    __slots__ = ("_records", "_source_id")

    def __init__(self, records: Iterable[T], *, source_id: str) -> None:
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string")
        if not source_id:
            raise ValueError("source_id must not be empty")
        self._records = tuple(records)
        self._source_id = source_id

    @property
    def source_id(self) -> str:
        """Return the caller-assigned stable identity used in source representations."""
        return self._source_id

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> T:
        return self._records[index]

    def __repr__(self) -> str:
        return f"InMemorySource(source_id={self._source_id!r}, size={len(self)})"
