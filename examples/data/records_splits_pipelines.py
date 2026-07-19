"""Write ArrayRecords, split by a column, and build batches with tinax.data."""

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import grain

from tinax.data import (
    evaluation_batches,
    open_multiprocessing_iterator,
    read_array_record,
    split_by_column,
    training_batches,
    write_array_record,
)


def _collate(rows: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(int(row["x"]) for row in rows)


def main() -> None:
    """Round-trip rows through an ArrayRecord file, split them, then batch each partition."""
    rows = [{"x": value, "split": "train" if value % 3 else "test"} for value in range(12)]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.array_record"
        write_array_record(path, rows)
        dataset = read_array_record(path)

        train, test = split_by_column(dataset, "split")
        options = grain.sharding.ShardOptions(shard_index=0, shard_count=1, drop_remainder=True)

        training = training_batches(
            train, shard_options=options, seed=0, shuffle=True, num_epochs=1, batch_size=2, batch_fn=_collate
        )
        evaluation = evaluation_batches(
            test,
            shard_options=grain.sharding.ShardOptions(shard_index=0, shard_count=1, drop_remainder=False),
            batch_size=2,
            batch_fn=_collate,
        )
        print(f"training_steps={len(training)} evaluation_steps={len(evaluation)}")

        with open_multiprocessing_iterator(training.to_iter_dataset(), num_workers=0) as iterator:
            print(f"first_training_batch={next(iter(iterator))}")


if __name__ == "__main__":
    main()
