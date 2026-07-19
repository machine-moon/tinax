"""Round-trip rows through a Parquet file with tinax.data."""

import tempfile
from pathlib import Path

from tinax.data import read_parquet_record, split_by_column, write_parquet_record


def main() -> None:
    """Write rows to a Parquet file, read them back, then split by a discriminator column."""
    rows = [{"x": value, "split": "train" if value % 3 else "test"} for value in range(12)]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rows.parquet"
        write_parquet_record(path, rows)
        dataset = read_parquet_record(path)

        train, test = split_by_column(dataset, "split")
        print(f"train_rows={len(train)} test_rows={len(test)}")


if __name__ == "__main__":
    main()
