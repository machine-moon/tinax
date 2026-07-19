# Data

`tinax.data` provides a read/write interface, deterministic process-aware training and evaluation pipelines, dataset splitting, random-access sources, and explicit multiprocessing worker lifetime.

Grain multiprocessing requires caller-parsed Abseil flags. Tinax validates this requirement but never mutates global flags.

Both the ArrayRecord and Parquet pairs return the same random-access `grain.MapDataset`, so they compose identically with `split_by_column`, `split_random`, `training_batches`, and `evaluation_batches`. `read_array_record`/`write_array_record` need the optional `array-record` package installed separately; `pyarrow` for the Parquet pair is a core dependency.

::: tinax.data.read_array_record

::: tinax.data.write_array_record

::: tinax.data.read_parquet_record

::: tinax.data.write_parquet_record

::: tinax.data.split_by_column

::: tinax.data.split_random

::: tinax.data.InMemorySource

::: tinax.data.training_batches

::: tinax.data.evaluation_batches

::: tinax.data.open_multiprocessing_iterator
