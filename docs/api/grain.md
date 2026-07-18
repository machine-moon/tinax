# Grain

`tinax.grain` provides deterministic process-aware training and evaluation pipelines, random-access sources, and explicit multiprocessing worker lifetime.

Grain multiprocessing requires caller-parsed Abseil flags. Tinax validates this requirement but never mutates global flags.

::: tinax.grain.InMemorySource

::: tinax.grain.training_batches

::: tinax.grain.evaluation_batches

::: tinax.grain.open_multiprocessing_iterator
