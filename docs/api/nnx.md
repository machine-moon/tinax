# Flax NNX

`tinax.nnx` captures independent graph snapshots, restores them with explicit copy policy, and clones mutable variables while preserving graph aliases and concrete module types.

```python
from tinax.nnx import clone_graph, restore_graph, snapshot_graph

snapshot = snapshot_graph(model)
independent_model = restore_graph(snapshot, copy=True)
clone = clone_graph(model)
```

::: tinax.nnx.snapshot_graph

::: tinax.nnx.restore_graph

::: tinax.nnx.clone_graph
