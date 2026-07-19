# Flax NNX

`tinax.nn` captures independent graph snapshots, restores them with explicit copy policy, and clones mutable variables while preserving graph aliases and concrete module types.

```python
from tinax.nn import clone_graph, restore_graph, snapshot_graph

snapshot = snapshot_graph(model)
independent_model = restore_graph(snapshot, copy=True)
clone = clone_graph(model)
```

::: tinax.nn.snapshot_graph

::: tinax.nn.restore_graph

::: tinax.nn.clone_graph
