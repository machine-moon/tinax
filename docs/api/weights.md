# Weights

`tinax.weights` owns tensor manifests and bounded Safetensors interchange. Loading inspects manifests and byte budgets before tensor materialization. Serialization accepts explicit host tensors and never silently gathers global JAX arrays.

See the [Safetensors Interchange](../design.md#safetensors-interchange) guide for task-oriented usage.

::: tinax.weights.TensorManifest

::: tinax.weights.TensorRule

::: tinax.weights.inspect_safetensors

::: tinax.weights.load_safetensors

::: tinax.weights.save_safetensors

::: tinax.weights.SafetensorsInfo

::: tinax.weights.TensorInfo

::: tinax.weights.LoadedSafetensors
