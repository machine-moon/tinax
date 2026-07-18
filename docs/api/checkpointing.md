# Checkpointing

`tinax.checkpointing` owns Orbax V1 checkpointables and explicit abstract restoration targets. It atomically saves related resume-critical values together and returns asynchronous responses for callers to complete.

See the [Training Checkpoints](../design.md#training-checkpoints) guide for task-oriented usage.

::: tinax.checkpointing.save_checkpointables

::: tinax.checkpointing.load_checkpointables

::: tinax.checkpointing.abstract_restore_target

::: tinax.checkpointing.TrainingCheckpoint

::: tinax.checkpointing.TrainingCheckpointNames

::: tinax.checkpointing.save_training_checkpoint

::: tinax.checkpointing.load_training_checkpoint

::: tinax.checkpointing.validate_checkpointable_name

## Legacy (Orbax V0)

!!! warning
    The `tinax.checkpointing.legacy.v0` namespace is retained only for explicit Orbax V0 compatibility work. Do not use it for new checkpoints.

::: tinax.checkpointing.legacy.v0.save_legacy_v0_pytree

::: tinax.checkpointing.legacy.v0.load_legacy_v0_pytree

::: tinax.checkpointing.legacy.v0.save_legacy_v0_grain_iterator

::: tinax.checkpointing.legacy.v0.load_legacy_v0_grain_iterator
