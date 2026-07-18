"""Portable names for Orbax V1 checkpointables."""

from dataclasses import dataclass
from os import PathLike, fspath
from posixpath import normpath
from unicodedata import category, normalize


def validate_checkpointable_name(name: object) -> str:
    """Validate one checkpointable name as a safe portable path component.

    Args:
        name: Candidate checkpointable name to validate.

    Returns:
        The validated ``name``, unchanged.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is empty, a dot component, contains reserved or control
            characters, ends with a space or dot, exceeds 255 UTF-8 bytes, or matches a
            reserved name.
    """
    if not isinstance(name, str):
        raise TypeError("checkpointable names must be strings")
    if not name:
        raise ValueError("checkpointable names must not be empty")
    if name in {".", ".."}:
        raise ValueError("checkpointable names must not be dot path components")
    if any(character in '<>:"/\\|?*' for character in name):
        raise ValueError("checkpointable names must be single portable path components")
    if any(category(character) in {"Cc", "Cs"} for character in name):
        raise ValueError("checkpointable names must not contain control characters")
    if name.endswith((" ", ".")):
        raise ValueError("checkpointable names must not end with a space or dot")
    if len(name.encode("utf-8")) > 255:
        raise ValueError("checkpointable names must not exceed 255 UTF-8 bytes")

    folded_name = name.casefold()
    folded_stem = name.split(".", maxsplit=1)[0].casefold()
    reserved_stems = {"aux", "clock$", "con", "conin$", "conout$", "nul", "prn"}
    reserved_stems.update(f"com{number}" for number in range(1, 10))
    reserved_stems.update(f"lpt{number}" for number in range(1, 10))
    if folded_name in {"_checkpoint_metadata", "auto", "metrics", "orbax.checkpoint"} or folded_stem in reserved_stems:
        raise ValueError(f"reserved checkpointable name: {name!r}")
    return name


def _portable_name_key(name: str) -> str:
    return normalize("NFC", name).casefold()


def _validated_checkpoint_path(path: str | PathLike[str]) -> str:
    value = fspath(path)
    if not isinstance(value, str):
        raise TypeError("checkpoint paths must resolve to strings")
    normalized_path = normpath(value.replace("\\", "/"))
    final_component = normalized_path.rsplit("/", maxsplit=1)[-1]
    if ".orbax-checkpoint-tmp" in final_component.casefold():
        raise ValueError("checkpoint paths must not contain Orbax's reserved temporary suffix")
    return value


@dataclass(frozen=True, slots=True)
class TrainingCheckpointNames:
    """Names for the five independently handled parts of a training checkpoint.

    Attributes:
        model: Checkpointable name for model state.
        optimizer: Checkpointable name for optimizer state.
        rng: Checkpointable name for RNG state.
        auxiliary: Checkpointable name for auxiliary state and step metadata.
        iterator: Checkpointable name for the input iterator state.

    Raises:
        TypeError: If any name is not a string.
        ValueError: If any name is invalid (see ``validate_checkpointable_name``) or the
            names are not portable and distinct.
    """

    model: str = "model"
    optimizer: str = "optimizer"
    rng: str = "rng"
    auxiliary: str = "auxiliary"
    iterator: str = "iterator"

    def __post_init__(self) -> None:
        names = (self.model, self.optimizer, self.rng, self.auxiliary, self.iterator)
        for name in names:
            validate_checkpointable_name(name)
        if len({_portable_name_key(name) for name in names}) != len(names):
            raise ValueError("training checkpointable names must be portable and distinct")
