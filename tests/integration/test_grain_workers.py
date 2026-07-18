import multiprocessing
import subprocess
import sys
import time
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import assert_type

import grain
import pytest
from absl import flags
from absl.testing import flagsaver

from tinax.grain import open_multiprocessing_iterator


def _iter_dataset(size: int) -> grain.IterDataset[int]:
    read_options = grain.ReadOptions(num_threads=0, prefetch_buffer_size=0)
    return grain.MapDataset.range(size).to_iter_dataset(read_options)


def _grain_worker_pids() -> set[int]:
    pids: set[int] = set()
    for process in multiprocessing.active_children():
        pid = process.pid
        if pid is not None and process.name.startswith("grain-process-prefetch-"):
            pids.add(pid)
    return pids


def _wait_for_new_workers(existing: set[int], count: int) -> set[int]:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        workers = _grain_worker_pids() - existing
        if len(workers) >= count:
            return workers
        time.sleep(0.05)
    raise AssertionError(f"expected {count} Grain workers")


def _wait_for_workers_to_stop(workers: set[int]) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if workers.isdisjoint(_grain_worker_pids()):
            return
        time.sleep(0.05)
    raise AssertionError(f"Grain workers still active: {workers & _grain_worker_pids()}")


def _record_worker_start(directory: str, worker_index: int, worker_count: int) -> None:
    Path(directory, f"{worker_index}-of-{worker_count}").touch()


def _wait_for_worker_markers(directory: Path, count: int) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if len(list(directory.iterdir())) == count:
            return
        time.sleep(0.05)
    raise AssertionError(f"expected {count} worker markers")


@pytest.fixture
def _parsed_absl_flags() -> Iterator[None]:
    saved_flags = flagsaver.save_flag_values()
    was_parsed = flags.FLAGS.is_parsed()
    if not was_parsed:
        flags.FLAGS.mark_as_parsed()
    try:
        yield
    finally:
        if not was_parsed:
            flags.FLAGS.unparse_flags()
        flagsaver.restore_flag_values(saved_flags)


def test_workers_close_after_normal_exhaustion(_parsed_absl_flags: None) -> None:
    existing = _grain_worker_pids()
    workers: set[int] = set()

    with open_multiprocessing_iterator(_iter_dataset(8), num_workers=1) as iterator:
        first = next(iterator)
        workers = _wait_for_new_workers(existing, 1)
        assert [first, *iterator] == list(range(8))

    assert workers
    _wait_for_workers_to_stop(workers)
    with pytest.raises(ValueError, match="closed iterator"):
        next(iterator)


def test_workers_close_on_early_context_exit_and_receive_options(
    tmp_path: Path, _parsed_absl_flags: None
) -> None:
    existing = _grain_worker_pids()
    worker_init_fn = partial(_record_worker_start, str(tmp_path))
    workers: set[int] = set()

    with open_multiprocessing_iterator(
        _iter_dataset(1_000),
        num_workers=2,
        per_worker_buffer_size=2,
        worker_init_fn=worker_init_fn,
        sequential_slice=True,
    ) as iterator:
        next(iterator)
        next(iterator)
        workers = _wait_for_new_workers(existing, 2)
        _wait_for_worker_markers(tmp_path, 2)

    assert workers
    _wait_for_workers_to_stop(workers)


def test_workers_close_when_context_body_raises(_parsed_absl_flags: None) -> None:
    existing = _grain_worker_pids()
    workers: set[int] = set()

    with pytest.raises(RuntimeError, match="body failed"):
        with open_multiprocessing_iterator(_iter_dataset(1_000), num_workers=1) as iterator:
            next(iterator)
            workers = _wait_for_new_workers(existing, 1)
            raise RuntimeError("body failed")

    assert workers
    _wait_for_workers_to_stop(workers)


def test_worker_options_reject_bool_as_integer() -> None:
    dataset = _iter_dataset(2)
    with pytest.raises(TypeError, match="num_workers must be an integer"):
        with open_multiprocessing_iterator(dataset, num_workers=True):
            pass
    with pytest.raises(TypeError, match="per_worker_buffer_size must be an integer"):
        with open_multiprocessing_iterator(dataset, num_workers=0, per_worker_buffer_size=True):
            pass


def test_open_iterator_preserves_record_type() -> None:
    context = open_multiprocessing_iterator(_iter_dataset(2), num_workers=0)
    with context as iterator:
        assert_type(iterator, grain.DatasetIterator[int])
        assert list(iterator) == [0, 1]


def test_installed_example_bootstraps_abseil_before_starting_workers() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "examples.grain.deterministic_pipelines"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "training steps by shard: (4, 4, 4)" in completed.stdout


def test_clean_interpreter_gets_a_clear_unparsed_flags_error() -> None:
    code = """
import grain
from tinax.grain import open_multiprocessing_iterator
dataset = grain.MapDataset.range(2).to_iter_dataset(grain.ReadOptions(num_threads=0, prefetch_buffer_size=0))
try:
    with open_multiprocessing_iterator(dataset, num_workers=1):
        pass
except RuntimeError as error:
    assert 'absl.app.run' in str(error)
else:
    raise AssertionError('unparsed Abseil flags were accepted')
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
