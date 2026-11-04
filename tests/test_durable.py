from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

from miroir.jobs.durable import append_lines, locked, read_lines, rewrite_lines


def test_a_torn_last_line_is_skipped_then_cut_off_by_the_next_append(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_bytes(b'{"a":1}\n{"b":')
    assert read_lines(path) == [b'{"a":1}']
    append_lines(path, [b'{"c":3}'])
    assert path.read_bytes() == b'{"a":1}\n{"c":3}\n'


def test_a_file_that_is_one_torn_line_reads_empty(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_bytes(b"x" * 100_000)  # longer than one read block, no newline at all
    assert read_lines(path) == []
    append_lines(path, [b"y"])
    assert path.read_bytes() == b"y\n"


def test_rewrite_replaces_the_whole_file(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    append_lines(path, [b"1", b"2"])
    rewrite_lines(path, [b"3"])
    assert read_lines(path) == [b"3"]
    assert [p.name for p in tmp_path.iterdir()] == ["a.txt"]


def _hold(path: Path, held: str) -> None:
    with locked(path):
        Path(held).touch()
        time.sleep(0.4)


def test_the_lock_holds_across_processes(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    held = tmp_path / "held"
    other = multiprocessing.get_context("spawn").Process(target=_hold, args=(path, str(held)))
    other.start()
    while not held.exists():
        time.sleep(0.01)
    start = time.monotonic()
    with locked(path):
        waited = time.monotonic() - start
    other.join()
    assert waited >= 0.2
