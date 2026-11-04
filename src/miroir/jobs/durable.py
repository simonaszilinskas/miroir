"""Files that hold paid work and must stay whole through a crash: lines that only grow.

A crash can leave the last line of an append half written. Readers skip a last line without
its newline, and the next append cuts it off first, so it never glues onto a new line. Every
write is flushed to disk (`fsync`) before it returns. A rewrite goes to a temporary file that
then replaces the old one: a reader sees the old file or the new one, never a mix.
`locked` holds a lock file, so several processes can share one file (the spend ledger).
"""

from __future__ import annotations

import fcntl
import logging
import os
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

_BLOCK = 1 << 16


def read_lines(path: Path) -> list[bytes]:
    """The file's whole, non-empty lines (no newline). A missing file holds none."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    whole, _, torn = data.rpartition(b"\n")
    if torn:
        logger.warning("%s: skipped a half-written last line (%d bytes)", path, len(torn))
    return [line for line in whole.split(b"\n") if line.strip()]


def append_lines(path: Path, lines: Iterable[bytes]) -> None:
    """Add lines (each without a newline) at the end, after cutting off a half-written one."""
    data = b"".join(line + b"\n" for line in lines)
    if not data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a+b") as out:
        _cut_torn_tail(out, path)
        out.write(data)
        _sync(out)
    if new:
        _sync_dir(path.parent)


def append_bytes(path: Path, data: bytes) -> None:
    """Add raw bytes at the end (rows of a checkpoint), on disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("ab") as out:
        out.write(data)
        _sync(out)
    if new:
        _sync_dir(path.parent)


def rewrite_lines(path: Path, lines: Iterable[bytes]) -> None:
    """Replace the file's lines in one step: the old file or the new one, never half of each."""
    write_atomic(path, b"".join(line + b"\n" for line in lines))


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as out:
        out.write(data)
        _sync(out)
    tmp.replace(path)
    _sync_dir(path.parent)


def truncate(path: Path, size: int) -> None:
    with path.open("r+b") as out:
        out.truncate(size)
        _sync(out)


@contextmanager
def locked(path: Path) -> Generator[None]:
    """Hold `<path>.lock` for the block, against every other process and thread using it."""
    lock = path.with_name(f"{path.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("ab") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _cut_torn_tail(out: BinaryIO, path: Path) -> None:
    """Truncate a last line left without its newline by a crash. `out` is open "a+b"."""
    end = out.seek(0, os.SEEK_END)
    pos = end
    while pos > 0:
        start = max(0, pos - _BLOCK)
        out.seek(start)
        found = out.read(pos - start).rfind(b"\n")
        if found >= 0:
            pos = start + found + 1
            break
        pos = start
    if pos < end:
        logger.warning("%s: cut a half-written last line (%d bytes)", path, end - pos)
        out.truncate(pos)


def _sync(out: BinaryIO) -> None:
    out.flush()
    os.fsync(out.fileno())


def _sync_dir(directory: Path) -> None:
    """Make a new or renamed file's name durable too."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
