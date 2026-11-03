"""Long jobs over all of POP that run unattended on Scaleway machines: where their files go.

Each job is split into shards by image id (`shard_of`); one process works one shard and
writes, to the images bucket, a status file the Mac reads (`JobStatus`) and its failures, one
JSON line each. Everything a job writes is keyed so that a rerun skips the work already done.

| Key, in `<prefix>-images` | What |
|---|---|
| `lists/images.parquet` | every distinct image to fetch: `image_id`, `url`, `collection` |
| `full/<id[:2]>/<id>.jpg` | an image at most `FULL_LONG_SIDE` px on its long side |
| `status/<job>-<shard>.json` | a shard's `JobStatus` |
| `failures/<job>-<shard>.jsonl` | the images a shard could not do, with why |
| `lists/july.txt` | files whose vectors the July runs made: not embedded again |
| `vectors/<option>/<piece>.npz` | one model's unit vectors (float16) of one piece, with ids |
| `check/<job>.npz` | a few July files embedded again, to compare with the July vectors |
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

#: Longest side, in pixels, of the stored images (POP serves no larger file for Mémoire).
FULL_LONG_SIDE = 1024
FULL_PREFIX = "full"
IMAGE_LIST_KEY = "lists/images.parquet"
JULY_LIST_KEY = "lists/july.txt"
#: Embedding work is cut by the first three hex digits of the file id.
PIECES = 4096


class JobState(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def full_key(image_id: str) -> str:
    return f"{FULL_PREFIX}/{image_id[:2]}/{image_id}.jpg"


def shard_of(image_id: str, of: int) -> int:
    """The shard an image belongs to: by the first two hex digits of its id, so a shard's
    stored images are the listing of a few key prefixes."""
    return int(image_id[:2], 16) % of


def shard_prefixes(shard: int, of: int) -> list[str]:
    """The `full/<xx>/` key prefixes that hold this shard's images."""
    return [f"{FULL_PREFIX}/{p:02x}/" for p in range(256) if p % of == shard]


def piece_prefix(piece: int) -> str:
    return f"{piece:03x}"


def piece_key(option_id: str, prefix3: str) -> str:
    return f"vectors/{option_id}/{prefix3}.npz"


def check_key(job: str) -> str:
    return f"check/{job}.npz"


def piece_ranges(first: int, last: int, parts: int) -> list[tuple[int, int]]:
    """Pieces `first`..`last` split into `parts` ranges of nearly equal length."""
    count = last - first + 1
    bounds = [
        (first + k * count // parts, first + (k + 1) * count // parts - 1) for k in range(parts)
    ]
    return [(a, b) for a, b in bounds if b >= a]


def status_key(job: str, shard: int) -> str:
    return f"status/{job}-{shard}.json"


def failures_key(job: str, shard: int) -> str:
    return f"failures/{job}-{shard}.jsonl"


@dataclass(frozen=True, slots=True)
class JobStatus:
    """What one shard has done so far. Written by the machine, read by the Mac."""

    job: str
    shard: int
    of: int
    total: int = 0
    """Items this shard had to do when it started (those already done are not counted)."""
    done: int = 0
    counts: dict[str, int] = field(default_factory=dict[str, int])
    """Outcomes so far: "ok", or a failure reason."""
    state: JobState = JobState.RUNNING
    started: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    note: str = ""

    @property
    def per_s(self) -> float:
        return self.done / max(self.updated - self.started, 1e-9)

    def to_json(self) -> str:
        return json.dumps({**asdict(self), "per_s": round(self.per_s, 2)}, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> JobStatus:
        data = json.loads(text)
        data.pop("per_s", None)
        data["state"] = JobState(data["state"])
        return cls(**data)

    def now(self, **changes: object) -> JobStatus:
        """A copy with `changes`, stamped with the time."""
        return replace(self, **changes, updated=time.time())

    def line(self) -> str:
        """One line for the Mac's log."""
        counts = ", ".join(f"{k} {v:,}" for k, v in sorted(self.counts.items()))
        return (
            f"{self.job} {self.shard}/{self.of} {self.state}: {self.done:,}/{self.total:,} "
            f"({self.per_s:.1f}/s) {counts} {self.note}"
        ).rstrip()
