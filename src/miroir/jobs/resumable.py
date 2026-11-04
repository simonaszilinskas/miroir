"""What every resumable embedding run shares, on OpenRouter and on the GPU machine.

A run never overwrites a finished one (`refuse_existing`). Each answer is checked against what
was asked (`check_answer`) and saved to the run's checkpoint as it arrives (`Resumable`), so
a stopped run resumes without paying twice; the vectors written at the end are those of the
ids asked for, once each.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from miroir.images.fetch import Fetched
from miroir.jobs.checkpoint import Checkpoint

logger = logging.getLogger(__name__)


def refuse_existing(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        msg = f"{out} already exists; a finished run is never overwritten"
        raise FileExistsError(msg)


def unit_float16(vectors: NDArray[np.floating]) -> NDArray[np.float16]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float16)


def check_answer(
    run: str, asked: Iterable[str], ids: Sequence[str], vectors: NDArray[np.floating]
) -> None:
    """Refuse an answer whose rows do not match its ids, that repeats an id, or that holds an
    id it was not asked for. Ids asked for may be missing (skipped images)."""
    rows = vectors.shape[0] if vectors.ndim == 2 else -1
    if rows != len(ids):
        msg = f"{run}: {len(ids)} ids but vectors of shape {vectors.shape}"
        raise ValueError(msg)
    if len(set(ids)) != len(ids):
        msg = f"{run}: the answer repeats an id"
        raise ValueError(msg)
    strangers = set(ids) - set(asked)
    if strangers:
        msg = f"{run}: the answer holds {len(strangers)} id(s) not asked for, e.g. {min(strangers)}"
        raise ValueError(msg)


def kept(fetched: Fetched, skips: Counter[str]) -> bool:
    """Whether an image was fetched; counts it by reason when it was not."""
    if fetched.jpeg is not None:
        return True
    skips[f"fetch_{fetched.skip.value if fetched.skip else 'unknown'}"] += 1
    return False


class Resumable:
    """A run's checkpoint: what is saved, what is left to ask, and the vectors at the end."""

    def __init__(self, run: str, checkpoint_dir: Path, outputs: Iterable[Path]) -> None:
        for out in outputs:
            refuse_existing(out)
        self.run = run
        self.checkpoint = Checkpoint(checkpoint_dir)
        self.skips: Counter[str] = Counter()

    def todo(self, wanted: Iterable[str]) -> list[str]:
        held = set(self.checkpoint.ids)
        return [i for i in dict.fromkeys(wanted) if i not in held]

    def keep(
        self,
        asked: Iterable[str],
        ids: Sequence[str],
        vectors: NDArray[np.floating],
        skips: Mapping[str, int] | None = None,
    ) -> None:
        """Check one answer and save it."""
        check_answer(self.run, asked, ids, vectors)
        if ids:
            self.checkpoint.append(list(ids), unit_float16(vectors))
        self.skips.update(skips or {})

    def finish(self, wanted: Iterable[str]) -> tuple[tuple[str, ...], NDArray[np.float16]]:
        """The saved vectors of the ids asked for, once each, in the order they were saved."""
        return self.checkpoint.only(wanted)
