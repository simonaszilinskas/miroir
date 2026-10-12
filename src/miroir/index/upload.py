"""Copy the full-sample index into the `miroir-index` Modal volume, where the website reads it.

Layout in the volume: `/vectors/<run>/` (the vector contract, technical spec §7.1) and
`/<file>.npz` for every full-sample file of `Settings.index_dir` (PCA matrix, the guess:
names ending in `-full.npz`; fits on the screening gallery stay on the Mac). A file already
there is replaced.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import modal

from miroir.config import Settings
from miroir.eval.vectors import read_meta

logger = logging.getLogger(__name__)

INDEX_VOLUME = "miroir-index"


def upload(settings: Settings, runs: Sequence[str]) -> list[str]:
    """Upload finished vector runs and the full-sample .npz files. Returns one line per item."""
    for run in runs:
        read_meta(settings.vectors_path(run))  # refuse unfinished runs
    volume = modal.Volume.from_name(INDEX_VOLUME, create_if_missing=True)
    done: list[str] = []
    with volume.batch_upload(force=True) as batch:
        for run in runs:
            batch.put_directory(str(settings.vectors_path(run)), f"/vectors/{run}")  # pyright: ignore[reportUnknownMemberType]
            done.append(f"{INDEX_VOLUME}:/vectors/{run}/")
        for path in sorted(settings.index_dir.glob("*-full.npz")):
            batch.put_file(str(path), f"/{path.name}")  # pyright: ignore[reportUnknownMemberType]
            done.append(f"{INDEX_VOLUME}:/{path.name}")
    logger.info("uploaded %d item(s) to %s", len(done), INDEX_VOLUME)
    return done
