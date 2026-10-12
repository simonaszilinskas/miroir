"""AmsterTime (set T9): download, layout, and the three query directions of the official code.

AmsterTime holds 1,231 places in Amsterdam, each with one archive photo (`old/NNNN.jpg`) and
one street-view photo (`new/NNNN.png`); the same number is the same place. Licence CC0 1.0.

The official evaluation (github.com/seyrankhademi/AmsterTime, `amstertime/eval.py`, task file
`tasks/retrieval.csv`) prints three scores:

- "Old": archive queries against the street-view gallery,
- "New": street-view queries against the archive gallery,
- "All": both sets of queries pooled (2,462).

It takes the top k = 128 neighbours by cosine distance; AP is 1/rank of the one match, or 0
past rank 128. `score_amstertime` reproduces that with the MIROIR scorer.

Embedding runs on a hosted service and returns a vector run (technical spec §7.1) whose ids
are `old/NNNN.jpg` and `new/NNNN.png`. Nothing here loads a model.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import urllib.request
import zipfile
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path

from miroir.eval.scoring import Gallery, ScoreResult, SearchQuery, score
from miroir.eval.vectors import VectorSet

logger = logging.getLogger(__name__)

#: Figshare copy of DOI 10.4121/19580806 (4TU.ResearchData). data.4tu.nl itself often fails.
URL = "https://ndownloader.figshare.com/files/34793920"
SIZE = 913_361_822
MD5 = "b9229f138e5a3e0dfa2c21f8db8aa443"
PAIRS = 1_231

#: Neighbours the official code retrieves; AP past this rank counts 0.
OFFICIAL_K = 128

_CHUNK = 1 << 20


class Direction(StrEnum):
    """Query direction, named as the official code prints it."""

    OLD = "old"
    """Archive photos query the street-view gallery."""
    NEW = "new"
    """Street-view photos query the archive gallery."""
    ALL = "all"
    """Both, pooled."""


def download(target: Path) -> Path:
    """Fetch and unpack AmsterTime into `target/images/{old,new}`. Skips work already done."""
    images = target / "images"
    if _complete(images):
        logger.info("AmsterTime already in %s", images)
        return images
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "AmsterTime-v1.0.zip"
    if not archive.exists() or archive.stat().st_size != SIZE:
        logger.info("downloading AmsterTime (%.1f GB) from %s", SIZE / 1e9, URL)
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(URL) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, _CHUNK)
        partial.rename(archive)
    if _md5(archive) != MD5:
        msg = f"{archive}: MD5 does not match {MD5}; delete it and download again"
        raise ValueError(msg)
    staging = target / "unpacked"
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(staging)
    old_dir = _find_dir(staging, "old")
    new_dir = _find_dir(staging, "new")
    images.mkdir(parents=True, exist_ok=True)
    shutil.move(old_dir, images / "old")
    shutil.move(new_dir, images / "new")
    shutil.rmtree(staging)
    if not _complete(images):
        msg = f"{images}: expected {PAIRS} matching old/new pairs"
        raise ValueError(msg)
    logger.info("AmsterTime ready in %s", images)
    return images


def image_ids(images: Path) -> list[str]:
    """Ids as `old/0000.jpg`, `new/0000.png`: relative paths, sorted."""
    return sorted(
        f"{side}/{path.name}" for side in ("old", "new") for path in (images / side).iterdir()
    )


def place_of(image_id: str) -> str:
    """The place number an image shows: `old/0042.jpg` -> `0042`."""
    return image_id.split("/", 1)[1].split(".", 1)[0]


def side_of(image_id: str) -> str:
    return image_id.split("/", 1)[0]


def queries(vectors: VectorSet, direction: Direction) -> tuple[SearchQuery, ...]:
    """Queries for one direction. Each excludes its own side, so it searches the other one."""
    sides = ("old", "new") if direction is Direction.ALL else (direction.value,)
    by_side = {side: tuple(i for i in vectors.ids if side_of(i) == side) for side in ("old", "new")}
    return tuple(
        SearchQuery(
            query_id=image_id,
            image_id=image_id,
            entity_ids=(place_of(image_id),),
            excluded_image_ids=by_side[side],
        )
        for side in sides
        for image_id in by_side[side]
    )


def score_amstertime(vectors: VectorSet, direction: Direction) -> ScoreResult:
    """Score one direction exactly as the official code does (cosine, AP cut at rank 128)."""
    gallery = Gallery.from_vectors(vectors, {i: place_of(i) for i in vectors.ids})
    return score(vectors, gallery, queries(vectors, direction), ap_max_rank=OFFICIAL_K)


def _complete(images: Path) -> bool:
    if not (images / "old").is_dir() or not (images / "new").is_dir():
        return False
    old = {place_of(f"old/{p.name}") for p in (images / "old").iterdir()}
    new = {place_of(f"new/{p.name}") for p in (images / "new").iterdir()}
    return len(old) == PAIRS and old == new


def _find_dir(root: Path, name: str) -> Path:
    found = [p for p in _walk_dirs(root) if p.name == name and not p.name.startswith(".")]
    if len(found) != 1:
        msg = f"expected one {name!r} folder in the AmsterTime archive, found {found}"
        raise ValueError(msg)
    return found[0]


def _walk_dirs(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_dir() and "__MACOSX" not in path.parts:
            yield path


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 (checksum the publisher gives, not security)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
