# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Keypoint re-ranking for the website's object searches (step E1). Runs only inside Modal.

For one query photo and the best image of each of its top entities: read the images from the
`miroir-images` volume (512 px, as the gallery was embedded), ALIKED keypoints on each, then
LightGlue and a homography (`miroir.remote.matching.Matcher`). Nothing is computed ahead:
every search pays for its candidates' keypoints, which is what the timings measure.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image as PILImage

from miroir.config import MODAL_IMAGES_MOUNT
from miroir.eval.wire import SiteMatches, SiteMatchTask
from miroir.images.fetch import ImageCache
from miroir.parallel import bounded_map
from miroir.remote.matching import Matcher

logger = logging.getLogger(__name__)

READERS = 16


def _open(data: bytes) -> PILImage.Image:
    with PILImage.open(io.BytesIO(data)) as opened:
        return opened.convert("RGB")


class KeypointReranker:
    """`miroir.web.engine.Reranker` on the GPU."""

    def __init__(self, images_dir: Path) -> None:
        start = time.monotonic()
        self.matcher = Matcher()
        self.cache = ImageCache(images_dir, None)
        self.timings: dict[str, float] = {}
        logger.info("keypoint matcher loaded in %.1f s", time.monotonic() - start)

    def inliers(self, query_jpeg: bytes, image_ids: Sequence[str]) -> list[int]:
        """Homography inliers between the query and each image; 0 for an image not found.
        `self.timings` then holds what each part took, in seconds."""
        start = time.monotonic()
        read = dict(bounded_map(lambda i: (i, self.cache.get(i)), list(image_ids), workers=READERS))
        read_done = time.monotonic()
        query = self.matcher.features(_open(query_jpeg))
        held = {
            i: self.matcher.features(_open(data)) for i, data in read.items() if data is not None
        }
        torch.cuda.synchronize()
        features_done = time.monotonic()
        out = [self.matcher.inliers(query, held[i]) if i in held else 0 for i in image_ids]
        torch.cuda.synchronize()
        self.timings = {
            "read": read_done - start,
            "features": features_done - read_done,
            "match": time.monotonic() - features_done,
        }
        return out


_LOADED: list[KeypointReranker] = []


def match_site_tasks(tasks: Sequence[SiteMatchTask]) -> list[SiteMatches]:
    """Re-rank evaluation: each task timed as the website would pay for it."""
    if not _LOADED:
        _LOADED.append(KeypointReranker(Path(MODAL_IMAGES_MOUNT)))
    reranker = _LOADED[0]
    if tasks and tasks[0].image_id is not None:  # warm the GPU kernels before timing
        warm = reranker.cache.get(tasks[0].image_id)
        if warm is not None:
            reranker.inliers(warm, tasks[0].candidates[:2])
    out: list[SiteMatches] = []
    for task in tasks:
        jpeg = task.jpeg
        if jpeg is None and task.image_id is not None:
            jpeg = reranker.cache.get(task.image_id)
        if jpeg is None:
            logger.warning("%s: no query image", task.query_id)
            continue
        start = time.monotonic()
        inliers = reranker.inliers(jpeg, task.candidates)
        out.append(
            SiteMatches(
                query_id=task.query_id,
                candidates=task.candidates,
                inliers=tuple(inliers),
                seconds=time.monotonic() - start,
                read_s=reranker.timings["read"],
                features_s=reranker.timings["features"],
                match_s=reranker.timings["match"],
            )
        )
    return out
