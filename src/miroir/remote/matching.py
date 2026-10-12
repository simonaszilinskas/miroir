# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Keypoint matching for re-ranking (step E1). Runs only inside Modal containers.

ALIKED keypoints (BSD-3; SuperPoint's weights are non-commercial), LightGlue matching,
then RANSAC (OpenCV MAGSAC) for a homography and a fundamental matrix. Images come from the
image volume at 512 px on the longer side and are matched at that size.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image as PILImage

from miroir.config import MODAL_IMAGES_MOUNT
from miroir.eval.rerank import CandidateMatch, MatchChunk, MatchTask, QueryMatches
from miroir.images.fetch import ImageCache
from miroir.parallel import bounded_map

logger = logging.getLogger(__name__)

DEVICE = "cuda"
MAX_KEYPOINTS = 2048
HOMOGRAPHY_PX = 4.0
FUNDAMENTAL_PX = 1.5
RANSAC_CONFIDENCE = 0.999
RANSAC_ITERS = 10_000
VOLUME_READERS = 32

_Features = tuple[dict[str, torch.Tensor], NDArray[np.float32]]


class Matcher:
    def __init__(self) -> None:
        from lightglue import ALIKED, LightGlue  # noqa: PLC0415

        self.extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(DEVICE)
        self.matcher = LightGlue(features="aliked").eval().to(DEVICE)

    @torch.inference_mode()
    def features(self, image: PILImage.Image) -> _Features:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(DEVICE)
        feats = self.extractor.extract(tensor, resize=None)
        return feats, feats["keypoints"][0].cpu().numpy()

    @torch.inference_mode()
    def match(self, a: _Features, b: _Features) -> tuple[int, int, int]:
        """(matches, homography inliers, fundamental inliers)."""
        out = self.matcher({"image0": a[0], "image1": b[0]})
        pairs = out["matches"][0].cpu().numpy()
        found = len(pairs)
        if found < 4:
            return found, 0, 0
        p0 = a[1][pairs[:, 0]].astype(np.float64)
        p1 = b[1][pairs[:, 1]].astype(np.float64)
        return found, _homography_inliers(p0, p1), _fundamental_inliers(p0, p1)


def _homography_inliers(p0: NDArray[np.float64], p1: NDArray[np.float64]) -> int:
    """Inliers of one homography; 0 when OpenCV finds no model (degenerate points)."""
    try:
        _, mask = cv2.findHomography(
            p0,
            p1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=HOMOGRAPHY_PX,
            maxIters=RANSAC_ITERS,
            confidence=RANSAC_CONFIDENCE,
        )
    except cv2.error:
        return 0
    return 0 if mask is None else int(mask.sum())


def _fundamental_inliers(p0: NDArray[np.float64], p1: NDArray[np.float64]) -> int:
    """Inliers of one fundamental matrix; 0 under 8 matches or when OpenCV finds no model."""
    if len(p0) < 8:
        return 0
    try:
        _, mask = cv2.findFundamentalMat(
            p0,
            p1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=FUNDAMENTAL_PX,
            confidence=RANSAC_CONFIDENCE,
            maxIters=RANSAC_ITERS,
        )
    except cv2.error:
        return 0
    return 0 if mask is None else int(mask.sum())


_LOADED: list[Matcher] = []


def match_tasks(tasks: Sequence[MatchTask]) -> MatchChunk:
    """Match every query of the chunk against its candidates."""
    load_s = 0.0
    if not _LOADED:
        start = time.monotonic()
        _LOADED.append(Matcher())
        load_s = time.monotonic() - start
    matcher = _LOADED[0]
    cache = ImageCache(Path(MODAL_IMAGES_MOUNT), None)
    wanted = sorted({i for t in tasks for i in (t.image_id, *t.candidates)})
    read = dict(bounded_map(lambda i: (i, cache.get(i)), wanted, workers=VOLUME_READERS))
    missing = tuple(i for i, data in read.items() if data is None)
    held: dict[str, _Features] = {}
    done: list[QueryMatches] = []
    for task in tasks:
        data = read[task.image_id]
        if data is None:
            continue
        t0 = time.monotonic()
        query = matcher.features(_open(data))
        t1 = time.monotonic()
        todo = [c for c in task.candidates if c not in held and read[c] is not None]
        for image_id in todo:
            held[image_id] = matcher.features(_open(read[image_id] or b""))
        t2 = time.monotonic()
        found = []
        for image_id in task.candidates:
            if image_id in held:
                found.append(CandidateMatch(image_id, *matcher.match(query, held[image_id])))
        t3 = time.monotonic()
        done.append(
            QueryMatches(
                query_id=task.query_id,
                image_id=task.image_id,
                candidates=tuple(found),
                query_s=t1 - t0,
                match_s=t3 - t2,
                candidates_s=t2 - t1,
            )
        )
    if missing:
        logger.warning("could not read %d image(s), e.g. %s", len(missing), missing[0])
    return MatchChunk(tuple(done), load_s, missing)


def _open(data: bytes) -> PILImage.Image:
    with PILImage.open(io.BytesIO(data)) as opened:
        return opened.convert("RGB")
