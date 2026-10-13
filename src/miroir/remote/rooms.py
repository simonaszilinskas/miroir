# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Step C on the Jeu de Paume room photos. Runs only inside Modal containers.

- `detect_rooms`: one detector on each room photo (at the size the page sends), then every
  raw box cut from the photo and embedded as the site embeds a query (`prepare_photo`, then
  EmbeddingGemma 2 and MegaLoc), so the Mac can rank each box against the full index.
- `locate_artworks`: where each T3 artwork hangs in its room photo, found by matching its
  crops (ALIKED + LightGlue + homography): the boxes detectors are scored against.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from miroir.config import MODAL_IMAGES_MOUNT
from miroir.eval.wire import DetectChunk, DetectedRoom, Located, LocateTask, RoomPhoto
from miroir.images.fetch import ImageCache
from miroir.outlines import Box, DetectorOption, clipped
from miroir.remote.detect import Detector, detect_size, load_detector
from miroir.remote.matching import HOMOGRAPHY_PX, RANSAC_CONFIDENCE, RANSAC_ITERS
from miroir.web.engine import prepare_photo

logger = logging.getLogger(__name__)

DEVICE = "cuda"
#: Keypoints on a room photo (2,048 px): enough to cover each of its paintings.
SCENE_KEYPOINTS = 8192
CROP_KEYPOINTS = 2048
#: Fewest homography inliers for a crop to count as found in the room photo.
MIN_INLIERS = 25
#: A located picture covers at least this share of the room photo, and at most the second.
AREA_RANGE = (0.0003, 0.6)

_LOADED: dict[str, object] = {}


def _open(data: bytes) -> PILImage.Image:
    with PILImage.open(io.BytesIO(data)) as opened:
        return opened.convert("RGB")


def _jpeg(image: PILImage.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=90)
    return out.getvalue()


def detect_rooms(
    option: DetectorOption | None,
    photos: Sequence[RoomPhoto],
    given: Mapping[str, tuple[Box, ...]],
) -> DetectChunk:
    """Detect with `option`, or embed the boxes `given` per photo id when `option` is None."""
    from miroir.remote.search import QueryModels  # noqa: PLC0415

    start = time.monotonic()
    if "models" not in _LOADED:
        _LOADED["models"] = QueryModels()
    if option is not None and option.id not in _LOADED:
        for old in [k for k in _LOADED if k not in ("models", "locator")]:
            del _LOADED[old]  # one detector at a time on the GPU
        torch.cuda.empty_cache()
        _LOADED[option.id] = load_detector(option)
    load_s = time.monotonic() - start
    models: QueryModels = _LOADED["models"]  # type: ignore[assignment]
    detector: Detector | None = None if option is None else _LOADED[option.id]  # type: ignore[assignment]
    if detector is not None and photos:  # warm the GPU kernels: time a warm search's detection
        detector.detect(detect_size(_open(photos[0].jpeg)))
    rooms: list[DetectedRoom] = []
    for photo in photos:
        full = _open(photo.jpeg)
        detect_s = 0.0
        if detector is None:
            boxes = list(given.get(photo.image_id, ()))
        else:
            small = detect_size(full)
            torch.cuda.synchronize()
            tick = time.monotonic()
            boxes = detector.detect(small)
            torch.cuda.synchronize()
            detect_s = time.monotonic() - tick
        a7: list[np.ndarray] = []
        a3: list[np.ndarray] = []
        for box in boxes:
            crop = full.crop(box.pixels(full.width, full.height))
            v7, v3 = models.image(prepare_photo(_jpeg(crop)))
            a7.append(v7)
            a3.append(v3)
        rooms.append(
            DetectedRoom(
                image_id=photo.image_id,
                width=full.width,
                height=full.height,
                boxes=tuple(boxes),
                a7=_rows(a7),
                a3=_rows(a3),
                detect_s=detect_s,
            )
        )
        logger.info("%s: %d raw boxes in %.0f ms", photo.image_id, len(boxes), 1000 * detect_s)
    return DetectChunk(tuple(rooms), load_s)


def _rows(vectors: list[np.ndarray]) -> np.ndarray:
    return np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), np.float32)


class _Locator:
    def __init__(self) -> None:
        from lightglue import ALIKED, LightGlue  # noqa: PLC0415

        self.scene = ALIKED(max_num_keypoints=SCENE_KEYPOINTS).eval().to(DEVICE)
        self.crop = ALIKED(max_num_keypoints=CROP_KEYPOINTS).eval().to(DEVICE)
        self.matcher = LightGlue(features="aliked").eval().to(DEVICE)

    @torch.inference_mode()
    def features(self, extractor, image: PILImage.Image):
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(DEVICE)
        feats = extractor.extract(tensor, resize=None)
        return feats, feats["keypoints"][0].cpu().numpy()

    @torch.inference_mode()
    def locate(self, crop, crop_size: tuple[int, int], scene, scene_size: tuple[int, int]):
        """(inliers, box in the room photo or None)."""
        out = self.matcher({"image0": crop[0], "image1": scene[0]})
        pairs = out["matches"][0].cpu().numpy()
        if len(pairs) < MIN_INLIERS:
            return len(pairs), None
        p0 = crop[1][pairs[:, 0]].astype(np.float64)
        p1 = scene[1][pairs[:, 1]].astype(np.float64)
        try:
            matrix, mask = cv2.findHomography(
                p0,
                p1,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=HOMOGRAPHY_PX,
                maxIters=RANSAC_ITERS,
                confidence=RANSAC_CONFIDENCE,
            )
        except cv2.error:
            return 0, None
        inliers = 0 if mask is None else int(mask.sum())
        if matrix is None or inliers < MIN_INLIERS:
            return inliers, None
        w, h = crop_size
        corners = np.asarray([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float64)
        quad = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
        sw, sh = scene_size
        area = abs(cv2.contourArea(quad.astype(np.float32))) / (sw * sh)
        if not cv2.isContourConvex(quad.astype(np.float32)) or not (
            AREA_RANGE[0] <= area <= AREA_RANGE[1]
        ):
            return inliers, None
        xs, ys = quad[:, 0] / sw, quad[:, 1] / sh
        return inliers, clipped(xs.min(), ys.min(), xs.max(), ys.max())


def locate_artworks(scenes: Sequence[RoomPhoto], tasks: Sequence[LocateTask]) -> list[Located]:
    if "locator" not in _LOADED:
        _LOADED["locator"] = _Locator()
    locator: _Locator = _LOADED["locator"]  # type: ignore[assignment]
    cache = ImageCache(Path(MODAL_IMAGES_MOUNT), None)
    scene_feats = {}
    for scene in scenes:
        image = _open(scene.jpeg)
        scene_feats[scene.image_id] = (locator.features(locator.scene, image), image.size)
    out: list[Located] = []
    for task in tasks:
        scene, scene_size = scene_feats[task.scene_id]
        best = Located(task.scene_id, task.entity_id, "", 0, None)
        for image_id in task.image_ids:
            data = cache.get(image_id)
            if data is None:
                logger.warning("no image %s in the volume", image_id)
                continue
            crop_image = _open(data)
            crop = locator.features(locator.crop, crop_image)
            inliers, box = locator.locate(crop, crop_image.size, scene, scene_size)
            if box is not None and (best.box is None or inliers > best.inliers):
                best = Located(task.scene_id, task.entity_id, image_id, inliers, box)
            elif best.box is None and inliers > best.inliers:
                best = Located(task.scene_id, task.entity_id, image_id, inliers, None)
        out.append(best)
    return out
