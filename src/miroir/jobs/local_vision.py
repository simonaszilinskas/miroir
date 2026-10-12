"""Embed local image files with a Hugging Face vision backbone. Needs the `jobs` extra.

Used on the Mac for R0 (AmsterTime, 2,462 images) only: everything else runs remotely.
Nothing in the base install imports this module; the CLI loads it with importlib, and pyright
skips it (`[tool.pyright] exclude`) because CI does not install torch.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from miroir.config import SEED
from miroir.eval.vectors import Route, RunInfo, VectorSet, write_vectors

logger = logging.getLogger(__name__)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_files(
    root: Path,
    ids: list[str],
    *,
    model_id: str,
    option: str,
    gallery: str,
    out_dir: Path,
    batch_size: int = 16,
) -> VectorSet:
    """CLS vectors (`pooler_output`: CLS after the final LayerNorm) of `root/<id>` for each id.

    Preprocessing is the model's own image processor; for DINOv2 that is shortest side 256,
    bicubic, centre crop 224, ImageNet mean and std (the DINOv3 paper's AmsterTime protocol).
    """
    torch.manual_seed(SEED)
    device = pick_device()
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to(device).eval()
    crop = processor.crop_size["height"]
    logger.info("embedding %d images with %s on %s at %d px", len(ids), model_id, device, crop)

    started = time.monotonic()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            images = [Image.open(root / i).convert("RGB") for i in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)
            pooled = model(**inputs).pooler_output
            chunks.append(pooled.float().cpu().numpy())
            if (start // batch_size) % 20 == 0:
                logger.info("%d / %d", start + len(batch), len(ids))
    wall = time.monotonic() - started

    info = RunInfo(
        option=option,
        model=model_id,
        resolution=int(crop),
        pooling="cls",
        gallery=gallery,
        created=datetime.now(UTC).date(),
        cost_usd=0.0,
        wall_time_s=round(wall, 1),
        route=Route.MAC,
        notes=f"{device}, fp32, model's own image processor, pooler_output",
    )
    return write_vectors(out_dir, ids, np.concatenate(chunks), info)
