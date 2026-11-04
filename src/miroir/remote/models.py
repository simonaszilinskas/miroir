# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Load a model and turn PIL images into vectors. Runs only on a Scaleway GPU machine.

Type-checked in basic mode: torch and the model libraries are not installed on the Mac.
"""

from __future__ import annotations

import copy
import gc
import io
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image as PILImage

from miroir.config import IMAGES_MOUNT
from miroir.images.fetch import Fetched, ImageCache, ImageFetcher
from miroir.jobs.remote_spec import (
    MAX_ASPECT,
    TEXT_PROMPT,
    ChunkResult,
    DType,
    Family,
    ModelOption,
    Pooling,
    Resize,
    SuppliedImage,
    weights_key,
)
from miroir.parallel import bounded_map
from miroir.records.model import Image
from miroir.remote.preprocess import dino_eval, dino_whole, shape_batches, squash

logger = logging.getLogger(__name__)

#: The GPU, or the CPU on a machine without one (the demo machine's fallback when no GPU
#: is free, `miroir.jobs.demo.DEMO_OFFERS`).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GEM_P = 3.0
#: Parallel image reads on the machine (image folder, else POP), running ahead of the GPU.
FETCH_CONCURRENCY = 32
#: Parallel file reads from the machine's image folder.
VOLUME_READERS = 32
#: Threads decoding and preparing images (resize, normalise) for the GPU.
PREPARE_WORKERS = 8


class Encoder(Protocol):
    option: ModelOption

    def prepare(self, image: PILImage.Image) -> object:
        """CPU work on one image (resize, normalise), safe to run in a reading thread."""
        ...

    def encode_prepared(self, items: list[object]) -> NDArray[np.float32]: ...

    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]: ...


_LOADED: dict[tuple[str, str, str, int], Encoder] = {}
"""One set of weights per machine, by `weights_key`."""


def encoder(option: ModelOption) -> tuple[Encoder, float]:
    """The option's encoder, its weights loaded once per machine. Returns it and the seconds
    loading took.

    Options with the same weights (`weights_key`: CLS and GeM of one model) share them. Another
    model's weights are freed before the new ones load, so two never sit on the GPU together.
    """
    key = weights_key(option)
    if key in _LOADED:
        return _for_option(_LOADED[key], option), 0.0
    _LOADED.clear()
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    start = time.monotonic()
    loaders = {
        Family.DINO: DinoEncoder,
        Family.SIGLIP2: Siglip2Encoder,
        Family.PE_CORE: PeCoreEncoder,
        Family.EMBEDDINGGEMMA: SentenceEncoder,
        Family.QWEN3_VL: SentenceEncoder,
        Family.MEGALOC: MegaLocEncoder,
    }
    loaded = loaders[option.family](option)
    _LOADED[key] = loaded
    took = time.monotonic() - start
    logger.info("loaded %s in %.1f s", option.model, took)
    return loaded, took


def _for_option(loaded: Encoder, option: ModelOption) -> Encoder:
    """The loaded weights with another option's pooling, size and batch."""
    if loaded.option == option:
        return loaded
    other = copy.copy(loaded)
    other.option = option
    return other


# --- the families -----------------------------------------------------------------------------


class DinoEncoder:
    """DINOv2 and DINOv3 through transformers. CLS or GeM over patch tokens, after the LayerNorm.

    Whole images (`Resize.WHOLE`) differ in shape, so each batch runs one forward pass per shape.
    DINOv2 interpolates its position embeddings to any grid; DINOv3 uses RoPE.
    """

    def __init__(self, option: ModelOption) -> None:
        from transformers import AutoModel  # noqa: PLC0415

        self.option = option
        dtype = torch.float32 if option.dtype is DType.FLOAT32 else torch.bfloat16
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(option.model, dtype=dtype).to(DEVICE).eval()
        self.registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.patch = int(self.model.config.patch_size)

    def prepare(self, image: PILImage.Image) -> NDArray[np.float32]:
        side = self.option.resolution
        if self.option.resize is Resize.WHOLE:
            return dino_whole(image, side, self.patch, MAX_ASPECT)
        return dino_eval(image, side)

    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        return self.encode_prepared([self.prepare(i) for i in images])

    @torch.inference_mode()
    def encode_prepared(self, items: list[object]) -> NDArray[np.float32]:
        arrays = [np.asarray(a, dtype=np.float32) for a in items]
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for i, array in enumerate(arrays):
            by_shape.setdefault(array.shape, []).append(i)
        out: NDArray[np.float32] | None = None
        for rows in by_shape.values():
            pixels = torch.from_numpy(np.stack([arrays[i] for i in rows]))
            output = self.model(pixel_values=pixels.to(DEVICE, dtype=self.dtype))
            pooled = self._pool(output.last_hidden_state.float())
            if out is None:
                out = np.empty((len(arrays), pooled.shape[1]), dtype=np.float32)
            out[rows] = pooled
        assert out is not None
        return out

    def _pool(self, tokens: torch.Tensor) -> NDArray[np.float32]:
        cls = tokens[:, 0]
        patches = tokens[:, 1 + self.registers :]
        gem = patches.clamp(min=1e-6).pow(GEM_P).mean(dim=1).pow(1.0 / GEM_P)
        if self.option.pooling is Pooling.CLS:
            pooled = cls
        elif self.option.pooling is Pooling.GEM:
            pooled = gem
        else:  # CLS_GEM: each unit length, so the Mac can split the row in two
            unit = torch.nn.functional.normalize
            pooled = torch.cat([unit(cls, dim=1), unit(gem, dim=1)], dim=1)
        return pooled.cpu().numpy()


class MegaLocEncoder:
    """MegaLoc (DINOv2-B + SALAD) through torch.hub: 322 x 322 input, 8,448-d output."""

    def __init__(self, option: ModelOption) -> None:
        self.option = option
        model = torch.hub.load(option.model, "get_trained_model", trust_repo=True)
        self.model = model.to(DEVICE).eval()

    def prepare(self, image: PILImage.Image) -> NDArray[np.float32]:
        return squash(image, self.option.resolution)

    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        return self.encode_prepared([self.prepare(i) for i in images])

    @torch.inference_mode()
    def encode_prepared(self, items: list[object]) -> NDArray[np.float32]:
        pixels = np.stack([np.asarray(a, dtype=np.float32) for a in items])
        return self.model(torch.from_numpy(pixels).to(DEVICE)).float().cpu().numpy()


class Siglip2Encoder:
    """SigLIP 2 image tower and its attention-pooling head, through transformers."""

    def __init__(self, option: ModelOption) -> None:
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        self.option = option
        self.model = AutoModel.from_pretrained(option.model, dtype=torch.float32).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(option.model)

    def prepare(self, image: PILImage.Image) -> PILImage.Image:
        return image

    def encode_prepared(self, items: list[object]) -> NDArray[np.float32]:
        return self.encode([i for i in items if isinstance(i, PILImage.Image)])

    @torch.inference_mode()
    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out = self.model.get_image_features(**inputs)
        features = out if isinstance(out, torch.Tensor) else out.pooler_output
        return features.float().cpu().numpy()


class PeCoreEncoder:
    """Perception Encoder (PE-Core) image tower through open_clip, with its own preprocessing."""

    def __init__(self, option: ModelOption) -> None:
        import open_clip  # noqa: PLC0415

        model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{option.model}")
        self.option = option
        self.model = model.to(DEVICE).eval()
        self.preprocess = preprocess

    def prepare(self, image: PILImage.Image) -> object:
        return self.preprocess(image)

    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        return self.encode_prepared([self.prepare(i) for i in images])

    @torch.inference_mode()
    def encode_prepared(self, items: list[object]) -> NDArray[np.float32]:
        batch = torch.stack(items).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            features = self.model.encode_image(batch)
        return features.float().cpu().numpy()


class SentenceEncoder:
    """EmbeddingGemma 2 and Qwen3-VL-Embedding through sentence-transformers, in bf16 on a
    GPU and float32 on a CPU.

    Images go in as PIL images, with no prompt (EmbeddingGemma 2) or the model's default
    instruction (Qwen3-VL-Embedding). Qwen's pixel budget is capped at resolution².
    """

    def __init__(self, option: ModelOption) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        if option.dtype is not DType.BFLOAT16:
            msg = f"{option.id}: sentence-transformers models run in bf16 here"
            raise ValueError(msg)
        kwargs: dict[str, object] = {}
        if option.family is Family.QWEN3_VL:
            side = option.resolution
            kwargs["processor_kwargs"] = {"min_pixels": 28 * 28, "max_pixels": side * side}
        self.option = option
        # On a CPU without native bfloat16, bf16 took 12 s for one photo: float32 there.
        dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
        self.model = SentenceTransformer(
            option.model, device=DEVICE, model_kwargs={"torch_dtype": dtype}, **kwargs
        )

    def prepare(self, image: PILImage.Image) -> PILImage.Image:
        return image

    def encode_prepared(self, items: list[object]) -> NDArray[np.float32]:
        return self.encode([i for i in items if isinstance(i, PILImage.Image)])

    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        vectors = self.model.encode(
            images, batch_size=self.option.batch_size, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> NDArray[np.float32]:
        """Search words, with the query prompt the website uses (`TEXT_PROMPT`)."""
        vectors = self.model.encode(
            texts,
            prompt_name=TEXT_PROMPT,
            batch_size=self.option.batch_size,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


# --- the two tasks ----------------------------------------------------------------------------


def embed_images(
    option: ModelOption, images: Sequence[Image], supplied: Sequence[SuppliedImage] = ()
) -> ChunkResult:
    """Fetch POP images by URL (512 px, IIIF fallback) and embed them, then the supplied ones.

    Reading, decoding and preparing run on threads, at least two batches ahead of the GPU."""
    model, load_s = encoder(option)
    start = time.monotonic()
    ahead = max(2 * FETCH_CONCURRENCY, 2 * option.batch_size)
    limits = httpx.Limits(max_connections=FETCH_CONCURRENCY)
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as http:
        cache = ImageCache(Path(IMAGES_MOUNT), None)
        fetcher = ImageFetcher(http, cache, concurrency=FETCH_CONCURRENCY)
        skips: dict[str, int] = {}
        given = (Fetched(s.image_id, s.jpeg, None) for s in supplied)
        fetched = _fetched(chain(fetcher.fetch_all(images, window=ahead), given), skips)
        prepared = bounded_map(
            partial(_prepare, model), fetched, workers=PREPARE_WORKERS, window=ahead
        )
        ids, vectors = _embed_batches(model, prepared, option.batch_size)
    return ChunkResult(ids, vectors, skips, load_s, time.monotonic() - start)


def embed_texts(option: ModelOption, ids: Sequence[str], texts: Sequence[str]) -> ChunkResult:
    """Search words as the website embeds them (EmbeddingGemma 2, `TEXT_PROMPT`)."""
    model, load_s = encoder(option)
    if not isinstance(model, SentenceEncoder):
        msg = f"{option.id} cannot embed words"
        raise TypeError(msg)
    start = time.monotonic()
    vectors = model.encode_text(list(texts))
    if not np.isfinite(vectors).all():
        msg = "the model returned NaN or infinite values"
        raise FloatingPointError(msg)
    return ChunkResult(tuple(ids), vectors, {}, load_s, time.monotonic() - start)


def fill_image_volume(images: Sequence[Image], concurrency: int) -> dict[str, int]:
    """Fetch images into the image volume (skipping those already there). Counts by outcome."""
    counts: dict[str, int] = {}
    limits = httpx.Limits(max_connections=concurrency)
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as http:
        cache = ImageCache(Path(IMAGES_MOUNT), None)
        fetcher = ImageFetcher(http, cache, concurrency=concurrency)
        for item in fetcher.fetch_all(images):
            key = "ok" if item.skip is None else f"fetch_{item.skip.value}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def _fetched(items: Iterable[Fetched], skips: dict[str, int]) -> Iterator[tuple[str, bytes]]:
    for item in items:
        if item.jpeg is None:
            reason = f"fetch_{item.skip.value if item.skip else 'unknown'}"
            skips[reason] = skips.get(reason, 0) + 1
            continue
        yield item.image_id, item.jpeg


def _prepare(model: Encoder, item: tuple[str, bytes]) -> tuple[str, object]:
    return item[0], model.prepare(_open(item[1]))


def _embed_batches(
    model: Encoder, items: Iterable[tuple[str, object]], batch_size: int
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    """Batches of one input shape (whole images differ), so each is one forward pass."""
    ids: list[str] = []
    rows: list[NDArray[np.float32]] = []
    for batch in shape_batches(items, batch_size, hold=4 * batch_size):
        ids.extend(image_id for image_id, _ in batch)
        rows.append(model.encode_prepared([item for _, item in batch]))
    if not rows:
        return (), np.zeros((0, 0), dtype=np.float32)
    vectors = np.concatenate(rows)
    if not np.isfinite(vectors).all():
        msg = "the model returned NaN or infinite values"
        raise FloatingPointError(msg)
    return tuple(ids), vectors


def _open(data: bytes) -> PILImage.Image:
    with PILImage.open(io.BytesIO(data)) as opened:
        return opened.convert("RGB")
