# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Load a model and turn PIL images into vectors. Runs only inside Modal containers.

Type-checked in basic mode: torch and the model libraries are not installed on the Mac.
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from itertools import batched, chain
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image as PILImage

from miroir.config import MODAL_IMAGES_MOUNT
from miroir.images.fetch import Fetched, ImageCache, ImageFetcher
from miroir.jobs.modal_spec import (
    MAX_ASPECT,
    TEXT_PROMPT,
    ChunkResult,
    DType,
    Family,
    ModalOption,
    Pooling,
    Resize,
    SuppliedImage,
)
from miroir.parallel import bounded_map
from miroir.records.model import Image
from miroir.remote.preprocess import dino_eval, dino_whole, squash

logger = logging.getLogger(__name__)

DEVICE = "cuda"
GEM_P = 3.0
#: Parallel image reads per container (volume, else POP), running ahead of the GPU.
FETCH_CONCURRENCY = 32
#: Parallel file reads from a Modal volume.
VOLUME_READERS = 32


class Encoder(Protocol):
    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]: ...


_LOADED: dict[str, Encoder] = {}


def encoder(option: ModalOption) -> tuple[Encoder, float]:
    """The option's encoder, loaded once per container. Returns it and the seconds loading took."""
    if option.id in _LOADED:
        return _LOADED[option.id], 0.0
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
    _LOADED.clear()  # one model per container: free the GPU before the next
    _LOADED[option.id] = loaded
    took = time.monotonic() - start
    logger.info("loaded %s in %.1f s", option.model, took)
    return loaded, took


# --- the families -----------------------------------------------------------------------------


class DinoEncoder:
    """DINOv2 and DINOv3 through transformers. CLS or GeM over patch tokens, after the LayerNorm.

    Whole images (`Resize.WHOLE`) differ in shape, so each batch runs one forward pass per shape.
    DINOv2 interpolates its position embeddings to any grid; DINOv3 uses RoPE.
    """

    def __init__(self, option: ModalOption) -> None:
        from transformers import AutoModel  # noqa: PLC0415

        self.option = option
        dtype = torch.float32 if option.dtype is DType.FLOAT32 else torch.bfloat16
        self.dtype = dtype
        self.model = AutoModel.from_pretrained(option.model, dtype=dtype).to(DEVICE).eval()
        self.registers = int(getattr(self.model.config, "num_register_tokens", 0) or 0)
        self.patch = int(self.model.config.patch_size)

    def _pixels(self, image: PILImage.Image) -> NDArray[np.float32]:
        side = self.option.resolution
        if self.option.resize is Resize.WHOLE:
            return dino_whole(image, side, self.patch, MAX_ASPECT)
        return dino_eval(image, side)

    @torch.inference_mode()
    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        arrays = [self._pixels(i) for i in images]
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for i, array in enumerate(arrays):
            by_shape.setdefault(array.shape, []).append(i)
        out: NDArray[np.float32] | None = None
        for rows in by_shape.values():
            pixels = torch.from_numpy(np.stack([arrays[i] for i in rows]))
            output = self.model(pixel_values=pixels.to(DEVICE, dtype=self.dtype))
            pooled = self._pool(output.last_hidden_state.float())
            if out is None:
                out = np.empty((len(images), pooled.shape[1]), dtype=np.float32)
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

    def __init__(self, option: ModalOption) -> None:
        self.option = option
        model = torch.hub.load(option.model, "get_trained_model", trust_repo=True)
        self.model = model.to(DEVICE).eval()

    @torch.inference_mode()
    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        pixels = np.stack([squash(i, self.option.resolution) for i in images])
        return self.model(torch.from_numpy(pixels).to(DEVICE)).float().cpu().numpy()


class Siglip2Encoder:
    """SigLIP 2 image tower and its attention-pooling head, through transformers."""

    def __init__(self, option: ModalOption) -> None:
        from transformers import AutoModel, AutoProcessor  # noqa: PLC0415

        self.model = AutoModel.from_pretrained(option.model, dtype=torch.float32).to(DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(option.model)

    @torch.inference_mode()
    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            out = self.model.get_image_features(**inputs)
        features = out if isinstance(out, torch.Tensor) else out.pooler_output
        return features.float().cpu().numpy()


class PeCoreEncoder:
    """Perception Encoder (PE-Core) image tower through open_clip, with its own preprocessing."""

    def __init__(self, option: ModalOption) -> None:
        import open_clip  # noqa: PLC0415

        model, preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{option.model}")
        self.model = model.to(DEVICE).eval()
        self.preprocess = preprocess

    @torch.inference_mode()
    def encode(self, images: list[PILImage.Image]) -> NDArray[np.float32]:
        batch = torch.stack([self.preprocess(i) for i in images]).to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            features = self.model.encode_image(batch)
        return features.float().cpu().numpy()


class SentenceEncoder:
    """EmbeddingGemma 2 and Qwen3-VL-Embedding through sentence-transformers, in bf16.

    Images go in as PIL images, with no prompt (EmbeddingGemma 2) or the model's default
    instruction (Qwen3-VL-Embedding). Qwen's pixel budget is capped at resolution².
    """

    def __init__(self, option: ModalOption) -> None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        if option.dtype is not DType.BFLOAT16:
            msg = f"{option.id}: sentence-transformers models run in bf16 here"
            raise ValueError(msg)
        kwargs: dict[str, object] = {}
        if option.family is Family.QWEN3_VL:
            side = option.resolution
            kwargs["processor_kwargs"] = {"min_pixels": 28 * 28, "max_pixels": side * side}
        self.option = option
        self.model = SentenceTransformer(
            option.model, device=DEVICE, model_kwargs={"torch_dtype": torch.bfloat16}, **kwargs
        )

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
    option: ModalOption, images: Sequence[Image], supplied: Sequence[SuppliedImage] = ()
) -> ChunkResult:
    """Fetch POP images by URL (512 px, IIIF fallback) and embed them, then the supplied ones."""
    model, load_s = encoder(option)
    start = time.monotonic()
    limits = httpx.Limits(max_connections=FETCH_CONCURRENCY)
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as http:
        cache = ImageCache(Path(MODAL_IMAGES_MOUNT), None)
        fetcher = ImageFetcher(http, cache, concurrency=FETCH_CONCURRENCY)
        skips: dict[str, int] = {}
        given = (Fetched(s.image_id, s.jpeg, None) for s in supplied)
        decoded = _decoded(chain(fetcher.fetch_all(images), given), skips)
        ids, vectors = _embed_batches(model, decoded, option.batch_size)
    return ChunkResult(ids, vectors, skips, load_s, time.monotonic() - start)


def embed_texts(option: ModalOption, ids: Sequence[str], texts: Sequence[str]) -> ChunkResult:
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
        cache = ImageCache(Path(MODAL_IMAGES_MOUNT), None)
        fetcher = ImageFetcher(http, cache, concurrency=concurrency)
        for item in fetcher.fetch_all(images):
            key = "ok" if item.skip is None else f"fetch_{item.skip.value}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def embed_files(option: ModalOption, files: Sequence[tuple[str, Path]]) -> ChunkResult:
    """Embed images read from disk (a benchmark on a Modal volume), keyed by the given ids."""
    model, load_s = encoder(option)
    start = time.monotonic()
    # Volume files load lazily and one read waits on the network: read many at once.
    opened = bounded_map(
        lambda item: (item[0], _open(item[1].read_bytes())), files, workers=VOLUME_READERS
    )
    ids, vectors = _embed_batches(model, opened, option.batch_size)
    return ChunkResult(ids, vectors, {}, load_s, time.monotonic() - start)


def _decoded(
    fetched: Iterable[Fetched], skips: dict[str, int]
) -> Iterator[tuple[str, PILImage.Image]]:
    for item in fetched:
        if item.jpeg is None:
            reason = f"fetch_{item.skip.value if item.skip else 'unknown'}"
            skips[reason] = skips.get(reason, 0) + 1
            continue
        yield item.image_id, _open(item.jpeg)


def _embed_batches(
    model: Encoder, items: Iterable[tuple[str, PILImage.Image]], batch_size: int
) -> tuple[tuple[str, ...], NDArray[np.float32]]:
    ids: list[str] = []
    rows: list[NDArray[np.float32]] = []
    for batch in batched(items, batch_size):
        ids.extend(image_id for image_id, _ in batch)
        rows.append(model.encode([image for _, image in batch]))
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
