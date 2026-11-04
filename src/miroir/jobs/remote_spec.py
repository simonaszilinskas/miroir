"""What runs on the GPU machine: the open-model options and the timeouts of a call.

Plain data, read on the Mac (to plan, check spend and name runs) and on the machine (to load
the right model). No model import here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

#: The model libraries, pinned: the Scaleway machine's Docker image (`miroir.jobs.scw_run`)
#: installs them.
MODEL_LIBRARIES = (
    "torch==2.14.1",
    "torchvision==0.29.1",
    "transformers==5.19.0",
    "sentence-transformers[image]==6.1.0",
    "open-clip-torch==3.3.0",
    "timm==1.0.30",
    "qwen-vl-utils==0.0.14",
    "accelerate==1.15.0",
    "httpx==0.28.1",
    "pillow==12.3.0",
    "numpy==2.5.4",
    "pydantic==2.14.1",
    "pydantic-settings==2.15.0",
)
#: Images per remote call. The Mac logs spend and checks the budget between calls.
CHUNK_IMAGES = 500
#: A call may run this long, model download included (the tunnel waits a little longer).
CHUNK_TIMEOUT_S = 20 * 60
#: EmbeddingGemma 2's prompt for words searched against images: the website's query side
#: (`miroir.remote.search`) and the T7 query runs (`embed_texts`) both use it.
TEXT_PROMPT = "SearchQuery"
#: Texts per remote call when embedding words.
CHUNK_TEXTS = 2_000


class Family(StrEnum):
    """Models that load and run the same way."""

    DINO = "dino"
    SIGLIP2 = "siglip2"
    PE_CORE = "pe-core"
    EMBEDDINGGEMMA = "embeddinggemma"
    QWEN3_VL = "qwen3-vl"
    MEGALOC = "megaloc"


class Pooling(StrEnum):
    CLS = "cls"
    """Class token after the final LayerNorm."""
    GEM = "gem"
    """Generalised mean (p = 3) over patch tokens after the final LayerNorm."""
    SALAD = "salad"
    """MegaLoc's optimal-transport aggregation of DINOv2 patch tokens."""
    HEAD = "head"
    """The model's own pooling and projection (CLIP-style towers, embedding models)."""
    CLS_GEM = "cls+gem"
    """CLS and GeM from one pass, each unit length, side by side. The Mac splits the run in two."""


class Resize(StrEnum):
    """How a DINO model sees an image. Other families use their own processor."""

    CENTRE_CROP = "centre-crop"
    """The published eval transform: shorter side to resolution x 256/224, then a centre square."""
    WHOLE = "whole"
    """The whole image, aspect kept: shorter side to `resolution`, each side a multiple of the
    patch size, the longer side at most `MAX_ASPECT` times the shorter."""


#: Longest side of a `Resize.WHOLE` image, as a multiple of its shorter side.
MAX_ASPECT = 2.0


class DType(StrEnum):
    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    """Autocast or weights in bf16. Never float16: EmbeddingGemma 2 returns NaN in it."""


@dataclass(frozen=True, slots=True)
class Machine:
    """The machine a call runs on, at one price for the whole machine."""

    gpu: str
    """Its type, e.g. "L4-1-24G"; run notes name it."""
    usd_per_s: float


#: Parallel downloads when filling the images bucket (`GpuRunner.fill_images`).
FILL_CONCURRENCY = 16
FILL_TIMEOUT_S = 20 * 60


@dataclass(frozen=True, slots=True)
class ModelOption:
    """One way to embed images with an open model."""

    id: str
    """Run-name prefix and probe key, e.g. "A1-dinov2-g-224-cls"."""
    experiment: str
    """Option id from experiments.md §2, e.g. "A1"."""
    family: Family
    model: str
    """Hugging Face model id."""
    resolution: int
    """Side of the square the model sees (DINO crop), its shorter side (DINO whole), or the
    longest side sent (others)."""
    pooling: Pooling
    dtype: DType
    batch_size: int
    gated: bool = False
    """Weights need an accepted licence on Hugging Face (read with `HF_TOKEN`)."""
    resize: Resize = Resize.CENTRE_CROP
    """DINO only: crop to a square, or keep the whole image."""


def weights_key(opt: ModelOption) -> tuple[str, str, str, int]:
    """What decides the weights a machine loads: options alike here share one copy (CLS and GeM
    of one model). Qwen3-VL's processor is built for its pixel budget, so its size counts."""
    size = opt.resolution if opt.family is Family.QWEN3_VL else 0
    return (opt.family.value, opt.model, opt.dtype.value, size)


def _dino(
    id_: str, experiment: str, model: str, pooling: Pooling, *, gated: bool = False, crop: int = 224
) -> ModelOption:
    return ModelOption(
        id_, experiment, Family.DINO, model, crop, pooling, DType.FLOAT32, 64, gated=gated
    )


_V3_L = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_V3_HPLUS = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
_V3_7B = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


def _whole(id_: str, model: str, short: int, batch: int, *, gated: bool = True) -> ModelOption:
    """A DINO model on the whole image, CLS and GeM in one pass, bf16 (R1 rerun, no crop)."""
    return ModelOption(
        id_, id_[:2], Family.DINO, model, short, Pooling.CLS_GEM, DType.BFLOAT16, batch,
        gated=gated, resize=Resize.WHOLE,
    )  # fmt: skip


_ALL = (
    _dino("A1-dinov2-l-224-cls", "A1", "facebook/dinov2-large", Pooling.CLS),
    _dino("A1-dinov2-l-224-gem", "A1", "facebook/dinov2-large", Pooling.GEM),
    _dino("A1-dinov2-g-224-cls", "A1", "facebook/dinov2-giant", Pooling.CLS),
    _dino("A1-dinov2-g-224-gem", "A1", "facebook/dinov2-giant", Pooling.GEM),
    _dino("A2-dinov3-l-256-cls", "A2", _V3_L, Pooling.CLS, gated=True, crop=256),
    _dino("A2-dinov3-l-256-gem", "A2", _V3_L, Pooling.GEM, gated=True, crop=256),
    _dino("A2-dinov3-hplus-256-cls", "A2", _V3_HPLUS, Pooling.CLS, gated=True, crop=256),
    _dino("A2-dinov3-hplus-256-gem", "A2", _V3_HPLUS, Pooling.GEM, gated=True, crop=256),
    ModelOption(
        "A2-dinov3-7b-256-cls", "A2", Family.DINO, _V3_7B,
        256, Pooling.CLS, DType.BFLOAT16, 32, gated=True,  # float32 needs a 48 GB GPU
    ),
    _whole("A1-dinov2-g-224-nocrop", "facebook/dinov2-giant", 224, 128, gated=False),
    _whole("A2-dinov3-l-256-nocrop", _V3_L, 256, 128),
    _whole("A2-dinov3-hplus-256-nocrop", _V3_HPLUS, 256, 128),
    _whole("A2-dinov3-7b-256-nocrop", _V3_7B, 256, 64),
    ModelOption(
        "A3-megaloc-322", "A3", Family.MEGALOC, "gmberton/MegaLoc",
        322, Pooling.SALAD, DType.FLOAT32, 64,
    ),
    ModelOption(
        "A4-siglip2-so400m-384", "A4", Family.SIGLIP2, "google/siglip2-so400m-patch14-384",
        384, Pooling.HEAD, DType.BFLOAT16, 64,
    ),
    ModelOption(
        "A4-siglip2-giant-384", "A4", Family.SIGLIP2, "google/siglip2-giant-opt-patch16-384",
        384, Pooling.HEAD, DType.BFLOAT16, 32,
    ),
    ModelOption(
        "A5-pe-core-l-336", "A5", Family.PE_CORE, "timm/PE-Core-L-14-336",
        336, Pooling.HEAD, DType.BFLOAT16, 64,
    ),
    ModelOption(
        "A5-pe-core-g-448", "A5", Family.PE_CORE, "timm/PE-Core-bigG-14-448",
        448, Pooling.HEAD, DType.BFLOAT16, 16,
    ),
    ModelOption(
        "A7-embeddinggemma-2-512", "A7", Family.EMBEDDINGGEMMA, "google/embeddinggemma-2",
        512, Pooling.HEAD, DType.BFLOAT16, 32,
    ),
    ModelOption(
        "A8-qwen3-vl-embedding-2b-512", "A8", Family.QWEN3_VL, "Qwen/Qwen3-VL-Embedding-2B",
        512, Pooling.HEAD, DType.BFLOAT16, 16,
    ),
)  # fmt: skip

OPTIONS: dict[str, ModelOption] = {o.id: o for o in _ALL}

#: MegaLoc loads its code through torch.hub from GitHub, so the exact commit is pinned:
#: a moved default branch would run different code on the GPU machine (HEAD of main,
#: 2026-07-28).
MEGALOC_COMMIT = "5fe0dd697c4a70ba3e23607f6716ab3c606b16db"


def option(option_id: str) -> ModelOption:
    """Look up a gallery option by its id."""
    found = OPTIONS.get(option_id)
    if found is None:
        msg = f"unknown model option {option_id!r}; known: {', '.join(OPTIONS)}"
        raise ValueError(msg)
    return found


def hub_repo(option: ModelOption) -> str:
    """The `torch.hub.load` source: `owner/repo:commit` with the commit pinned for MegaLoc."""
    if option.family is Family.MEGALOC:
        return f"{option.model}:{MEGALOC_COMMIT}"
    return option.model


@dataclass(frozen=True, slots=True)
class SuppliedImage:
    """An image the Mac sends as bytes: a Commons query photo, which the machine never fetches."""

    image_id: str
    jpeg: bytes
    """At most 512 px on its long side."""


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """What one remote call sends back."""

    ids: tuple[str, ...]
    vectors: NDArray[np.float32]
    """One row per id, as the model gave it (not yet normalised)."""
    skips: dict[str, int]
    """Images left out, by reason."""
    load_s: float
    """Seconds spent loading the model in this call; 0 when the machine had it already."""
    embed_s: float
    """Seconds spent fetching and embedding."""
