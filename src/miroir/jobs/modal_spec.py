"""What runs on Modal: the open-model options, their GPUs, and Modal's prices.

Plain data, read on the Mac (to plan, check spend and name runs) and inside the containers
(to load the right model). No Modal or model import here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

# Modal prices, checked 2026-10-12 on modal.com/pricing, in dollars per second.
GPU_USD_PER_S: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-80GB": 0.000694,
}
CPU_CORE_USD_PER_S = 0.0000131
MEMORY_GIB_USD_PER_S = 0.00000222

#: Images per remote call. The Mac logs spend and checks the budget between calls.
CHUNK_IMAGES = 500
#: A call may run this long, model download included, before Modal stops it.
CHUNK_TIMEOUT_S = 20 * 60
#: AmsterTime download (0.9 GB) and unpacking.
DOWNLOAD_TIMEOUT_S = 30 * 60
#: Embedding the 2,462 AmsterTime images, model download included.
R0_TIMEOUT_S = 30 * 60
#: Queries per keypoint-matching call (step E1), each with its shortlist of 50.
MATCH_QUERIES = 40
#: A warm container waits this long for the next call before it shuts down.
SCALEDOWN_S = 120


class Family(StrEnum):
    """Models that load and run the same way. One Modal function each."""

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
    """What one Modal function asks for. Billing counts all three."""

    gpu: str | None
    cpu: float
    memory_gib: int

    @property
    def usd_per_s(self) -> float:
        return (
            (0.0 if self.gpu is None else GPU_USD_PER_S[self.gpu])
            + self.cpu * CPU_CORE_USD_PER_S
            + self.memory_gib * MEMORY_GIB_USD_PER_S
        )


#: Every family function. The cheapest GPU that holds every model here: 24 GB, bf16, and
#: more compute per dollar than a T4.
L4 = Machine(gpu="L4", cpu=4.0, memory_gib=16)
#: R0 embedding: the DINOv3 7B weights (13.4 GB in bf16) pass through host memory.
R0_MACHINE = Machine(gpu="L4", cpu=4.0, memory_gib=32)
#: DINO models: the DINOv3 7B weights (13.4 GB in bf16) pass through host memory.
L4_DINO = Machine(gpu="L4", cpu=4.0, memory_gib=32)
#: R0 download: AmsterTime into a volume, no GPU.
DOWNLOAD_MACHINE = Machine(gpu=None, cpu=1.0, memory_gib=2)
#: Filling the image volume: many downloads at once, no GPU.
FILL_MACHINE = Machine(gpu=None, cpu=2.0, memory_gib=4)
#: Containers filling the image volume at once, and downloads in each.
FILL_CONTAINERS = 8
FILL_CONCURRENCY = 16
FILL_TIMEOUT_S = 20 * 60


@dataclass(frozen=True, slots=True)
class ModalOption:
    """One way to embed images on Modal."""

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
    """Weights need an accepted licence on Hugging Face (read through the `huggingface` secret)."""
    resize: Resize = Resize.CENTRE_CROP
    """DINO only: crop to a square, or keep the whole image."""

    @property
    def machine(self) -> Machine:
        """What its family's Modal function asks for."""
        return L4_DINO if self.family is Family.DINO else L4


def _dino(
    id_: str, experiment: str, model: str, pooling: Pooling, *, gated: bool = False, crop: int = 224
) -> ModalOption:
    return ModalOption(
        id_, experiment, Family.DINO, model, crop, pooling, DType.FLOAT32, 64, gated=gated
    )


_V3_L = "facebook/dinov3-vitl16-pretrain-lvd1689m"
_V3_HPLUS = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
_V3_7B = "facebook/dinov3-vit7b16-pretrain-lvd1689m"


def _whole(id_: str, model: str, short: int, batch: int, *, gated: bool = True) -> ModalOption:
    """A DINO model on the whole image, CLS and GeM in one pass, bf16 (R1 rerun, no crop)."""
    return ModalOption(
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
    ModalOption(
        "A2-dinov3-7b-256-cls", "A2", Family.DINO, _V3_7B,
        256, Pooling.CLS, DType.BFLOAT16, 32, gated=True,  # float32 needs a 48 GB GPU
    ),
    _whole("A1-dinov2-g-224-nocrop", "facebook/dinov2-giant", 224, 128, gated=False),
    _whole("A2-dinov3-l-256-nocrop", _V3_L, 256, 128),
    _whole("A2-dinov3-hplus-256-nocrop", _V3_HPLUS, 256, 128),
    _whole("A2-dinov3-7b-256-nocrop", _V3_7B, 256, 64),
    ModalOption(
        "A3-megaloc-322", "A3", Family.MEGALOC, "gmberton/MegaLoc",
        322, Pooling.SALAD, DType.FLOAT32, 64,
    ),
    ModalOption(
        "A4-siglip2-so400m-384", "A4", Family.SIGLIP2, "google/siglip2-so400m-patch14-384",
        384, Pooling.HEAD, DType.BFLOAT16, 64,
    ),
    ModalOption(
        "A4-siglip2-giant-384", "A4", Family.SIGLIP2, "google/siglip2-giant-opt-patch16-384",
        384, Pooling.HEAD, DType.BFLOAT16, 32,
    ),
    ModalOption(
        "A5-pe-core-l-336", "A5", Family.PE_CORE, "timm/PE-Core-L-14-336",
        336, Pooling.HEAD, DType.BFLOAT16, 64,
    ),
    ModalOption(
        "A5-pe-core-g-448", "A5", Family.PE_CORE, "timm/PE-Core-bigG-14-448",
        448, Pooling.HEAD, DType.BFLOAT16, 16,
    ),
    ModalOption(
        "A7-embeddinggemma-2-512", "A7", Family.EMBEDDINGGEMMA, "google/embeddinggemma-2",
        512, Pooling.HEAD, DType.BFLOAT16, 32,
    ),
    ModalOption(
        "A8-qwen3-vl-embedding-2b-512", "A8", Family.QWEN3_VL, "Qwen/Qwen3-VL-Embedding-2B",
        512, Pooling.HEAD, DType.BFLOAT16, 16,
    ),
)  # fmt: skip

OPTIONS: dict[str, ModalOption] = {o.id: o for o in _ALL}

# Round R0 (T9): AmsterTime with the DINOv3 paper's protocol (§D.8): shorter side 256,
# centre crop 224, CLS token, cosine. Expected mAP from its Tables 9 and 23.
R0_OPTIONS: dict[str, ModalOption] = {
    "dinov2-giant-224": _dino("R0-dinov2-giant-224", "A1", "facebook/dinov2-giant", Pooling.CLS),
    "dinov3-vit7b16-224": ModalOption(
        "R0-dinov3-vit7b16-224",
        "A2",
        Family.DINO,
        _V3_7B,
        224,
        Pooling.CLS,
        DType.BFLOAT16,  # 26.9 GB in float32 would need a 48 GB GPU
        32,
        gated=True,
    ),
}
R0_EXPECTED_MAP: dict[str, float] = {"dinov2-giant-224": 48.9, "dinov3-vit7b16-224": 56.5}


def option(option_id: str) -> ModalOption:
    """Look up a gallery option or an R0 option by its id."""
    found = OPTIONS.get(option_id) or {o.id: o for o in R0_OPTIONS.values()}.get(option_id)
    if found is None:
        msg = f"unknown Modal option {option_id!r}; known: {', '.join(OPTIONS)}"
        raise ValueError(msg)
    return found


@dataclass(frozen=True, slots=True)
class SuppliedImage:
    """An image the Mac sends as bytes: a Commons query photo, which containers never fetch."""

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
    """Seconds spent loading the model in this call; 0 when the container had it already."""
    embed_s: float
    """Seconds spent fetching and embedding."""
