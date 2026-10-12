"""The Modal app: one GPU function per model family, and the two R0 steps.

Importing this module on the Mac only describes the functions; nothing runs until
`open_modal` starts the app and a caller invokes `.remote()`. Each function body imports
`miroir.remote` inside the container, so no model library is imported on the Mac.

Weights are cached in the `miroir-weights` volume, AmsterTime in `miroir-benchmarks`.
Gated weights (DINOv3) use the `huggingface` Modal secret, which holds `HF_TOKEN`.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import modal

from miroir.config import (
    MODAL_BENCHMARKS_MOUNT,
    MODAL_IMAGES_MOUNT,
    MODAL_WEIGHTS_MOUNT,
    Settings,
)
from miroir.eval.rerank import MatchChunk, MatchTask
from miroir.jobs.modal_spec import (
    CHUNK_TIMEOUT_S,
    DOWNLOAD_MACHINE,
    DOWNLOAD_TIMEOUT_S,
    FILL_CONCURRENCY,
    FILL_CONTAINERS,
    FILL_MACHINE,
    FILL_TIMEOUT_S,
    L4,
    L4_DINO,
    R0_MACHINE,
    R0_TIMEOUT_S,
    SCALEDOWN_S,
    ChunkResult,
    Family,
    ModalOption,
    SuppliedImage,
)
from miroir.records.model import Image

logger = logging.getLogger(__name__)

APP_NAME = "miroir"
HF_SECRET_NAME = "huggingface"  # noqa: S105 (the name of a Modal secret, not its value)

app = modal.App(APP_NAME)

_libraries = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
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
_CACHES = {"HF_HOME": f"{MODAL_WEIGHTS_MOUNT}/hf", "TORCH_HOME": f"{MODAL_WEIGHTS_MOUNT}/torch"}
image = _libraries.env(_CACHES).add_local_python_source("miroir")
#: Keypoint matching (step E1): LightGlue from its repository, pinned, without its own
#: dependencies (it asks for the GUI build of OpenCV; the headless one is enough here).
match_image = (
    _libraries.apt_install("git")
    .uv_pip_install("kornia==0.8.3", "opencv-python-headless==4.12.0.88", "packaging")
    .pip_install(
        "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8",
        extra_options="--no-deps",
    )
    .env(_CACHES)
    .add_local_python_source("miroir")
)

weights = modal.Volume.from_name("miroir-weights", create_if_missing=True)
benchmarks = modal.Volume.from_name("miroir-benchmarks", create_if_missing=True)
images_volume = modal.Volume.from_name("miroir-images", create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

_MIB = 1024


# --- POP images, one function per family ----------------------------------------------------
# modal_spec.ModalOption.machine says what each asks for; max_containers=1 keeps one warm
# container that loads its model once and takes the chunks one after another. Images come
# from the `miroir-images` volume when it holds them, else from POP. Commons query photos
# come as bytes (`supplied`): the Mac fetches them from Wikimedia at its polite pace.


def _embed(option: ModalOption, images: list[Image], supplied: list[SuppliedImage]) -> ChunkResult:
    from miroir.remote.models import embed_images  # noqa: PLC0415 (container only)

    result = embed_images(option, images, supplied)
    if result.load_s > 0:
        weights.commit()  # keep downloaded weights for the next container
    return result


@app.function(
    image=image,
    gpu=L4_DINO.gpu,
    cpu=L4_DINO.cpu,
    memory=L4_DINO.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_dino(
    option: ModalOption, images: list[Image], supplied: list[SuppliedImage] | None = None
) -> ChunkResult:
    return _embed(option, images, supplied or [])


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_siglip2(
    option: ModalOption, images: list[Image], supplied: list[SuppliedImage] | None = None
) -> ChunkResult:
    return _embed(option, images, supplied or [])


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_pe_core(
    option: ModalOption, images: list[Image], supplied: list[SuppliedImage] | None = None
) -> ChunkResult:
    return _embed(option, images, supplied or [])


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_embeddinggemma(
    option: ModalOption, images: list[Image], supplied: list[SuppliedImage] | None = None
) -> ChunkResult:
    return _embed(option, images, supplied or [])


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_qwen3_vl(
    option: ModalOption, images: list[Image], supplied: list[SuppliedImage] | None = None
) -> ChunkResult:
    return _embed(option, images, supplied or [])


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_megaloc(
    option: ModalOption, images: list[Image], supplied: list[SuppliedImage] | None = None
) -> ChunkResult:
    return _embed(option, images, supplied or [])


# --- step E1: keypoint matching of each query against its shortlist ------------------------


@app.function(
    image=match_image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def match_shortlists(tasks: list[MatchTask]) -> MatchChunk:
    from miroir.remote.matching import match_tasks  # noqa: PLC0415 (container only)

    result = match_tasks(tasks)
    if result.load_s > 0:
        weights.commit()  # keep the ALIKED and LightGlue weights for the next container
    return result


FAMILY_FUNCTIONS = {
    Family.DINO: embed_dino,
    Family.SIGLIP2: embed_siglip2,
    Family.PE_CORE: embed_pe_core,
    Family.EMBEDDINGGEMMA: embed_embeddinggemma,
    Family.QWEN3_VL: embed_qwen3_vl,
    Family.MEGALOC: embed_megaloc,
}


# --- the image volume: every screening image at 512 px, fetched once ----------------------


@app.function(
    image=image,
    cpu=FILL_MACHINE.cpu,
    memory=FILL_MACHINE.memory_gib * _MIB,
    timeout=FILL_TIMEOUT_S,
    volumes={MODAL_IMAGES_MOUNT: images_volume},
    max_containers=FILL_CONTAINERS,
)
def fill_images(images: list[Image]) -> dict[str, int]:
    """Fetch images (512 px, IIIF fallback) into the image volume. Returns counts by outcome."""
    from miroir.remote.models import fill_image_volume  # noqa: PLC0415

    counts = fill_image_volume(images, FILL_CONCURRENCY)
    images_volume.commit()
    return counts


# --- R0: AmsterTime, kept in a volume, never on the Mac -------------------------------------


@app.function(
    image=image,
    cpu=DOWNLOAD_MACHINE.cpu,
    memory=DOWNLOAD_MACHINE.memory_gib * _MIB,
    timeout=DOWNLOAD_TIMEOUT_S,
    volumes={MODAL_BENCHMARKS_MOUNT: benchmarks},
)
def amstertime_download() -> int:
    """Fetch AmsterTime into the benchmarks volume (skipped when there). Returns the image count."""
    from miroir.eval import amstertime  # noqa: PLC0415

    images = amstertime.download(Path(MODAL_BENCHMARKS_MOUNT) / "amstertime")
    benchmarks.commit()
    return len(amstertime.image_ids(images))


@app.function(
    image=image,
    gpu=R0_MACHINE.gpu,
    cpu=R0_MACHINE.cpu,
    memory=R0_MACHINE.memory_gib * _MIB,
    timeout=R0_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_BENCHMARKS_MOUNT: benchmarks},
    secrets=[hf_secret],
)
def amstertime_embed(option: ModalOption) -> ChunkResult:
    """Embed every AmsterTime image from the volume, ids `old/NNNN.jpg` and `new/NNNN.png`."""
    from miroir.eval import amstertime  # noqa: PLC0415
    from miroir.remote.models import embed_files  # noqa: PLC0415

    benchmarks.reload()
    images = Path(MODAL_BENCHMARKS_MOUNT) / "amstertime" / "images"
    files = [(i, images / i) for i in amstertime.image_ids(images)]
    result = embed_files(option, files)
    weights.commit()
    return result


# --- calling it from the Mac ----------------------------------------------------------------


class ModalCalls:
    """The remote calls a run makes, inside a running app. `miroir.jobs.modal_run` uses these."""

    def embed(
        self,
        option: ModalOption,
        images: Sequence[Image],
        supplied: Sequence[SuppliedImage] = (),
    ) -> ChunkResult:
        return FAMILY_FUNCTIONS[option.family].remote(option, list(images), list(supplied))

    def fill_images(self, chunks: Sequence[Sequence[Image]]) -> dict[str, int]:
        totals: dict[str, int] = {}
        for counts in fill_images.map([list(c) for c in chunks]):
            for key, n in counts.items():
                totals[key] = totals.get(key, 0) + n
        return totals

    def amstertime_download(self) -> int:
        return amstertime_download.remote()

    def amstertime_embed(self, option: ModalOption) -> ChunkResult:
        return amstertime_embed.remote(option)

    def match(self, tasks: Sequence[MatchTask]) -> MatchChunk:
        return match_shortlists.remote(list(tasks))


@contextmanager
def open_modal(settings: Settings) -> Generator[ModalCalls]:
    """Start the app (ephemeral: it stops when the block ends) and yield its calls."""
    if not settings.modal_configured:
        msg = (
            "Modal is not set up: run `uv run modal token new`, or set MODAL_TOKEN_ID and "
            "MODAL_TOKEN_SECRET in .env"
        )
        raise RuntimeError(msg)
    client = None
    if settings.modal_token_id is not None and settings.modal_token_secret is not None:
        client = modal.Client.from_credentials(
            settings.modal_token_id.get_secret_value(),
            settings.modal_token_secret.get_secret_value(),
        )
    with modal.enable_output(), app.run(client=client):
        yield ModalCalls()
