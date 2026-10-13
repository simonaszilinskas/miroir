"""The Modal app: one GPU function per model family, and the two R0 steps.

Importing this module on the Mac only describes the functions; nothing runs until
`open_modal` starts the app and a caller invokes `.remote()`. Each function body imports
`miroir.remote` inside the container, so no model library is imported on the Mac.

Weights are cached in the `miroir-weights` volume, AmsterTime in `miroir-benchmarks`.
Gated weights (DINOv3) use the `huggingface` Modal secret, which holds `HF_TOKEN`.
"""

from __future__ import annotations

import logging
from collections.abc import Generator, Mapping, Sequence
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
from miroir.eval.wire import (
    CaptionChunk,
    CaptionPhoto,
    DetectChunk,
    Located,
    LocateTask,
    RoomPhoto,
    SiteMatches,
    SiteMatchTask,
)
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
from miroir.outlines import Box, DetectorOption
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
#: The model libraries and weight caches, shared with the website's search app
#: (`miroir.web.modal_app`) so it reuses this image's built layers.
model_libraries, model_caches = _libraries, _CACHES
#: Keypoint matching (step E1): LightGlue from its repository, pinned, without its own
#: dependencies (it asks for the GUI build of OpenCV; the headless one is enough here).
match_libraries = (
    _libraries.apt_install("git")
    .uv_pip_install("kornia==0.8.3", "opencv-python-headless==4.12.0.88", "packaging")
    .pip_install(
        "lightglue @ git+https://github.com/cvg/LightGlue.git@eb42fee2d71449efb0aa5c10549752b5d75384d8",
        extra_options="--no-deps",
    )
    .env(_CACHES)
)
match_image = match_libraries.add_local_python_source("miroir")

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


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def embed_texts(option: ModalOption, ids: list[str], texts: list[str]) -> ChunkResult:
    """Search words (T7 and other word queries), as the website embeds them."""
    from miroir.remote.models import embed_texts as embed  # noqa: PLC0415 (container only)

    result = embed(option, ids, texts)
    if result.load_s > 0:
        weights.commit()
    return result


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


# --- step C: detectors on the room photos; E1 on the website's own shortlists ------------------


@app.function(
    image=match_image,
    gpu=L4_DINO.gpu,
    cpu=L4_DINO.cpu,
    memory=L4_DINO.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_IMAGES_MOUNT: images_volume},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def detect_rooms(
    option: DetectorOption | None,
    photos: list[RoomPhoto],
    given: dict[str, tuple[Box, ...]] | None = None,
) -> DetectChunk:
    from miroir.remote.rooms import detect_rooms as run  # noqa: PLC0415 (container only)

    result = run(option, photos, given or {})
    if result.load_s > 0:
        weights.commit()
    return result


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
def locate_artworks(scenes: list[RoomPhoto], tasks: list[LocateTask]) -> list[Located]:
    from miroir.remote.rooms import locate_artworks as run  # noqa: PLC0415 (container only)

    return run(scenes, tasks)


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
def match_site(tasks: list[SiteMatchTask]) -> list[SiteMatches]:
    from miroir.remote.rerank import match_site_tasks  # noqa: PLC0415 (container only)

    return match_site_tasks(tasks)


# --- archive prints: their caption, read by a small open vision model -------------------------


@app.function(
    image=image,
    gpu=L4.gpu,
    cpu=L4.cpu,
    memory=L4.memory_gib * _MIB,
    timeout=CHUNK_TIMEOUT_S,
    volumes={MODAL_WEIGHTS_MOUNT: weights},
    secrets=[hf_secret],
    max_containers=1,
    scaledown_window=SCALEDOWN_S,
)
def read_captions(model: str, photos: list[CaptionPhoto]) -> CaptionChunk:
    from miroir.remote.caption import read_captions as read  # noqa: PLC0415 (container only)

    result = read(model, photos)
    weights.commit()  # keep downloaded weights for the next container
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

    def embed_texts(
        self, option: ModalOption, ids: Sequence[str], texts: Sequence[str]
    ) -> ChunkResult:
        return embed_texts.remote(option, list(ids), list(texts))

    def scale(self, option: ModalOption, containers: int) -> None:
        """Let the option's function run `containers` containers at once, for this app run."""
        FAMILY_FUNCTIONS[option.family].update_autoscaler(max_containers=containers)

    def fill_images(self, chunks: Sequence[Sequence[Image]]) -> dict[str, int]:
        totals: dict[str, int] = {}
        # A chunk that fails is counted, not fatal: what the others fetched is kept, and a
        # rerun skips images already in the volume.
        for counts in fill_images.map([list(c) for c in chunks], return_exceptions=True):
            if isinstance(counts, BaseException):
                logger.warning("image volume: a chunk failed: %s", counts)
                totals["chunk_failed"] = totals.get("chunk_failed", 0) + 1
                continue
            for key, n in counts.items():
                totals[key] = totals.get(key, 0) + n
        return totals

    def amstertime_download(self) -> int:
        return amstertime_download.remote()

    def amstertime_embed(self, option: ModalOption) -> ChunkResult:
        return amstertime_embed.remote(option)

    def match(self, tasks: Sequence[MatchTask]) -> MatchChunk:
        return match_shortlists.remote(list(tasks))

    def detect_rooms(
        self,
        option: DetectorOption | None,
        photos: Sequence[RoomPhoto],
        given: Mapping[str, tuple[Box, ...]] | None = None,
    ) -> DetectChunk:
        return detect_rooms.remote(option, list(photos), dict(given or {}))

    def locate_artworks(
        self, scenes: Sequence[RoomPhoto], tasks: Sequence[LocateTask]
    ) -> list[Located]:
        return locate_artworks.remote(list(scenes), list(tasks))

    def match_site(self, tasks: Sequence[SiteMatchTask]) -> list[SiteMatches]:
        return match_site.remote(list(tasks))

    def read_captions(self, model: str, photos: Sequence[CaptionPhoto]) -> CaptionChunk:
        return read_captions.remote(model, list(photos))


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
