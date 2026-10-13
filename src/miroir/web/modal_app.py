"""The deployed website: one Modal app, `miroir-web`, with two parts.

- `Search`, a GPU class (L4): loads EmbeddingGemma 2, MegaLoc and the index once per
  container, then answers searches. It also finds the objects in a photo (F4 outlines,
  `detect`): the detector loads in a thread beside the search models, so it never delays a
  search. It sleeps after `SEARCH_IDLE_S` without a search; at most one container runs, so a
  burst queues instead of starting GPUs.
- `web`, a small CPU container: the page, login, record pages and example photos. Only a
  search (or the page's warm-up call) wakes the GPU.

Importing this module on the Mac only describes the app; `miroir web deploy` deploys it.
Model code is imported inside the container (`miroir.remote.search`). Secrets come from the
`miroir-app` Modal secret (APP_PASSWORD, SESSION_SECRET), written by `miroir web secret`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

import modal

from miroir.config import MODAL_WEB_INDEX_MOUNT, MODAL_WEIGHTS_MOUNT, WEB_INDEX
from miroir.jobs.modal_app import HF_SECRET_NAME, model_caches, model_libraries

if TYPE_CHECKING:
    from fastapi import FastAPI

    from miroir.outlines import Outlines
    from miroir.remote.search import OutlineFinder
    from miroir.web.engine import Engine, SearchRequest, SearchResponse

APP_NAME = "miroir-web"
#: The site's address is https://<workspace>--<label>.modal.run.
WEB_LABEL = "miroir"
APP_SECRET_NAME = "miroir-app"  # noqa: S105 (the name of a Modal secret, not its value)
INDEX_VOLUME = "miroir-web-index"
#: A GPU container waits this long for the next search before it stops.
SEARCH_IDLE_S = 10 * 60
#: A photo's outlines wait at most this long for the detector to finish loading (cold start).
DETECTOR_WAIT_S = 60.0
#: The web container waits this long for the next request.
WEB_IDLE_S = 5 * 60
_MIB = 1024


def _source(path: Path) -> bool:
    """Files left out of the containers: caches only (the page's static files go in)."""
    return "__pycache__" in path.parts or path.suffix == ".pyc"


app = modal.App(APP_NAME)
weights = modal.Volume.from_name("miroir-weights", create_if_missing=True)
index_volume = modal.Volume.from_name(INDEX_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
app_secret = modal.Secret.from_name(APP_SECRET_NAME)

gpu_image = model_libraries.env(model_caches).add_local_python_source("miroir", ignore=_source)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "fastapi==0.143.0",
        "numpy==2.5.4",
        "pydantic==2.14.1",
        "pydantic-settings==2.15.0",
        "httpx==0.28.1",
        "pillow==12.3.0",
    )
    .add_local_python_source("miroir", ignore=_source)
)


@app.cls(
    image=gpu_image,
    gpu="L4",
    cpu=4.0,
    memory=16 * _MIB,
    volumes={MODAL_WEIGHTS_MOUNT: weights, MODAL_WEB_INDEX_MOUNT: index_volume},
    secrets=[hf_secret],
    scaledown_window=SEARCH_IDLE_S,
    min_containers=0,
    max_containers=1,
    timeout=300,
)
class Search:
    @modal.enter()  # pyright: ignore[reportUnknownMemberType]
    def load(self) -> None:
        from miroir.logs import setup  # noqa: PLC0415
        from miroir.remote.search import load_search  # noqa: PLC0415 (container only)

        setup()
        self.outlines: OutlineFinder | None = None
        self.outlines_ready = threading.Event()
        threading.Thread(target=self._load_outlines, daemon=True).start()
        self.engine: Engine = load_search(Path(MODAL_WEB_INDEX_MOUNT) / WEB_INDEX)

    def _load_outlines(self) -> None:
        from miroir.remote.search import OutlineFinder  # noqa: PLC0415 (container only)

        try:
            self.outlines = OutlineFinder()
        finally:
            self.outlines_ready.set()

    @modal.method()
    def search(self, request: SearchRequest) -> SearchResponse:
        return self.engine.search(request)

    @modal.method()
    def detect(self, jpeg: bytes) -> Outlines:
        from miroir.outlines import Outlines  # noqa: PLC0415

        self.outlines_ready.wait(DETECTOR_WAIT_S)
        if self.outlines is None:
            return Outlines(())  # the detector failed to load: no outlines, search still works
        return self.outlines.find(jpeg)

    @modal.method()
    def ping(self) -> bool:
        return True


class ModalBackend:
    """The way to the GPU class: from the web container, or from the Mac with `deployed`
    (`miroir web dev --real`: the local page searches with the deployed models)."""

    def __init__(self, *, deployed: bool = False) -> None:
        found = modal.Cls.from_name(APP_NAME, "Search")() if deployed else Search()
        self._search = cast("Search", found)

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self._search.search.remote.aio(request)

    async def warm(self) -> None:
        await self._search.ping.spawn.aio()

    async def detect(self, jpeg: bytes) -> Outlines:
        return await self._search.detect.remote.aio(jpeg)


@app.function(
    image=web_image,
    cpu=1.0,
    memory=4 * _MIB,  # the full catalog: 137,000 images, 101,000 records
    volumes={MODAL_WEB_INDEX_MOUNT: index_volume},
    secrets=[app_secret],
    scaledown_window=WEB_IDLE_S,
    min_containers=0,
    max_containers=1,
    timeout=600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label=WEB_LABEL)  # pyright: ignore[reportUnknownMemberType]
def web() -> FastAPI:
    from miroir.config import Settings  # noqa: PLC0415
    from miroir.logs import setup  # noqa: PLC0415
    from miroir.web.app import create_app  # noqa: PLC0415
    from miroir.web.catalog import CATALOG_FILE, read_catalog  # noqa: PLC0415

    setup()
    settings = Settings()
    if settings.app_password is None or settings.session_secret is None:
        msg = f"the {APP_SECRET_NAME} Modal secret needs APP_PASSWORD and SESSION_SECRET"
        raise RuntimeError(msg)
    return create_app(
        catalog=read_catalog(Path(MODAL_WEB_INDEX_MOUNT) / WEB_INDEX / CATALOG_FILE),
        backend=ModalBackend(),
        password=settings.app_password.get_secret_value(),
        session_key=settings.session_secret.get_secret_value().encode(),
    )
