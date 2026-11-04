# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""The job server on a Scaleway GPU machine (`miroir scw serve`, inside the job image).

The GPU jobs (embedding, matching, detection, captions) and the website's search, behind one HTTP
port on the machine's loopback; the Mac reaches it through an SSH tunnel
(`miroir.jobs.scw_run`). A call is a pickle of (name, arguments) and its answer the pickled
result, or HTTP 500 with the traceback. Calls run one at a time on the one GPU.

Files: Object Storage buckets, mirrored into the machine's folders (`miroir.jobs.objects`). The
weights bucket is copied down at start and new weights go back after a call that loaded a
model; a call's POP images come from the images bucket when it has them, and those fetched
from POP go back after the call. The machine deletes itself past its deadline, so a Mac that
died mid-run cannot leave it running.
"""

from __future__ import annotations

import logging
import pickle
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from miroir.config import (
    BENCHMARKS_MOUNT,
    IMAGES_MOUNT,
    WEB_INDEX_MOUNT,
    WEIGHTS_MOUNT,
    Settings,
)
from miroir.images.fetch import cache_key
from miroir.jobs.objects import (
    Bucket,
    bucket_name,
    download_keys,
    download_tree,
    s3_client,
    upload_files,
    upload_tree,
)
from miroir.jobs.remote_spec import FILL_CONCURRENCY, ChunkResult, ModelOption, SuppliedImage
from miroir.records.model import Image

if TYPE_CHECKING:
    from miroir.remote.search import SearchService
    from miroir.web.engine import SearchRequest
    from miroir.web.toolsearch import ToolQuery

logger = logging.getLogger(__name__)

WATCH_S = 30.0


class Worker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.s3 = s3_client(settings)
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self._search: dict[str, object] = {}
        self.calls: dict[str, Callable[..., object]] = {
            "embed": self.embed,
            "embed_texts": self.embed_texts,
            "fill_images": self.fill_images,
            "amstertime_download": self.amstertime_download,
            "amstertime_embed": self.amstertime_embed,
            "match": self.match,
            "detect_rooms": self.detect_rooms,
            "locate_artworks": self.locate_artworks,
            "match_site": self.match_site,
            "read_captions": self.read_captions,
            "search": self.search,
            "detect": self.detect,
            "tool_search": self.tool_search,
            "ping": lambda *_: True,
            "persist": self.persist,
        }

    def _bucket(self, bucket: Bucket) -> str:
        return bucket_name(self.settings, bucket)

    def start(self) -> None:
        for mount in (WEIGHTS_MOUNT, IMAGES_MOUNT, BENCHMARKS_MOUNT):
            Path(mount).mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        count = download_tree(self.s3, self._bucket(Bucket.WEIGHTS), Path(WEIGHTS_MOUNT))
        logger.info("weights: %d files in %.0f s", count, time.monotonic() - start)
        self.ready.set()

    # --- files --------------------------------------------------------------------------

    def _save_weights(self, load_s: float) -> None:
        """Keep downloaded weights for the next machine. A failure only costs a download."""
        if load_s <= 0:
            return
        try:
            upload_tree(self.s3, self._bucket(Bucket.WEIGHTS), Path(WEIGHTS_MOUNT))
        except Exception:
            logger.exception("weights not saved to the bucket; the next machine downloads them")

    def _images_down(self, images: Sequence[Image]) -> set[str]:
        """Copy the images the bucket holds into the image folder. Returns the keys it lacks."""
        keys = [cache_key(i.image_id) for i in images]
        return download_keys(self.s3, self._bucket(Bucket.IMAGES), Path(IMAGES_MOUNT), keys)

    def _images_up(self, keys: set[str]) -> None:
        """Send back the images fetched from POP during the call."""
        root = Path(IMAGES_MOUNT)
        fetched = [root / k for k in keys if (root / k).exists()]
        upload_files(self.s3, self._bucket(Bucket.IMAGES), root, fetched)

    def persist(self) -> dict[str, int]:
        weights = upload_tree(self.s3, self._bucket(Bucket.WEIGHTS), Path(WEIGHTS_MOUNT))
        return {"weights": weights}

    # --- the jobs ----------------------------------------------------------------------------

    def embed(
        self, option: ModelOption, images: list[Image], supplied: list[SuppliedImage]
    ) -> ChunkResult:
        from miroir.remote.models import embed_images  # noqa: PLC0415

        lacking = self._images_down(images)
        result = embed_images(option, images, supplied)
        self._images_up(lacking)
        self._save_weights(result.load_s)
        return result

    def embed_texts(self, option: ModelOption, ids: list[str], texts: list[str]) -> ChunkResult:
        from miroir.remote.models import embed_texts  # noqa: PLC0415

        result = embed_texts(option, ids, texts)
        self._save_weights(result.load_s)
        return result

    def fill_images(self, chunks: list[list[Image]]) -> dict[str, int]:
        from miroir.remote.models import fill_image_volume  # noqa: PLC0415

        totals: dict[str, int] = {}
        for chunk in chunks:
            lacking = self._images_down(chunk)
            counts = fill_image_volume(chunk, FILL_CONCURRENCY)
            self._images_up(lacking)
            for key, n in counts.items():
                totals[key] = totals.get(key, 0) + n
        return totals

    def amstertime_download(self) -> int:
        from miroir.eval import amstertime  # noqa: PLC0415

        root = Path(BENCHMARKS_MOUNT)
        download_tree(self.s3, self._bucket(Bucket.BENCHMARKS), root)
        images = amstertime.download(root / "amstertime")
        upload_tree(self.s3, self._bucket(Bucket.BENCHMARKS), root)
        return len(amstertime.image_ids(images))

    def amstertime_embed(self, option: ModelOption) -> ChunkResult:
        from miroir.eval import amstertime  # noqa: PLC0415
        from miroir.remote.models import embed_files  # noqa: PLC0415

        root = Path(BENCHMARKS_MOUNT)
        download_tree(self.s3, self._bucket(Bucket.BENCHMARKS), root)
        images = root / "amstertime" / "images"
        result = embed_files(option, [(i, images / i) for i in amstertime.image_ids(images)])
        self._save_weights(result.load_s)
        return result

    def match(self, tasks: list) -> object:
        from miroir.remote.matching import match_tasks  # noqa: PLC0415

        result = match_tasks(tasks)
        self._save_weights(result.load_s)
        return result

    def detect_rooms(self, option, photos: list, given: dict) -> object:
        from miroir.remote.rooms import detect_rooms  # noqa: PLC0415

        result = detect_rooms(option, photos, given)
        self._save_weights(result.load_s)
        return result

    def locate_artworks(self, scenes: list, tasks: list) -> object:
        from miroir.remote.rooms import locate_artworks  # noqa: PLC0415

        return locate_artworks(scenes, tasks)

    def match_site(self, tasks: list) -> object:
        from miroir.remote.rerank import match_site_tasks  # noqa: PLC0415

        return match_site_tasks(tasks)

    def read_captions(self, model: str, photos: list) -> object:
        from miroir.remote.caption import read_captions  # noqa: PLC0415

        result = read_captions(model, photos)
        self._save_weights(1.0)
        return result

    # --- the website's search (`miroir web dev --scw`) ---------------------------------------

    def _service(self, index: str) -> SearchService:
        from miroir.remote.search import SearchService  # noqa: PLC0415

        found = self._search.get(index)
        if found is None:
            root = Path(WEB_INDEX_MOUNT)
            download_tree(self.s3, self._bucket(Bucket.INDEX), root, prefix=f"{index}/")
            found = SearchService(root / index)
            self._save_weights(1.0)
            self._search.clear()  # one index on the GPU at a time
            self._search[index] = found
        assert isinstance(found, SearchService)
        return found

    def search(self, index: str, request: SearchRequest) -> object:
        return self._service(index).engine.search(request)

    def detect(self, index: str, jpeg: bytes) -> object:
        return self._service(index).detect(jpeg)

    def tool_search(self, index: str, query: ToolQuery) -> object:
        return self._service(index).tool_search(query)


# --- the HTTP side ---------------------------------------------------------------------------


def _handler(worker: Worker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            ok = self.path == "/health" and worker.ready.is_set()
            self._send(200 if ok else 503, b"ok" if ok else b"starting")

        def do_POST(self) -> None:
            if self.path != "/call":
                self._send(404, b"")
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            name, args = pickle.loads(body)  # noqa: S301 (reachable only through SSH)
            start = time.monotonic()
            try:
                with worker.lock:
                    result = worker.calls[name](*args)
            except Exception:
                logger.exception("%s failed", name)
                self._send(500, traceback.format_exc().encode())
                return
            logger.info("%s: %.1f s", name, time.monotonic() - start)
            self._send(200, pickle.dumps(result))

        def _send(self, status: int, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: object) -> None:
            logger.debug(format, *args)

    return Handler


def watch_deadline(settings: Settings, server_id: str, zone: str, deadline: float) -> None:
    """Delete this machine once its deadline passes (the Mac normally deletes it first)."""
    from miroir.jobs.scaleway import InstanceApi, delete_machine  # noqa: PLC0415

    while time.time() < deadline:
        time.sleep(WATCH_S)
    logger.error("deadline passed: deleting this machine")
    api = InstanceApi(settings)
    delete_machine(api, zone, server_id)


def serve(settings: Settings, *, port: int, server_id: str, zone: str, deadline: float) -> None:
    worker = Worker(settings)
    threading.Thread(
        target=watch_deadline, args=(settings, server_id, zone, deadline), daemon=True
    ).start()
    server = ThreadingHTTPServer(("0.0.0.0", port), _handler(worker))  # noqa: S104 (Docker maps it to loopback)

    def start() -> None:
        try:
            worker.start()
        except Exception:
            logger.exception("start-up failed")
            server.shutdown()  # the container stops; the Mac sees it and shows these logs

    threading.Thread(target=start, daemon=True).start()
    logger.info("worker on port %d", port)
    server.serve_forever()
