"""Jobs on a Scaleway GPU machine, driven from the Mac.

`open_scaleway` starts a machine (`miroir.jobs.scaleway.gpu_machine`), lets the Mac in over
SSH with a key made for this machine alone, builds the job image there (`docker/Dockerfile`,
with this checkout's code), starts the worker (`miroir.remote.worker`) and opens an SSH tunnel
to it. It yields `ScalewayCalls`, the calls `GpuRunner` (`miroir.jobs.gpu_run`) and the step C,
E1 and archive runs make. When the block ends, new weights and images go back to their buckets and
the machine is deleted, whatever happened.

The worker listens only on the machine's loopback; the tunnel is the one way in. Calls travel
as pickles between two copies of the same code. The keys the worker needs (Object Storage, and
deleting its own machine at its deadline) go over SSH into a root-only file, never into the
cloud-init data or a command line.
"""

from __future__ import annotations

import io
import logging
import pickle
import shlex
import shutil
import socket
import subprocess
import tarfile
import time
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import httpx

from miroir.config import REPO_ROOT, WEB_UI_BUILD, WEB_UI_SOURCE, Settings
from miroir.jobs.remote_spec import (
    CHUNK_TIMEOUT_S,
    LIGHTGLUE,
    MATCH_LIBRARIES,
    MODEL_LIBRARIES,
    ChunkResult,
    Machine,
    ModelOption,
    SuppliedImage,
)
from miroir.jobs.scaleway import (
    EUR_TO_USD,
    OFFERS,
    InstanceApi,
    Offer,
    gpu_machine,
)
from miroir.jobs.scaleway import Machine as GpuMachine
from miroir.records.model import Image
from miroir.web.pagebuild import check_built

if TYPE_CHECKING:
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
    from miroir.outlines import Box, DetectorOption, Outlines
    from miroir.web.engine import SearchRequest, SearchResponse
    from miroir.web.toolsearch import ToolHits, ToolQuery

logger = logging.getLogger(__name__)

#: The worker's port, on the machine's loopback only.
WORKER_PORT = 8700
#: Folders on the machine: the build context, and the bucket mirrors mounted into the worker.
REMOTE_ROOT = "/opt/miroir"
IMAGE_TAG = "miroir-jobs"
#: Libraries the job image adds to the model libraries, pinned as in uv.lock: the package's
#: own base dependencies, boto3, and the website's (the demo machine serves it from this image).
WORKER_LIBRARIES = (
    "pyarrow==26.0.0",
    "typer==0.27.3",
    "boto3==1.43.123",
    "fastapi==0.143.0",
    "starlette==1.7.0",
    "uvicorn==0.54.0",
)
SSH_TIMEOUT_S = 10 * 60
BUILD_TIMEOUT_S = 40 * 60
#: Startup of the worker: it first copies the weights bucket down.
READY_TIMEOUT_S = 20 * 60
#: A call may take this long on the machine, plus the tunnel's own time.
CALL_TIMEOUT_S = CHUNK_TIMEOUT_S + 120
#: PyTorch wheel indexes by the newest CUDA the machine's driver runs, newest first.
TORCH_INDEXES = ((13.0, "cu130"), (12.8, "cu128"), (12.6, "cu126"))
#: Files of the checkout the image is built from.
CONTEXT_FILES = ("pyproject.toml", "README.md", "LICENSE")
#: The bucket mirrors, mounted into the job containers at the paths the model code reads
#: (`config.WEIGHTS_MOUNT`, ...).
MOUNTS = " ".join(
    f"-v {REMOTE_ROOT}/{name}:/{name}" for name in ("weights", "images", "benchmarks", "web-index")
)


class Shell:
    """Commands on the machine over SSH, as root, with this machine's own key."""

    def __init__(self, host: str, key: Path, known_hosts: Path) -> None:
        self.host = host
        self._options = [
            "-i", str(key),
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=30",
        ]  # fmt: skip

    def run(
        self, command: str, *, stdin: bytes | None = None, timeout: float = 120, check: bool = True
    ) -> str:
        done = subprocess.run(  # noqa: S603 (ssh with our own arguments)
            ["ssh", *self._options, f"root@{self.host}", command],  # noqa: S607
            input=stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        out = done.stdout.decode(errors="replace")
        if check and done.returncode != 0:
            err = done.stderr.decode(errors="replace")[-2000:]
            msg = (
                f"on the machine, `{command[:80]}` failed ({done.returncode}):\n{out[-2000:]}{err}"
            )
            raise RuntimeError(msg)
        return out

    def wait(self) -> None:
        """Until SSH lets us in: cloud-init adds the key a little after boot."""
        end = time.monotonic() + SSH_TIMEOUT_S
        while time.monotonic() < end:
            try:
                self.run("true", timeout=30)
                return
            except (RuntimeError, subprocess.TimeoutExpired):
                time.sleep(5)
        msg = f"no SSH to {self.host} after {SSH_TIMEOUT_S} s"
        raise TimeoutError(msg)

    @contextmanager
    def tunnel(self, remote_port: int) -> Generator[int]:
        """Forward a free local port to the machine's loopback port. Yields the local port."""
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            local = int(probe.getsockname()[1])
        forward = f"127.0.0.1:{local}:127.0.0.1:{remote_port}"
        process = subprocess.Popen(  # noqa: S603
            ["ssh", *self._options, "-N", "-o", "ExitOnForwardFailure=yes", "-L", forward,  # noqa: S607
             f"root@{self.host}"],
        )  # fmt: skip
        try:
            yield local
        finally:
            process.terminate()
            process.wait(timeout=10)


# --- building and starting the worker -------------------------------------------------------


def requirements() -> str:
    return "\n".join((*MODEL_LIBRARIES, *WORKER_LIBRARIES)) + "\n"


def build_context(*, page: bool = False) -> bytes:
    """The Dockerfile, the pinned requirements and this checkout's package, as a .tar.gz.

    The package carries the site's built page (`miroir.web.pagebuild`): with `page`, refuse
    to ship one that is missing or older than its sources."""
    if page:
        check_built(WEB_UI_SOURCE, WEB_UI_BUILD)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        tar.add(REPO_ROOT / "docker" / "Dockerfile", arcname="Dockerfile")
        for name in CONTEXT_FILES:
            tar.add(REPO_ROOT / name, arcname=name)
        source = REPO_ROOT / "src" / "miroir"
        for path in sorted(source.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                tar.add(path, arcname=str(path.relative_to(REPO_ROOT)))
        texts = {
            "requirements.txt": requirements(),
            "match.txt": "\n".join(MATCH_LIBRARIES) + "\n",
            "lightglue.txt": LIGHTGLUE + "\n",
        }
        for name, text in texts.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def torch_index(cuda: float) -> str:
    """The PyTorch wheel index for the newest CUDA the driver runs."""
    for least, tag in TORCH_INDEXES:
        if cuda >= least:
            return f"https://download.pytorch.org/whl/{tag}"
    msg = f"the machine's driver runs CUDA {cuda}; the wheels need {TORCH_INDEXES[-1][0]} or newer"
    raise RuntimeError(msg)


def worker_env(settings: Settings) -> str:
    """The worker's settings, keys included. Written to a root-only file, never shown."""
    return "".join(f"{k}={v}\n" for k, v in settings.scaleway_worker_env().items())


def send_env(shell: Shell, settings: Settings) -> None:
    """The worker's settings and keys, into a root-only file on the machine."""
    shell.run(
        f"umask 077 && mkdir -p {REMOTE_ROOT} && cat > {REMOTE_ROOT}/env",
        stdin=worker_env(settings).encode(),
    )


def start_worker(
    shell: Shell, settings: Settings, machine: GpuMachine, *, match: bool = False
) -> None:
    """Send the keys and the build context, build the image, start the worker container.

    `match` adds the keypoint-matching libraries (steps C and E1)."""
    build_image(shell, settings, match=match)
    shell.run(
        f"docker run -d --name miroir-worker --gpus all --shm-size 8g "
        f"--env-file {REMOTE_ROOT}/env -p 127.0.0.1:{WORKER_PORT}:{WORKER_PORT} {MOUNTS} "
        f"{IMAGE_TAG} --port {WORKER_PORT} --server-id {machine.server_id} "
        f"--zone {machine.offer.zone} --deadline {machine.deadline:.0f}"
    )


def build_image(
    shell: Shell,
    settings: Settings,
    *,
    match: bool = False,
    torch_tag: str = "",
    page: bool = False,
) -> None:
    """Send the keys and the build context, and build the job image on the machine.
    `page`: the image serves the site, so its built page must be there and current (the demo
    machine). `torch_tag` (e.g. "cu126") overrides the wheels picked from the driver: an older GPU
    needs a build that still carries its kernels."""
    send_env(shell, settings)
    smi = shell.run("nvidia-smi", check=False)
    if "CUDA Version:" in smi:
        logger.info("GPU: %s", " ".join(line.strip() for line in smi.splitlines()[2:4]))
        cuda = float(smi.split("CUDA Version:")[1].split()[0])
    else:  # the demo machine on a type without a GPU: the newest wheels run on the CPU too
        logger.info("no GPU: building with the newest PyTorch wheels")
        cuda = TORCH_INDEXES[0][0]
    index = f"https://download.pytorch.org/whl/{torch_tag}" if torch_tag else torch_index(cuda)
    shell.run(
        f"rm -rf {REMOTE_ROOT}/build && mkdir -p {REMOTE_ROOT}/build "
        f"&& tar -xzf - -C {REMOTE_ROOT}/build",
        stdin=build_context(page=page),
    )
    start = time.monotonic()
    logger.info("building the job image on the machine (a few minutes)")
    shell.run(
        f"cd {REMOTE_ROOT}/build && docker build -t {IMAGE_TAG} "
        f"--build-arg TORCH_INDEX={shlex.quote(index)} "
        f"--build-arg MATCH={int(match)} . "
        f"> {REMOTE_ROOT}/build.log 2>&1 || (tail -40 {REMOTE_ROOT}/build.log; exit 1)",
        timeout=BUILD_TIMEOUT_S,
    )
    logger.info("image built in %.0f s", time.monotonic() - start)


# --- the calls ------------------------------------------------------------------------------


class ScalewayCalls:
    """The remote calls of a job, sent to the worker through the tunnel."""

    def __init__(self, url: str, machine: GpuMachine) -> None:
        self._http = httpx.Client(base_url=url, timeout=CALL_TIMEOUT_S)
        self.gpu = machine
        self.busy_s = 0.0
        """Seconds spent in calls."""

    @property
    def machine(self) -> Machine:
        """The whole machine at its price, in dollars, as `GpuRunner` counts."""
        offer = self.gpu.offer
        return Machine(gpu=offer.commercial_type, usd_per_s=offer.eur_per_s * EUR_TO_USD)

    def close(self) -> None:
        self._http.close()

    def wait_ready(self, shell: Shell) -> None:
        end = time.monotonic() + READY_TIMEOUT_S
        while time.monotonic() < end:
            try:
                if self._http.get("/health", timeout=10).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            state = shell.run("docker inspect -f '{{.State.Running}}' miroir-worker", check=False)
            if state.strip() == "false":
                logs = shell.run("docker logs --tail 60 miroir-worker 2>&1", check=False)
                msg = f"the worker stopped while starting:\n{logs}"
                raise RuntimeError(msg)
            time.sleep(5)
        msg = f"the worker was not ready after {READY_TIMEOUT_S} s"
        raise TimeoutError(msg)

    def call(self, name: str, *args: object) -> object:
        start = time.monotonic()
        try:
            response = self._http.post("/call", content=pickle.dumps((name, args)))
        finally:
            self.busy_s += time.monotonic() - start
        if response.status_code != 200:
            msg = f"{name} failed on the machine:\n{response.text[-3000:]}"
            raise RuntimeError(msg)
        return pickle.loads(response.content)  # noqa: S301 (our own worker, through SSH)

    def persist(self) -> dict[str, int]:
        """Send new weights and images back to their buckets."""
        return cast("dict[str, int]", self.call("persist"))

    # --- `RemoteCalls` (miroir.jobs.gpu_run) and the calls of steps C, E1 and the archive ------

    def embed(
        self, option: ModelOption, images: Sequence[Image], supplied: Sequence[SuppliedImage] = ()
    ) -> ChunkResult:
        return cast("ChunkResult", self.call("embed", option, list(images), list(supplied)))

    def embed_texts(
        self, option: ModelOption, ids: Sequence[str], texts: Sequence[str]
    ) -> ChunkResult:
        return cast("ChunkResult", self.call("embed_texts", option, list(ids), list(texts)))

    def fill_images(self, chunks: Sequence[Sequence[Image]]) -> dict[str, int]:
        return cast("dict[str, int]", self.call("fill_images", [list(c) for c in chunks]))

    def amstertime_download(self) -> int:
        return cast("int", self.call("amstertime_download"))

    def amstertime_embed(self, option: ModelOption) -> ChunkResult:
        return cast("ChunkResult", self.call("amstertime_embed", option))

    def match(self, tasks: Sequence[MatchTask]) -> MatchChunk:
        return cast("MatchChunk", self.call("match", list(tasks)))

    def detect_rooms(
        self,
        option: DetectorOption | None,
        photos: Sequence[RoomPhoto],
        given: Mapping[str, tuple[Box, ...]] | None = None,
    ) -> DetectChunk:
        result = self.call("detect_rooms", option, list(photos), dict(given or {}))
        return cast("DetectChunk", result)

    def locate_artworks(
        self, scenes: Sequence[RoomPhoto], tasks: Sequence[LocateTask]
    ) -> list[Located]:
        return cast("list[Located]", self.call("locate_artworks", list(scenes), list(tasks)))

    def match_site(self, tasks: Sequence[SiteMatchTask]) -> list[SiteMatches]:
        return cast("list[SiteMatches]", self.call("match_site", list(tasks)))

    def read_captions(self, model: str, photos: Sequence[CaptionPhoto]) -> CaptionChunk:
        return cast("CaptionChunk", self.call("read_captions", model, list(photos)))


class ScalewayBackend:
    """The website's search (`SearchBackend`) on the machine: `miroir web dev --scw`.

    The index bundle comes from the index bucket (`miroir web upload`)."""

    def __init__(self, calls: ScalewayCalls, index: str) -> None:
        self._calls = calls
        self._index = index

    async def _run(self, name: str, *args: object) -> object:
        import asyncio  # noqa: PLC0415

        return await asyncio.to_thread(self._calls.call, name, self._index, *args)

    async def search(self, request: SearchRequest) -> SearchResponse:
        return cast("SearchResponse", await self._run("search", request))

    async def warm(self) -> None:
        return None  # the machine is up for as long as the page runs

    async def detect(self, jpeg: bytes) -> Outlines:
        return cast("Outlines", await self._run("detect", jpeg))

    async def tool_search(self, query: ToolQuery) -> ToolHits:
        return cast("ToolHits", await self._run("tool_search", query))


# --- the whole session ----------------------------------------------------------------------


@contextmanager
def open_scaleway(
    settings: Settings,
    *,
    max_hours: float,
    purpose: str,
    offers: Sequence[Offer] = OFFERS,
    match: bool = False,
) -> Generator[ScalewayCalls]:
    """Start a GPU machine with the worker on it, yield its calls, then delete the machine.

    `match` builds the image with the keypoint-matching libraries (`start_worker`)."""
    keys = settings.scaleway_dir / "keys" / f"{time.time_ns()}"
    keys.mkdir(parents=True, mode=0o700)
    key = keys / "id_ed25519"
    subprocess.run(  # noqa: S603
        ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "miroir-scw", "-f", str(key)],  # noqa: S607
        check=True,
    )
    public = key.with_suffix(".pub").read_text().strip()
    api = InstanceApi(settings)
    try:
        with gpu_machine(
            settings, api, public_key=public, max_hours=max_hours, offers=offers, purpose=purpose
        ) as machine:
            shell = Shell(machine.ip, key, keys / "known_hosts")
            shell.wait()
            start_worker(shell, settings, machine, match=match)
            with shell.tunnel(WORKER_PORT) as port:
                calls = ScalewayCalls(f"http://127.0.0.1:{port}", machine)
                try:
                    calls.wait_ready(shell)
                    yield calls
                    logger.info("sent back to the buckets: %s", calls.persist())
                finally:
                    calls.close()
    finally:
        api.close()
        shutil.rmtree(keys, ignore_errors=True)
