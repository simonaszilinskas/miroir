"""Long jobs on Scaleway machines that run without the Mac: start, watch, always delete.

`run_unattended` starts a machine (`miroir.jobs.scaleway.gpu_machine`, any offer, with or
without a GPU), prepares it, starts one process per shard in the background and then only
watches: every `poll_s` it reads each shard's status from the images bucket
(`miroir.jobs.bulk.JobStatus`) and asks the machine how many job processes are still alive.
When every shard is done, or no process is left, the machine is deleted. A Mac that dies
leaves the machine to its own deadline (`miroir scw watch`); what the shards stored stays in
the bucket, and a rerun starts where they stopped.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from miroir.config import DATA_DIR_VARIABLE, Settings
from miroir.jobs.bulk import (
    IMAGE_LIST_KEY,
    JULY_LIST_KEY,
    JobState,
    JobStatus,
    check_key,
    status_key,
)
from miroir.jobs.objects import Bucket, bucket_name, s3_client
from miroir.jobs.remote_spec import MODEL_LIBRARIES
from miroir.jobs.scaleway import InstanceApi, Machine, Offer, gpu_machine
from miroir.jobs.scw_run import (
    IMAGE_TAG,
    MOUNTS,
    REMOTE_ROOT,
    WORKER_LIBRARIES,
    Shell,
    build_context,
    build_image,
    send_env,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

#: Libraries a machine without a GPU needs for the jobs here (no model): pinned as in the
#: job image.
CPU_LIBRARY_NAMES = frozenset(
    {"httpx", "pillow", "numpy", "pydantic", "pydantic-settings", "pyarrow", "typer", "boto3"}
)
VENV = "/opt/venv"
SETUP_TIMEOUT_S = 20 * 60
#: Every image row with the file that holds it, written by `upload_image_list`.
IMAGE_FILES = "image-files.parquet"


def cpu_requirements() -> str:
    pins = [
        p for p in (*MODEL_LIBRARIES, *WORKER_LIBRARIES) if p.split("==")[0] in CPU_LIBRARY_NAMES
    ]
    return "\n".join(pins) + "\n"


# --- the image list -----------------------------------------------------------------------


def image_files(settings: Settings, embedded: frozenset[str] = frozenset()) -> pa.Table:
    """Every image row of the full POP tables, then the July rows they lack, with its file:
    one file per distinct URL, named by the id of one of its rows (`file_id`): a row already
    `embedded` when there is one, so its vector serves the file, else the first row."""
    columns = ["image_id", "url", "collection"]
    tables = (settings.full_images_parquet, settings.images_parquet)
    ids: list[str] = []
    urls: list[str] = []
    collections: list[str] = []
    seen: set[str] = set()
    for path in tables:
        data = pq.read_table(path, columns=columns).to_pydict()  # pyright: ignore[reportUnknownMemberType]
        for image_id, url, collection in zip(
            data["image_id"], data["url"], data["collection"], strict=True
        ):
            if image_id in seen:
                continue
            seen.add(image_id)
            ids.append(image_id)
            urls.append(url)
            collections.append(collection)
    file_of: dict[str, str] = {}
    for image_id, url in zip(ids, urls, strict=True):
        if url not in file_of or (image_id in embedded and file_of[url] not in embedded):
            file_of[url] = image_id
    files = [file_of[u] for u in urls]
    return pa.table(  # pyright: ignore[reportUnknownMemberType]
        {"image_id": ids, "file_id": files, "url": urls, "collection": collections}
    )


def image_list(files: pa.Table) -> pa.Table:
    """One row per file to fetch: `image_id` (the file id), `url`, `collection` (of its
    first row)."""
    data = files.to_pydict()
    rows: dict[str, list[str]] = {"image_id": [], "url": [], "collection": []}
    seen: set[str] = set()
    for file_id, url, collection in zip(
        data["file_id"], data["url"], data["collection"], strict=True
    ):
        if file_id in seen:
            continue
        seen.add(file_id)
        rows["image_id"].append(file_id)
        rows["url"].append(url)
        rows["collection"].append(collection)
    return pa.table(rows)  # pyright: ignore[reportUnknownMemberType]


def upload_image_list(settings: Settings, embedded: frozenset[str]) -> int:
    """Write the image rows with their files (`Settings.scaleway_dir`), and the list of files
    to the images bucket. Returns how many files."""
    files = image_files(settings, embedded)
    folder = settings.scaleway_dir
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(files, folder / IMAGE_FILES)  # pyright: ignore[reportUnknownMemberType]
    table = image_list(files)
    path = folder / "images-list.parquet"
    pq.write_table(table, path)  # pyright: ignore[reportUnknownMemberType]
    s3_client(settings).upload_file(str(path), bucket_name(settings, Bucket.IMAGES), IMAGE_LIST_KEY)
    return table.num_rows


# --- preparing a machine -------------------------------------------------------------------


def cpu_setup(shell: Shell, settings: Settings) -> None:
    """Python, the few libraries and this checkout's package, in a venv on plain Ubuntu."""
    send_env(shell, settings)
    start = time.monotonic()
    shell.run("cloud-init status --wait > /dev/null; true", timeout=SETUP_TIMEOUT_S)
    shell.run(
        "DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 update -qq "
        "&& DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 "
        "install -y -qq python3-venv > /dev/null",
        timeout=SETUP_TIMEOUT_S,
    )
    shell.run(
        f"rm -rf {REMOTE_ROOT}/build && mkdir -p {REMOTE_ROOT}/build "
        f"&& tar -xzf - -C {REMOTE_ROOT}/build",
        stdin=build_context(),
    )
    shell.run(f"cat > {REMOTE_ROOT}/build/cpu.txt", stdin=cpu_requirements().encode())
    uv = f"{VENV}/bin/uv pip install -q --python {VENV}/bin/python"
    shell.run(
        f"python3 -m venv {VENV} && {VENV}/bin/pip install -q uv==0.11.29 "
        f"&& {uv} -r {REMOTE_ROOT}/build/cpu.txt && {uv} --no-deps {REMOTE_ROOT}/build",
        timeout=SETUP_TIMEOUT_S,
    )
    logger.info("machine ready in %.0f s", time.monotonic() - start)


def start_cpu_process(
    shell: Shell, name: str, args: str, env: Mapping[str, str] | None = None
) -> None:
    """Run `miroir <args>` in the background with the worker's settings (and `env`), its
    output in `<REMOTE_ROOT>/<name>.log`. The keys stay in the env file, off the command line."""
    extra = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in (env or {}).items())
    script = (
        f"#!/bin/bash\nset -a; . {REMOTE_ROOT}/env; set +a\n{extra}exec {VENV}/bin/miroir {args}\n"
    )
    shell.run(
        f"cat > {REMOTE_ROOT}/{name}.sh && chmod 700 {REMOTE_ROOT}/{name}.sh", stdin=script.encode()
    )
    shell.run(f"nohup setsid {REMOTE_ROOT}/{name}.sh > {REMOTE_ROOT}/{name}.log 2>&1 < /dev/null &")


def start_watch(shell: Shell, machine: Machine) -> None:
    """The machine deletes itself at its deadline, should the Mac be gone by then."""
    start_cpu_process(
        shell,
        "watch",
        f"scw watch --server-id {machine.server_id} --zone {machine.offer.zone} "
        f"--deadline {machine.deadline:.0f}",
    )


# --- watching ------------------------------------------------------------------------------


def read_status(s3: S3Client, bucket: str, job: str, shard: int) -> JobStatus | None:
    from botocore.exceptions import ClientError  # noqa: PLC0415

    try:
        body = s3.get_object(Bucket=bucket, Key=status_key(job, shard))["Body"].read()
    except ClientError:
        return None
    return JobStatus.from_json(body.decode())


@dataclass(frozen=True, slots=True)
class Launch:
    """What one machine runs: shards of one job."""

    job: str
    shards: tuple[int, ...]
    start: Callable[[Shell, Machine], None]
    """Prepares the machine and starts one background process per shard."""
    alive: str
    """A command printing how many job processes still run."""
    logs: str
    """A command printing the end of the processes' logs."""
    probe: str = ""
    """A command whose output goes to the Mac's log at each poll (how busy the machine is)."""


@contextmanager
def session(
    settings: Settings, *, offers: Sequence[Offer], max_hours: float, purpose: str
) -> Generator[tuple[Machine, Shell]]:
    """A machine with SSH in, deleted when the block ends, whatever happens."""
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
            yield machine, shell
    finally:
        api.close()
        shutil.rmtree(keys, ignore_errors=True)


def run_unattended(
    settings: Settings,
    launch: Launch,
    *,
    offers: Sequence[Offer],
    max_hours: float,
    purpose: str,
    poll_s: float = 60.0,
) -> list[JobStatus]:
    """Start the machine, start the job, watch it until it ends, delete the machine."""
    s3 = s3_client(settings)
    bucket = bucket_name(settings, Bucket.IMAGES)
    with session(settings, offers=offers, max_hours=max_hours, purpose=purpose) as (
        machine,
        shell,
    ):
        logger.info(
            "%s: %s in %s, shards %s", machine.name, machine.offer.commercial_type,
            machine.offer.zone, ",".join(map(str, launch.shards)),
        )  # fmt: skip
        started = time.time()
        launch.start(shell, machine)
        while True:
            time.sleep(poll_s)
            statuses = [read_status(s3, bucket, launch.job, k) for k in launch.shards]
            fresh = [s for s in statuses if s is not None and s.started >= started - 600]
            for status in fresh:
                logger.info("%s", status.line())
            if len(fresh) == len(launch.shards) and all(s.state is JobState.DONE for s in fresh):
                return fresh
            if launch.probe:
                logger.info("machine: %s", shell.run(launch.probe, check=False).strip())
            alive = int(shell.run(launch.alive, check=False).strip() or "0")
            if alive == 0:
                tail = shell.run(launch.logs, check=False)
                msg = f"{launch.job}: no process left and not every shard done:\n{tail[-4000:]}"
                raise RuntimeError(msg)
            if time.time() > machine.deadline - 300:
                msg = f"{launch.job}: the machine reaches its deadline; rerun to go on"
                raise TimeoutError(msg)


def download_launch(settings: Settings, shards: Sequence[int], of: int, workers: int) -> Launch:
    """Shards of `miroir scw download`, `workers` threads each, on a machine without a GPU."""

    def start(shell: Shell, machine: Machine) -> None:
        cpu_setup(shell, settings)
        start_watch(shell, machine)
        for shard in shards:
            start_cpu_process(
                shell,
                f"download-{shard}",
                f"scw download --shard {shard} --of {of} --workers {workers}",
            )

    return Launch(
        job="download",
        shards=tuple(shards),
        start=start,
        alive="pgrep -fc 'miroir scw [d]ownload' || true",
        logs=f"tail -n 30 {REMOTE_ROOT}/download-*.log",
    )


def upload_july_list(settings: Settings, embedded: frozenset[str]) -> int:
    """Write the files July's vectors cover to the images bucket (`JULY_LIST_KEY`)."""
    body = "".join(f"{i}\n" for i in sorted(embedded)).encode()
    bucket = bucket_name(settings, Bucket.IMAGES)
    s3_client(settings).put_object(Bucket=bucket, Key=JULY_LIST_KEY, Body=body)
    return len(embedded)


def embed_launch(
    settings: Settings,
    first: int,
    last: int,
    *,
    a7_batch: int,
    a7_streams: int,
    a3_batch: int,
    check: int,
    a7_trials: str = "",
) -> Launch:
    """Pieces `first`..`last` of `miroir scw embed`, in the job image on a GPU machine."""
    job = f"embed-{first}-{last}"

    def start(shell: Shell, machine: Machine) -> None:
        build_image(shell, settings)
        run = f"docker run -d --env-file {REMOTE_ROOT}/env {MOUNTS} --entrypoint miroir "
        shell.run(
            f"{run} --name watch {IMAGE_TAG} scw watch --server-id {machine.server_id} "
            f"--zone {machine.offer.zone} --deadline {machine.deadline:.0f}"
        )
        shell.run(
            f"{run} --name embed --gpus all --shm-size 8g {IMAGE_TAG} scw embed "
            f"--first {first} --last {last} --a7-batch {a7_batch} --a7-streams {a7_streams} "
            f"--a3-batch {a3_batch} --check {check}"
            + (f" --a7-trials {a7_trials}" if a7_trials else "")
        )

    return Launch(
        job=job,
        shards=(0,),
        start=start,
        alive="docker ps -q --filter name=^embed$ | wc -l",
        logs="docker logs --tail 40 embed 2>&1",
        probe="nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader "
        "&& cat /proc/loadavg",
    )


#: Where the index build reads its inputs and writes its scratch files, on the machine.
BUILD_ROOT = "/data"


def index_launch(settings: Settings) -> Launch:
    """`miroir scw build-pop-index` on a machine with much memory and a large disk."""

    def start(shell: Shell, machine: Machine) -> None:
        cpu_setup(shell, settings)
        start_watch(shell, machine)
        start_cpu_process(
            shell,
            "pop-index",
            f"scw build-pop-index --work {BUILD_ROOT}/work",
            env={DATA_DIR_VARIABLE: f"{BUILD_ROOT}/inputs"},
        )

    return Launch(
        job="pop-index",
        shards=(0,),
        start=start,
        alive="pgrep -fc 'miroir scw [b]uild-pop-index' || true",
        logs=f"tail -n 40 {REMOTE_ROOT}/pop-index.log",
    )


def compare_check(settings: Settings, job: str) -> dict[str, tuple[int, float, float]]:
    """The July files a run embedded again (`miroir.jobs.bulk.check_key`) against their July
    vectors: per model, how many, the lowest and the median cosine."""
    import io  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from miroir.eval.vectors import read_vectors  # noqa: PLC0415
    from miroir.index.pop_all import JULY_RUNS  # noqa: PLC0415

    body = s3_client(settings).get_object(
        Bucket=bucket_name(settings, Bucket.IMAGES), Key=check_key(job)
    )["Body"]
    out: dict[str, tuple[int, float, float]] = {}
    with np.load(io.BytesIO(body.read()), allow_pickle=False) as data:
        ids = [str(i) for i in data["ids"].tolist()]
        for key, option_id in (("a7", "A7-embeddinggemma-2-512"), ("a3", "A3-megaloc-322")):
            july = read_vectors(settings.vectors_path(JULY_RUNS[option_id]))
            rows = july.index()
            theirs = np.stack([np.asarray(july.vectors[rows[i]], dtype=np.float32) for i in ids])
            mine = np.asarray(data[key], dtype=np.float32)
            cosines = (mine * theirs).sum(axis=1)
            out[option_id] = (len(ids), float(cosines.min()), float(np.median(cosines)))
        if "timed" in data:
            for line in data["timed"].tolist():
                logger.info("timed (model, batch, streams, images/s): %s", line)
    return out
