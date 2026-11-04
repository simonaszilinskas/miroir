"""The index of all of POP (`pop-all`): every record, every image the models could read.

Three steps, the last two on a Scaleway machine with much memory (`miroir scw pop-index`):

1. On the Mac, `upload_inputs` sends what the build reads to the index bucket, under
   `inputs/`: the full POP tables with the July records they lack (`merged_tables`), the
   image rows with their files (`miroir.jobs.scw_bulk.IMAGE_FILES`), the July vectors, the
   test sets' tune halves and query vectors, the Commons tables, places and the saved type
   guess and margin rules.
2. On the machine, `assemble` writes the two vector runs (`A7_RUN`, `A3_RUN`): one row per
   image row of the merged tables, one per file and entity (`index_rows`), its vector taken
   from the July runs when they have the file, else from the pieces the GPUs wrote
   (`miroir.jobs.bulk.piece_key`); plus the Commons photos of the July runs.
3. `miroir.web.bundle.build_index("pop-all")` builds the bundle as for July, MegaLoc stored as
   a shortlist (`IndexSpec.a3_shortlist_dims`), and it goes to the index bucket.

`coverage.json`, written beside the runs, lists the files with vectors, for the coverage
report (`miroir.index.coverage`).
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

from miroir.config import Settings
from miroir.eval.vectors import VectorSet, read_vectors, write_vectors
from miroir.jobs.bulk import PIECES, JobState, JobStatus, piece_key, piece_prefix, status_key
from miroir.jobs.objects import Bucket, bucket_name, listing
from miroir.parallel import bounded_map
from miroir.records.store import write_table

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

INDEX = "pop-all"
A7_OPTION = "A7-embeddinggemma-2-512"
A3_OPTION = "A3-megaloc-322"
A7_RUN = f"{A7_OPTION}-{INDEX}"
A3_RUN = f"{A3_OPTION}-{INDEX}"
#: The July runs (with the Commons photos of step F2) whose vectors are kept.
JULY_RUNS = {A7_OPTION: f"{A7_OPTION}-full+F2", A3_OPTION: f"{A3_OPTION}-full+F2"}
QUERY_RUNS = (f"{A7_OPTION}-T2-T2b-T8c-tune-queries", f"{A3_OPTION}-T2-T2b-T8c-tune-queries")
#: The rules of the July bundle that `pop-all` keeps (`miroir web sure`, `miroir web levels`).
KEPT_RULES = ("sure", "levels")
JULY_INDEX = "full-f2"
GUESS_FILE = "guess-A7-embeddinggemma-2-512-full.npz"
INPUTS = "inputs"
COVERAGE_FILE = "coverage.json"
#: The image rows with their files, under the inputs.
IMAGE_FILES = "scaleway/image-files.parquet"


# --- 1. on the Mac ----------------------------------------------------------------------------


def merged_tables(settings: Settings, out: Path) -> tuple[int, int]:
    """Write `records.parquet` and `images.parquet` into `out`: the full POP tables, then the
    July records (with their images) whose ref they lack. Returns the records and image rows."""
    out.mkdir(parents=True, exist_ok=True)
    full_records = pq.read_table(settings.full_records_parquet)  # pyright: ignore[reportUnknownMemberType]
    july_records = pq.read_table(settings.records_parquet)  # pyright: ignore[reportUnknownMemberType]
    known = full_records.column("ref")
    extra = july_records.filter(pc.invert(pc.is_in(july_records.column("ref"), known)))  # pyright: ignore[reportUnknownMemberType]
    records = pa.concat_tables([full_records, extra.cast(full_records.schema)])
    full_images = pq.read_table(settings.full_images_parquet)  # pyright: ignore[reportUnknownMemberType]
    july_images = pq.read_table(settings.images_parquet)  # pyright: ignore[reportUnknownMemberType]
    extra_images = july_images.filter(pc.is_in(july_images.column("ref"), extra.column("ref")))  # pyright: ignore[reportUnknownMemberType]
    images = pa.concat_tables([full_images, extra_images.cast(full_images.schema)])
    write_table(records, out / "records.parquet")
    write_table(images, out / "images.parquet")
    logger.info(
        "merged: %d records (%d from July alone), %d image rows",
        records.num_rows,
        extra.num_rows,
        images.num_rows,
    )
    return records.num_rows, images.num_rows


def input_files(settings: Settings, merged: Path) -> dict[str, Path]:
    """Each input's key under `inputs/`, relative to a data folder, and its file on the Mac."""
    root = settings.data_root
    files: dict[str, Path] = {
        "records.parquet": merged / "records.parquet",
        "images.parquet": merged / "images.parquet",
        IMAGE_FILES: settings.scaleway_dir / "image-files.parquet",
        f"index/{GUESS_FILE}": settings.index_dir / GUESS_FILE,
    }
    for rule in KEPT_RULES:
        saved = settings.index_dir / f"{rule}-{JULY_INDEX}.json"
        if saved.exists():
            files[f"index/{rule}-{INDEX}.json"] = saved
    folders = [settings.vectors_path(r) for r in (*JULY_RUNS.values(), *QUERY_RUNS)]
    singles = [
        settings.commons_parquet,
        *settings.commons_gallery_parquets(),
        settings.places_parquet,
        settings.query_positions_parquet,
    ]
    for path in singles:
        if path.exists():
            files[path.relative_to(root).as_posix()] = path
    for folder in [*folders, settings.testsets_dir]:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and ".sealed." not in path.name:
                files[path.relative_to(root).as_posix()] = path
    return files


def upload_inputs(settings: Settings, client: S3Client, files: Mapping[str, Path]) -> int:
    bucket = bucket_name(settings, Bucket.INDEX)
    stored = {s.key: s.size for s in listing(client, bucket, f"{INPUTS}/")}

    def put(item: tuple[str, Path]) -> int:
        key, path = item
        if stored.get(f"{INPUTS}/{key}") == path.stat().st_size:
            return 0
        client.upload_file(str(path), bucket, f"{INPUTS}/{key}")
        return 1

    return sum(bounded_map(put, list(files.items()), workers=8))


# --- 2. on the machine -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rows:
    """The index rows: `ids[i]` is an image row, `files[i]` the file whose vector it takes."""

    ids: tuple[str, ...]
    files: tuple[str, ...]


def index_rows(
    image_ids: Sequence[str],
    entity_ids: Sequence[str],
    file_of: Mapping[str, str],
    has_vector: Iterable[str],
    july: frozenset[str],
) -> Rows:
    """One row per file and entity, among the files with a vector. Its id is the July image
    id when the group holds one (the test sets name those), else its first image row's."""
    have = set(has_vector)
    chosen: dict[tuple[str, str], str] = {}
    for image_id, entity_id in zip(image_ids, entity_ids, strict=True):
        file_id = file_of[image_id]
        if file_id not in have:
            continue
        key = (file_id, entity_id)
        if key not in chosen or (image_id in july and chosen[key] not in july):
            chosen[key] = image_id
    ids = tuple(chosen.values())
    return Rows(ids, tuple(file_id for file_id, _ in chosen))


@dataclass(frozen=True, slots=True)
class Sources:
    """Where one model's vectors are: the July run and the pieces' matrix, by file."""

    july: VectorSet
    july_row: Mapping[str, int]
    pieces: NDArray[np.float16]
    piece_row: Mapping[str, int]


class Gathered:
    """The vectors of `files`, in that order, from the July run or the pieces' matrix, read
    in blocks: what `write_vectors` reads, without another copy of every vector."""

    def __init__(self, files: Sequence[str], sources: Sources) -> None:
        self._files = files
        self._sources = sources
        self.shape = (len(files), sources.july.dim)
        self.ndim = 2
        self.dtype = np.dtype(np.float16)

    def __getitem__(self, rows: slice) -> NDArray[np.float16]:
        files = self._files[rows]
        src = self._sources
        out = np.empty((len(files), self.shape[1]), dtype=np.float16)
        old = [(k, src.july_row[f]) for k, f in enumerate(files) if f in src.july_row]
        new = [(k, src.piece_row[f]) for k, f in enumerate(files) if f not in src.july_row]
        for matrix, pairs in ((src.july.vectors, old), (src.pieces, new)):
            if pairs:
                pairs.sort(key=lambda p: p[1])
                out[[k for k, _ in pairs]] = matrix[[r for _, r in pairs]]
        return out


def gather_pieces(
    client: S3Client, bucket: str, option_id: str, path: Path, shape: tuple[int, int]
) -> dict[str, int]:
    """Copy every piece of one model into one float16 matrix on disk, of `shape` (rows: at
    least the files to embed). Returns each file's row."""
    keys = [piece_key(option_id, piece_prefix(p)) for p in range(PIECES)]

    def fetch(key: str) -> tuple[list[str], NDArray[np.float16]]:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        with np.load(io.BytesIO(body), allow_pickle=False) as data:
            ids = [str(i) for i in np.asarray(data["ids"]).tolist()]
            return ids, np.asarray(data["vectors"], dtype=np.float16)

    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    row_of: dict[str, int] = {}
    for ids, vectors in bounded_map(fetch, keys, workers=16):
        if ids:
            at = len(row_of)
            matrix[at : at + len(ids)] = vectors
            row_of.update((i, at + k) for k, i in enumerate(ids))
    matrix.flush()
    logger.info("%s: %d new vectors from %d pieces", option_id, len(row_of), PIECES)
    return row_of


def assemble(settings: Settings, client: S3Client, work: Path) -> dict[str, object]:
    """Write `A7_RUN` and `A3_RUN` (step 2 of the module doc). Returns the coverage facts."""
    images = pq.read_table(settings.images_parquet, columns=["image_id", "entity_id"]).to_pydict()  # pyright: ignore[reportUnknownMemberType]
    files = pq.read_table(  # pyright: ignore[reportUnknownMemberType]
        settings.data_root / IMAGE_FILES, columns=["image_id", "file_id"]
    ).to_pydict()
    file_of = dict(zip(files["image_id"], files["file_id"], strict=True))
    bucket = bucket_name(settings, Bucket.IMAGES)
    sources: dict[str, Sources] = {}
    for option_id, july_name in JULY_RUNS.items():
        july = read_vectors(settings.vectors_path(july_name))
        path = work / f"{option_id}.npy"
        shape = (len(set(file_of.values())), july.dim)
        rows = gather_pieces(client, bucket, option_id, path, shape)
        pieces: NDArray[np.float16] = np.load(path, mmap_mode="r")
        sources[option_id] = Sources(july, july.index(), pieces, rows)
    july_files = frozenset(i for i in sources[A7_OPTION].july.ids if not i.startswith("commons-"))
    new = [set(s.piece_row) - july_files for s in sources.values()]
    with_vectors = july_files | (new[0] & new[1])
    rows = index_rows(images["image_id"], images["entity_id"], file_of, with_vectors, july_files)
    commons = tuple(i for i in sources[A7_OPTION].july.ids if i.startswith("commons-"))
    for option_id, run in ((A7_OPTION, A7_RUN), (A3_OPTION, A3_RUN)):
        source = sources[option_id]
        info = replace(
            source.july.info,
            gallery=INDEX,
            created=datetime.now(UTC).date(),
            notes=f"July vectors ({source.july.info.gallery}) and Scaleway L40S vectors of "
            "the files July lacked, from the stored images shrunk to 512 px as in July",
        )
        gathered = Gathered((*rows.files, *commons), source)
        write_vectors(settings.vectors_path(run), (*rows.ids, *commons), gathered, info)  # pyright: ignore[reportArgumentType]
    facts: dict[str, object] = {
        "files_with_vectors": sorted(with_vectors),
        "files_from_july": len(july_files),
        "rows": len(rows.ids),
        "commons_rows": len(commons),
    }
    (work / COVERAGE_FILE).write_text(json.dumps(facts) + "\n")
    return facts


# --- the whole build on the machine ------------------------------------------------------------

JOB = "pop-index"
COVERAGE_KEY = f"coverage/{INDEX}.json"


def build_on_machine(settings: Settings, work: Path) -> JobStatus:
    """Steps 2 and 3 of the module doc, inputs read from the bucket into `Settings.data_root`
    (whose parent holds `inputs/`), the bundle and coverage facts written back."""
    import time  # noqa: PLC0415

    from miroir.eval.bm25 import Bm25, record_text  # noqa: PLC0415
    from miroir.jobs.objects import download_tree, s3_client, upload_tree  # noqa: PLC0415
    from miroir.web.bundle import build_index  # noqa: PLC0415
    from miroir.web.catalog import CATALOG_FILE, read_catalog  # noqa: PLC0415

    client = s3_client(settings)
    images = bucket_name(settings, Bucket.IMAGES)
    index = bucket_name(settings, Bucket.INDEX)
    status = JobStatus(JOB, 0, 1, total=4)

    def step(**changes: object) -> None:
        nonlocal status
        status = status.now(**changes)
        client.put_object(Bucket=images, Key=status_key(JOB, 0), Body=status.to_json().encode())
        logger.info("%s", status.line())

    try:
        step(note="reading the inputs")
        download_tree(client, index, settings.data_root.parent, prefix=f"{INPUTS}/")
        step(done=1, note="gathering the vectors")
        work.mkdir(parents=True, exist_ok=True)
        facts = assemble(settings, client, work)
        client.upload_file(str(work / COVERAGE_FILE), index, COVERAGE_KEY)
        step(done=2, note=f"building the bundle: {facts['rows']:,} rows")
        folder = build_index(settings, INDEX)
        start = time.monotonic()
        catalog = read_catalog(folder / CATALOG_FILE)
        Bm25([record_text(r) for r in catalog.records])
        words_s = time.monotonic() - start
        step(done=3, note=f"sending the bundle; catalog and word index load in {words_s:.0f} s")
        upload_tree(client, index, folder.parent, prefix=INDEX)
    except Exception as err:
        step(state=JobState.FAILED, note=f"failed: {type(err).__name__}: {err}"[:500])
        raise
    step(
        done=4,
        state=JobState.DONE,
        note=f"bundle {INDEX} in {index}; word index over {len(catalog.records):,} records "
        f"built in {words_s:.0f} s",
    )
    return status


def run_pop_index(settings: Settings, *, max_hours: float) -> list[JobStatus]:
    """Build the `pop-all` bundle on a machine with much memory; it lands in the index
    bucket. The vectors must be in the images bucket (`pop-download`, `pop-embed`)."""
    from miroir.jobs.objects import s3_client  # noqa: PLC0415
    from miroir.jobs.scaleway import MEMORY_OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import index_launch, run_unattended  # noqa: PLC0415

    merged = settings.scaleway_dir / "pop-all-tables"
    records, images = merged_tables(settings, merged)
    logger.info("tables: %d records, %d image rows", records, images)
    sent = upload_inputs(settings, s3_client(settings), input_files(settings, merged))
    logger.info("inputs: %d file(s) sent", sent)
    return run_unattended(
        settings,
        index_launch(settings),
        offers=MEMORY_OFFERS,
        max_hours=max_hours,
        purpose="pop-index",
        poll_s=120.0,
    )
