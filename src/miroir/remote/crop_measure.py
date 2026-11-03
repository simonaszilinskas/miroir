# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""A bundle with crops, searched crops off and on, on a Scaleway GPU machine
(`miroir scw crop-measure-run`; the Mac's side is `miroir.jobs.crop_measure`).

The bundle is downloaded and loaded as the demo site loads it (`load_search`). Each query's
vectors: a POP photo's stored ones (EmbeddingGemma 2 from the gallery, MegaLoc whole from the
int8 store, as `Engine.stored_cosines` reads them) and its chroma spread from its stored file;
a Commons photo's from the Mac; an outside photo's from the site's models. Then the site's own
steps, crops off and on. The outside photos are also searched end to end
(`Engine.search`, models included), for the time a visitor waits.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from miroir.config import WEB_INDEX_MOUNT, WEIGHTS_MOUNT, Settings
from miroir.images.colour import chroma_spread
from miroir.jobs.bulk import JobState, JobStatus, full_key, status_key
from miroir.jobs.crop_measure import (
    JOB,
    MeasureOut,
    QueryRanks,
    out_json,
    out_key,
    outside_key,
    read_measure_spec,
    spec_key,
    supplied_key,
)
from miroir.jobs.crop_slice import QuerySource
from miroir.jobs.objects import Bucket, bucket_name, download_tree, s3_client
from miroir.remote.search import load_search
from miroir.web import ranking
from miroir.web.bundle import right_rank
from miroir.web.engine import Engine, SearchRequest, prepare_photo, unit

logger = logging.getLogger(__name__)

#: End-to-end searches of each outside photo, crops off and on (the first warms up).
REPEATS = 4
#: Crops rescored in the cheaper variant (the bundle's rule is the other).
FEW = 5000


def run_crop_measure(settings: Settings, index: str) -> None:
    s3 = s3_client(settings)
    bucket = bucket_name(settings, Bucket.IMAGES)
    status = JobStatus(JOB, 0, 1)

    def report(**changes: object) -> None:
        nonlocal status
        status = status.now(**changes)
        s3.put_object(Bucket=bucket, Key=status_key(JOB, 0), Body=status.to_json().encode())
        logger.info("%s", status.line())

    try:
        report(note="weights and bundle")
        spec = read_measure_spec(s3.get_object(Bucket=bucket, Key=spec_key(index))["Body"].read())
        download_tree(s3, bucket_name(settings, Bucket.WEIGHTS), Path(WEIGHTS_MOUNT))
        root = Path(WEB_INDEX_MOUNT)
        start = time.monotonic()
        download_tree(s3, bucket_name(settings, Bucket.INDEX), root, prefix=f"{index}/")
        report(note=f"bundle down in {time.monotonic() - start:.0f} s; loading")
        engine = load_search(root / index)
        if engine.crops is None:
            msg = f"{index} has no crops"
            raise ValueError(msg)
        report(note="searching", total=len(spec.queries))
        full = engine.crops
        few = replace(full, rule=replace(full.rule, rescore=FEW))
        variants = {"off": None, "on": full, "few": few}
        ranks: list[QueryRanks] = []
        supplied = _supplied(s3, bucket, index)
        for n, query in enumerate(spec.queries):
            a7, a3, chroma = _vectors(
                s3, bucket, index, engine=engine, query=query, supplied=supplied
            )
            ranks.append(_search(engine, query, (a7, a3, chroma), variants))
            if n % 50 == 0:
                report(done=n + 1)
        photo_ms = _photo_searches(s3, bucket, index, engine, spec)
        out = MeasureOut(
            index=index, gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            ranks=tuple(ranks), photo_search_ms=photo_ms, memory_gb=_memory(),
        )  # fmt: skip
        s3.put_object(Bucket=bucket, Key=out_key(index), Body=out_json(out))
    except Exception as err:
        report(state=JobState.FAILED, note=f"{type(err).__name__}: {err}"[:500])
        raise
    report(state=JobState.DONE, done=len(spec.queries), note="ranks written")


def _search(engine: Engine, query, vectors, variants) -> QueryRanks:
    """One query searched with each crop variant (None: crops off), as the site ranks."""
    a7, a3, chroma = vectors
    catalog = engine.catalog
    shown = np.ones(len(catalog.images), dtype=bool)
    shown[[catalog.image_row[i] for i in query.hidden if i in catalog.image_row]] = False
    right = [catalog.entity_row[e] for e in query.expected if e in catalog.entity_row]
    full = engine.crops
    got: dict[str, tuple[int, float, float, str]] = {}
    for name, crops in variants.items():
        engine.use_crops = crops is not None
        engine.crops = crops or full
        timings: dict[str, float] = {}
        tick = time.monotonic()
        cosines = engine.vector_cosines(a7, a3, chroma, shown, timings)
        rank = right_rank(ranking.fuse(engine.photo_of(cosines).signals), right)
        ms = 1000 * (time.monotonic() - tick)
        got[name] = (rank, ms, timings.get("crops", 0.0), cosines.bucket.name)
    engine.crops = full
    return QueryRanks(
        query_id=query.query_id, set_id=query.set_id, bucket=got["on"][3],
        off=got["off"][0], on=got["on"][0], off_ms=round(got["off"][1], 1),
        on_ms=round(got["on"][1], 1), crops_ms=round(got["on"][2], 1),
        on_few=got["few"][0], few_crops_ms=round(got["few"][2], 1),
    )  # fmt: skip


def _memory() -> dict[str, float]:
    import resource  # noqa: PLC0415

    host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB on Linux
    gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    return {"gpu_max_allocated": round(gpu, 2), "host_max_rss": round(host, 2)}


def _supplied(s3, bucket: str, index: str) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    body = s3.get_object(Bucket=bucket, Key=supplied_key(index))["Body"].read()
    with np.load(io.BytesIO(body), allow_pickle=False) as data:
        return {
            str(i): (data["a7"][k], data["a3"][k], float(data["chroma"][k]))
            for k, i in enumerate(data["ids"].tolist())
        }


def _vectors(s3, bucket, index, *, engine: Engine, query, supplied):
    if query.source is QuerySource.CATALOG:
        row = engine.catalog.image_row[query.image_id]
        a7 = unit(engine.gallery.a7_row(row))
        assert engine.exact is not None
        a3 = unit(np.asarray(engine.exact.rows[row], dtype=np.float32) * engine.exact.scale[row])
        photo = s3.get_object(Bucket=bucket, Key=full_key(query.file_id))["Body"].read()
        return a7, a3, chroma_spread(prepare_photo(photo))
    if query.source is QuerySource.SUPPLIED:
        a7, a3, chroma = supplied[query.image_id]
        return np.asarray(a7, np.float32), np.asarray(a3, np.float32), chroma
    jpeg = prepare_photo(
        s3.get_object(Bucket=bucket, Key=outside_key(index, query.image_id))["Body"].read()
    )
    a7, a3 = engine.encoders.image(jpeg)
    return a7, a3, chroma_spread(jpeg)


def _photo_searches(s3, bucket, index, engine: Engine, spec) -> dict[str, tuple[float, ...]]:
    """`Engine.search` on each outside photo, crops off and on, `REPEATS` times each."""
    photos = [
        s3.get_object(Bucket=bucket, Key=outside_key(index, q.image_id))["Body"].read()
        for q in spec.queries
        if q.source is QuerySource.OUTSIDE
    ]
    out: dict[str, list[float]] = {"off": [], "on": []}
    for photo in photos:
        for crops in (False, True):
            engine.use_crops = crops
            for repeat in range(REPEATS):
                response = engine.search(SearchRequest(image=photo))
                if repeat:
                    out["on" if crops else "off"].append(response.timings_ms["total"])
    return {k: tuple(v) for k, v in out.items()}
