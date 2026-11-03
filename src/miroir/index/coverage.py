"""Coverage of the `pop-all` index: every record with an image has vectors, or the reason why not.

Read on the Mac after `miroir scw pop-index`: the merged tables and image files it was built
from (`miroir.index.pop_all`), the failures the download and embedding shards logged
(`miroir.jobs.bulk.failures_key`), the files with vectors (`COVERAGE_KEY`) and the bundle's
row ids. A record counts as covered when one of its images has vectors; one that is not
takes the reason of its first image (the download's, else the embedding's). A record is in
the index when its entity has a row (a record whose image another record of the same entity
also holds is reached through that row).
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from miroir.config import Settings
from miroir.index.pop_all import COVERAGE_KEY, IMAGE_FILES, INDEX
from miroir.jobs.objects import Bucket, bucket_name, listing

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = logging.getLogger(__name__)

#: Records the ministry named, with the entity each belongs to.
TARGETS = {"APMH047981": "PM95000723", "APMH00020986": "PM21002346"}


class CollectionCoverage:
    def __init__(self) -> None:
        self.records = 0
        self.with_vectors = 0
        self.in_index = 0
        self.reasons: Counter[str] = Counter()
        """Records without vectors, by the reason of their first image."""


def failure_reasons(client: S3Client, bucket: str) -> dict[str, str]:
    """Why each file failed: the embedding's reason where the download stored it."""
    reasons: dict[str, str] = {}
    for stage in ("download", "embed"):
        for item in listing(client, bucket, f"failures/{stage}-"):
            body = client.get_object(Bucket=bucket, Key=item.key)["Body"].read().decode()
            for line in body.splitlines():
                if line.strip():
                    row = json.loads(line)
                    reasons[row["image_id"]] = f"{stage}: {row['reason']}"
    return reasons


def coverage(
    records: Mapping[str, Sequence[str]],
    images: Mapping[str, tuple[str, str]],
    file_of: Mapping[str, str],
    *,
    with_vectors: set[str],
    indexed_entities: set[str],
    reasons: Mapping[str, str],
) -> dict[str, CollectionCoverage]:
    """Per collection. `records`: ref -> (collection, entity); `images`: image id -> (ref,
    entity), a record's images in their order."""
    by_ref: dict[str, list[str]] = defaultdict(list)
    for image_id, (ref, _) in images.items():
        by_ref[ref].append(image_id)
    out: dict[str, CollectionCoverage] = defaultdict(CollectionCoverage)
    for ref, (collection, entity) in records.items():
        row = out[collection]
        row.records += 1
        files = [file_of[i] for i in by_ref.get(ref, [])]
        if not files:
            row.reasons["no image"] += 1
            continue
        if any(f in with_vectors for f in files):
            row.with_vectors += 1
        else:
            row.reasons[reasons.get(files[0], "not stored, no failure logged")] += 1
        if entity in indexed_entities:
            row.in_index += 1
    return dict(out)


def write_report(
    settings: Settings, client: S3Client, tables: Path, out: Path
) -> dict[str, object]:
    """Gather everything and write the report (`out`, JSON). Returns it."""
    index_bucket = bucket_name(settings, Bucket.INDEX)
    facts = json.loads(client.get_object(Bucket=index_bucket, Key=COVERAGE_KEY)["Body"].read())
    with_vectors = set(facts["files_with_vectors"])
    rows_body = client.get_object(Bucket=index_bucket, Key=f"{INDEX}/ids.txt")["Body"].read()
    row_ids = set(rows_body.decode().split())
    rec = pq.read_table(tables / "records.parquet", columns=["ref", "collection", "entity_id"])  # pyright: ignore[reportUnknownMemberType]
    img = pq.read_table(tables / "images.parquet", columns=["image_id", "ref", "entity_id"])  # pyright: ignore[reportUnknownMemberType]
    files = pq.read_table(settings.data_root / IMAGE_FILES, columns=["image_id", "file_id"])  # pyright: ignore[reportUnknownMemberType]
    r = rec.to_pydict()
    records = {
        ref: (c, e) for ref, c, e in zip(r["ref"], r["collection"], r["entity_id"], strict=True)
    }
    i = img.to_pydict()
    images = {
        k: (ref, e) for k, ref, e in zip(i["image_id"], i["ref"], i["entity_id"], strict=True)
    }
    f = files.to_pydict()
    file_of = dict(zip(f["image_id"], f["file_id"], strict=True))
    indexed_entities = {images[k][1] for k in row_ids if k in images}
    reasons = failure_reasons(client, bucket_name(settings, Bucket.IMAGES))
    table = coverage(
        records,
        images,
        file_of,
        with_vectors=with_vectors,
        indexed_entities=indexed_entities,
        reasons=reasons,
    )
    targets: dict[str, object] = {}
    for ref, entity in TARGETS.items():
        own = [k for k, (rr, _) in images.items() if rr == ref]
        targets[ref] = {
            "entity": entity,
            "record_in_tables": ref in records,
            "images": len(own),
            "images_with_vectors": sum(1 for k in own if file_of[k] in with_vectors),
            "image_rows_in_index": sum(1 for k in own if k in row_ids),
            "entity_rows_in_index": sum(1 for k in row_ids if images.get(k, ("", ""))[1] == entity),
            "in_index": entity in indexed_entities and ref in records,
            "reasons": sorted({reasons.get(file_of[k], "") for k in own} - {""}),
        }
    files_total = len(set(file_of.values()))
    report: dict[str, object] = {
        "index": INDEX,
        "files": files_total,
        "files_with_vectors": len(with_vectors),
        "files_without": dict(
            Counter(
                reasons.get(x, "none") for x in set(file_of.values()) - with_vectors
            ).most_common()
        ),
        "index_rows": len(row_ids),
        "by_collection": {
            c: {
                "records": v.records,
                "with_vectors": v.with_vectors,
                "in_index": v.in_index,
                "without_vectors_by_reason": dict(v.reasons.most_common()),
            }
            for c, v in sorted(table.items())
        },
        "targets": targets,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
