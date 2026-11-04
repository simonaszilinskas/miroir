"""Round R4: the chosen stack and its runners-up on the sealed halves, once (experiments.md §5).

This is the only code that reads a sealed half (`_sealed`, the one `allow_sealed=True`), and
only once every parameter is frozen. `miroir eval r4 freeze` writes `eval/r4.json`, to commit
before anything else: per website bundle, a SHA-256 of each file that sets its ranking
(routing weights and cut-offs, type guess, card levels, sure margin, catalog, image ids,
manifest), and the vector runs the sealed Commons query photos will have. Then:

1. `miroir eval r4 embed --option <A7> / --option <A3>` embeds the sealed T2, T2b and T8c
   photos on a Scaleway GPU (paid, probe first, as any query run); POP queries are gallery
   images and already have vectors;
2. `miroir eval r4 run` refuses a bundle that differs from the frozen one and a round already
   written, scores every sealed half with the site's own ranking (`measure.site_search`, as the
   build report scores the tune halves), and writes `eval/results/R4/G-site-<index>.json`
   (hit rates, cluster bootstrap) and `R4/G-detail/site-<index>.json` (not-found and card
   level shares).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter

from miroir.config import REPO_ROOT, Settings, eval_result_path
from miroir.eval.results import RunResult, SetScore, set_score, write_result
from miroir.eval.run import combined_info
from miroir.eval.scoring import QueryScore, ScoreResult
from miroir.eval.sets import Query, load_queries
from miroir.eval.vectors import VectorSet, read_vectors
from miroir.web.bundle import MANIFEST_FILE
from miroir.web.catalog import CATALOG_FILE
from miroir.web.levels import LEVELS_FILE, Level, Levels, read_levels
from miroir.web.measure import (
    COMMONS_SETS,
    IDS_FILE,
    POP_SETS,
    AskedSet,
    SiteBundle,
    SiteHit,
    query_sets,
    site_search,
)
from miroir.web.routing import GUESS_FILE, ROUTING_FILE
from miroir.web.sure import SURE_FILE

logger = logging.getLogger(__name__)

ROUND = "R4"
STEP = "G"
CONFIG = REPO_ROOT / "eval" / "r4.json"
SETS = (*POP_SETS, *COMMONS_SETS)
#: The run label of the sealed Commons query photos (as `tune_query_images` names tune ones).
SEALED_QUERIES = "-".join(COMMONS_SETS) + "-sealed-queries"
#: Files whose bytes decide how a bundle ranks and labels; `run` checks each against the freeze.
FROZEN_FILES = (
    ROUTING_FILE,
    GUESS_FILE,
    LEVELS_FILE,
    SURE_FILE,
    CATALOG_FILE,
    IDS_FILE,
    MANIFEST_FILE,
)
#: Cards kept per search: the card levels read the gap over them (`miroir.eval.levels.TOP`).
TOP = 30


@dataclass(frozen=True, slots=True)
class FrozenStack:
    """One website bundle as it was when R4 was frozen."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    index: str
    files: dict[str, str]
    """File name to SHA-256, for the `FROZEN_FILES` the bundle holds."""
    a7_query_run: str
    a3_query_run: str
    """Vector runs of the sealed Commons query photos (`miroir eval r4 embed`)."""


@dataclass(frozen=True, slots=True)
class R4Config:
    __pydantic_config__ = ConfigDict(extra="forbid")

    frozen: date
    sets: tuple[str, ...]
    stacks: tuple[FrozenStack, ...]
    """The chosen stack first, then its runners-up."""


_CONFIG = TypeAdapter(R4Config)


def fingerprint(folder: Path) -> dict[str, str]:
    """SHA-256 of each `FROZEN_FILES` file the bundle folder holds."""
    out: dict[str, str] = {}
    for name in FROZEN_FILES:
        path = folder / name
        if path.exists():
            digest = hashlib.sha256()
            with path.open("rb") as data:
                for block in iter(lambda: data.read(1 << 20), b""):
                    digest.update(block)
            out[name] = digest.hexdigest()
    return out


def query_run(option: str) -> str:
    """The run `miroir eval r4 embed --option <option>` writes (`gpu_run.run_name`)."""
    return f"{option}-{SEALED_QUERIES}"


def freeze(
    settings: Settings,
    indexes: Sequence[str],
    *,
    a7_option: str,
    a3_option: str,
    path: Path = CONFIG,
    today: date | None = None,
) -> R4Config:
    """Write the R4 config from the bundles as they are now. Refuses to overwrite one."""
    if path.exists():
        msg = f"{path} exists: R4 is frozen once (delete it only if R4 has not run)"
        raise FileExistsError(msg)
    if not indexes:
        msg = "give the bundle(s) R4 scores, the chosen one first"
        raise ValueError(msg)
    stacks: list[FrozenStack] = []
    for index in indexes:
        folder = settings.web_index_dir(index)
        files = fingerprint(folder)
        missing = [f for f in (ROUTING_FILE, GUESS_FILE, IDS_FILE, MANIFEST_FILE) if f not in files]
        if missing:
            msg = f"{folder} is not a built bundle (no {', '.join(missing)})"
            raise FileNotFoundError(msg)
        stacks.append(FrozenStack(index, files, query_run(a7_option), query_run(a3_option)))
    config = R4Config(frozen=today or datetime.now(UTC).date(), sets=SETS, stacks=tuple(stacks))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_CONFIG.dump_json(config, indent=2) + b"\n")
    logger.info("froze R4 for %s in %s: commit it before running", ", ".join(indexes), path)
    return config


def read_config(path: Path = CONFIG) -> R4Config:
    if not path.exists():
        msg = f"{path} is missing: freeze R4 first (`miroir eval r4 freeze`) and commit it"
        raise FileNotFoundError(msg)
    return _CONFIG.validate_json(path.read_bytes())


def check_stack(settings: Settings, stack: FrozenStack) -> None:
    """Raise unless the bundle's files are byte for byte the frozen ones."""
    now = fingerprint(settings.web_index_dir(stack.index))
    changed = sorted(f for f in {*now, *stack.files} if now.get(f) != stack.files.get(f))
    if changed:
        msg = f"{stack.index}: {', '.join(changed)} differ from the R4 freeze; R4 does not run"
        raise ValueError(msg)


def _sealed(settings: Settings, config: R4Config, set_id: str) -> list[Query]:
    """A sealed half, for a frozen R4 only (the one place `allow_sealed=True` is passed)."""
    if set_id not in config.sets:
        msg = f"{set_id} is not one of the frozen R4 sets {config.sets}"
        raise ValueError(msg)
    return load_queries(settings, set_id, "sealed", allow_sealed=True)


def sealed_query_images(settings: Settings, config: R4Config) -> tuple[list[str], str]:
    """The sealed Commons query photos to embed, once each, and their run label."""
    ids = [
        q.query_image_id
        for s in COMMONS_SETS
        if s in config.sets
        for q in _sealed(settings, config, s)
    ]
    return list(dict.fromkeys(ids)), SEALED_QUERIES


@dataclass(frozen=True, slots=True)
class StackScore:
    result: RunResult
    detail: dict[str, dict[str, float]]


def run(settings: Settings, config: R4Config) -> list[StackScore]:
    """Score every frozen stack on every sealed half; write its result and detail files."""
    from miroir.eval.router3 import query_chroma  # noqa: PLC0415 (fetches photos)

    for stack in config.stacks:
        check_stack(settings, stack)
        if eval_result_path(ROUND, STEP, _option(stack)).exists():
            msg = f"R4 already scored {stack.index}: the sealed halves open once"
            raise FileExistsError(msg)
    queries = {s: _sealed(settings, config, s) for s in config.sets}
    photos = list(dict.fromkeys(q.query_image_id for qs in queries.values() for q in qs))
    chroma = query_chroma(settings, photos)
    return [_score_stack(settings, stack, queries, chroma) for stack in config.stacks]


def _score_stack(
    settings: Settings,
    stack: FrozenStack,
    queries: Mapping[str, Sequence[Query]],
    chroma: Mapping[str, float],
) -> StackScore:
    folder = settings.web_index_dir(stack.index)
    bundle = SiteBundle.load(folder)
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    gallery = (_run(settings, manifest["a7_run"]), _run(settings, manifest["a3_run"]))
    commons = (_run(settings, stack.a7_query_run), _run(settings, stack.a3_query_run))
    asked = query_sets(bundle.catalog, queries, gallery=gallery, commons=commons, chroma=chroma)
    scores, detail = score_sealed(
        bundle, asked, levels=read_levels(folder / LEVELS_FILE), gallery=f"web-{stack.index}"
    )
    vector_run = f"{manifest['a7_run']}+{manifest['a3_run']}"
    notes = (
        f"Round R4, sealed halves, the site's ranking on bundle {stack.index} as frozen in "
        f"eval/r4.json (routing, guess, levels unchanged since the tune halves set them)"
    )
    result = write_result(
        eval_result_path(ROUND, STEP, _option(stack)),
        round_id=ROUND,
        step=STEP,
        option=_option(stack),
        vector_run=vector_run,
        query_vector_run=f"{stack.a7_query_run}+{stack.a3_query_run}",
        vectors=combined_info([g.info for g in gallery], notes),
        scores=scores,
        today=datetime.now(UTC).date(),
    )
    path = eval_result_path(ROUND, STEP, _option(stack)).parent / f"{STEP}-detail"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{_option(stack)}.json").write_text(json.dumps(detail, indent=1) + "\n")
    return StackScore(result, detail)


def score_sealed(
    bundle: SiteBundle, asked: Mapping[str, AskedSet], *, levels: Levels | None, gallery: str
) -> tuple[list[SetScore], dict[str, dict[str, float]]]:
    """Each set searched as the site searches: a `SetScore` per in-index set (ranks, cluster
    bootstrap), and per set what a visitor would see: the share of confident first cards
    (the bucket cut-off) and of "Très proche" ones (the card levels), and how often those
    are right. Queries whose answer has no image left in the bundle are counted, not ranked."""
    cutoff = {b.name: b.cutoff for b in bundle.routing.buckets}
    scores: list[SetScore] = []
    detail: dict[str, dict[str, float]] = {}
    for set_id, a in asked.items():
        if not a.query_ids:
            continue
        hits = site_search(bundle, a, top=TOP)
        confident = [bool(h.confidences) and h.confidences[0] >= cutoff[h.bucket] for h in hits]
        very_close = [_very_close(levels, h) for h in hits]
        row: dict[str, float] = {"queries": float(len(hits))}
        if not any(a.right):  # T8, T8c: any answer is wrong
            row["confident_but_wrong"] = _share(confident)
            row["very_close_but_wrong"] = _share(very_close)
            detail[set_id] = row
            continue
        ranked = [j for j, h in enumerate(hits) if h.right_rank > 0]
        row["answer_in_index"] = float(len(ranked))
        row["very_close"] = _share([very_close[j] for j in ranked])
        row["very_close_right"] = _share([hits[j].right_rank == 1 for j in ranked if very_close[j]])
        row["confident_right"] = _share([hits[j].right_rank == 1 for j in ranked if confident[j]])
        detail[set_id] = row
        if not ranked:
            continue
        clusters = a.clusters or ("",) * len(hits)
        per_query = tuple(
            QueryScore(
                a.query_ids[j], hits[j].right_rank, 1.0 / hits[j].right_rank, cluster=clusters[j]
            )
            for j in ranked
        )
        result = ScoreResult(
            per_query=per_query,
            skipped_no_answer=len(hits) - len(ranked),
            gallery_images=len(bundle.catalog.images),
            gallery_entities=bundle.catalog.entity_count,
            ap_max_rank=None,
        )
        scores.append(set_score(result, test_set=set_id, half="sealed", gallery=gallery))
        logger.info("%s: %s", set_id, ", ".join(f"{k} {v:.3f}" for k, v in row.items()))
    return scores, detail


def _very_close(levels: Levels | None, hit: SiteHit) -> bool:
    rule = None if levels is None else levels.for_route(hit.bucket)
    return rule is not None and rule.first_level(hit.confidences) is Level.VERY_CLOSE


def _share(flags: Sequence[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _option(stack: FrozenStack) -> str:
    return f"site-{stack.index}"


def _run(settings: Settings, name: str) -> VectorSet:
    return read_vectors(settings.vectors_path(name))
