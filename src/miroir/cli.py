"""The `miroir` command. Every entry point is a subcommand here: `uv run miroir --help`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from miroir import logs
from miroir.config import DEMO_INDEX, ENV_PASSWORD_KEY, WEB_INDEX, get_settings

if TYPE_CHECKING:
    from miroir.eval.build_sets import BuildReport
    from miroir.eval.h4 import Loaded
    from miroir.eval.results import SetScore
    from miroir.eval.scoring import Interval
    from miroir.jobs.demo import Demo, DemoMachine
    from miroir.jobs.h4_agent import Outcome, Scenario

app = typer.Typer(no_args_is_help=True, add_completion=False)
records_app = typer.Typer(no_args_is_help=True, help="POP records: parse the exports.")
app.add_typer(records_app, name="records")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: vectors, scoring, results.")
app.add_typer(eval_app, name="eval")
testsets_app = typer.Typer(
    no_args_is_help=True, help="Test sets T1, T2, T2b, T3, T5, T8, VET, PARTS and galleries."
)
app.add_typer(testsets_app, name="testsets")
amstertime_app = typer.Typer(no_args_is_help=True, help="AmsterTime (T9, round R0).")
eval_app.add_typer(amstertime_app, name="amstertime")
embed_app = typer.Typer(no_args_is_help=True, help="Embedding runs on remote models.")
app.add_typer(embed_app, name="embed")
rerank_app = typer.Typer(no_args_is_help=True, help="Step E1: re-rank shortlists by keypoints.")
app.add_typer(rerank_app, name="rerank")
r0_app = typer.Typer(no_args_is_help=True, help="Round R0: harness check on AmsterTime (T9).")
app.add_typer(r0_app, name="r0")
index_app = typer.Typer(
    no_args_is_help=True, help="Full-sample index for the website: PCA and the guess."
)
app.add_typer(index_app, name="index")
vlm_app = typer.Typer(
    no_args_is_help=True, help="Step H1: ask a vision model about query photos, search its answer."
)
app.add_typer(vlm_app, name="vlm")
words_app = typer.Typer(
    no_args_is_help=True, help="Step B, search by words: test set T7, its vectors, its scores."
)
app.add_typer(words_app, name="words")
web_app = typer.Typer(
    no_args_is_help=True, help="The prototype website: build its index, upload it, run it locally."
)
app.add_typer(web_app, name="web")
geo_app = typer.Typer(
    no_args_is_help=True, help="Step H2: where buildings, objects and query photos are."
)
app.add_typer(geo_app, name="geo")
scw_app = typer.Typer(
    no_args_is_help=True,
    help="Scaleway: GPU machines and Object Storage. A machine never outlives its command.",
)
app.add_typer(scw_app, name="scw")
demo_app = typer.Typer(
    no_args_is_help=True,
    help="The demo website on its own Scaleway GPU machine, powered on only for demos.",
)
app.add_typer(demo_app, name="demo")

MaxHours = Annotated[
    float, typer.Option(help="Longest life of the machine; its worker deletes it then.")
]
DEFAULT_GALLERY = "screening-20k-publishable-fetched"
DEFAULT_SETS = ("T1", "T3", "T5")
#: The model whose text tower the website searches words with (A7).
TEXT_OPTION = "A7-embeddinggemma-2-512"


@app.callback()
def _root(
    *,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """MIROIR: visual search over French heritage records."""
    logs.setup(verbose=verbose)


@records_app.command("build")
def records_build() -> None:
    """Parse the exports into records.parquet and images.parquet in the data dir."""
    from miroir.records.build import build  # noqa: PLC0415 (keeps `miroir --help` fast)

    report = build(get_settings())
    mismatches = report.mismatches()
    if mismatches:
        typer.echo(f"{len(mismatches)} count(s) differ from the spec: see warnings.", err=True)


@records_app.command("add")
def records_add(
    entity: Annotated[
        list[str], typer.Option(help="Entity id to copy, e.g. PM95000723; repeat it.")
    ],
) -> None:
    """Copy entities from the whole download (full-pop/) into extra/, for the site's index."""
    from miroir.records.extra import add_entities  # noqa: PLC0415

    settings = get_settings()
    held = add_entities(settings, entity)
    typer.echo(f"{settings.extra_records_parquet}: {held} records")


@records_app.command("build-full")
def records_build_full(
    *,
    count_only: Annotated[
        bool, typer.Option(help="Only recount the tables a previous build wrote.")
    ] = False,
) -> None:
    """Map the whole POP download (exports-full) into full-pop/ tables and count its images.

    Reads about 6.6 GB of JSON lines: about 10 minutes. Start it detached (CLAUDE.md rule 9).
    """
    from miroir.records.full import build_full, count_full, summary_lines  # noqa: PLC0415

    settings = get_settings()
    build = None if count_only else build_full(settings)
    report = count_full(settings, build)
    for line in summary_lines(report):
        typer.echo(line)
    typer.echo(f"wrote {settings.full_report_json}")


@geo_app.command("places")
def geo_places() -> None:
    """One position per entity: Mérimée or Muséofile point, else the commune centre."""
    from miroir.geo.places import build_places  # noqa: PLC0415

    report = build_places(get_settings())
    for (collection, source), count in sorted(report.counts.items()):
        typer.echo(f"{collection:<14}{source:<36}{count:>8,}")
    for collection in sorted({c for c, _ in report.counts}):
        placed, total = report.coverage(collection)
        typer.echo(f"{collection}: {placed:,} of {total:,} entities placed")


@geo_app.command("positions")
def geo_positions() -> None:
    """Where each Commons query photo was taken: EXIF GPS, else the file page's camera."""
    from miroir.geo.queries import build_positions  # noqa: PLC0415

    for (set_id, source), count in sorted(build_positions(get_settings()).items()):
        typer.echo(f"{set_id:<6}{source:<8}{count:>6}")


@eval_app.command("check-vectors")
def eval_check_vectors(
    run: Annotated[str, typer.Argument(help="Run folder name under the data dir's vectors/.")],
) -> None:
    """Check one embedding run against the vector contract (technical spec §7.1)."""
    from miroir.eval.vectors import read_vectors  # noqa: PLC0415

    loaded = read_vectors(get_settings().vectors_path(run))
    info = loaded.info
    typer.echo(
        f"{run}: {len(loaded.ids)} vectors, dim {loaded.dim}, {info.option} {info.model} "
        f"@{info.resolution} {info.pooling}, gallery {info.gallery}: OK"
    )


@testsets_app.command("build")
def testsets_build() -> None:
    """Build the test sets and galleries from records.parquet (technical spec §7)."""
    from miroir.eval.build_sets import build  # noqa: PLC0415

    _echo_build(build(get_settings()))


@testsets_app.command("commons")
def build_commons_sets(
    sample_out: Annotated[
        Path | None,
        typer.Option(
            "--check-sample",
            help="Also write 100 tune T2/T2b queries to this JSON file, to check by eye.",
        ),
    ] = None,
) -> None:
    """Find Commons photos for T2, T2b and T8 (cached), then rebuild every set and gallery."""
    from miroir.commons.build import fetch_commons  # noqa: PLC0415 (imports httpx, Pillow)
    from miroir.commons.check import write_check_sample  # noqa: PLC0415
    from miroir.eval.build_sets import build  # noqa: PLC0415

    settings = get_settings()
    report = fetch_commons(settings)
    for s in report.sets:
        typer.echo(f"{s.set_id} {s.kind}: {s.photos} photos of {s.entities} entities")
    _echo_build(build(settings))
    if sample_out is not None:
        count = write_check_sample(settings, sample_out)
        typer.echo(f"{sample_out}: {count} queries to check")


@testsets_app.command("commons-gallery")
def build_commons_gallery(
    gallery: Annotated[str, typer.Option(help="Gallery whose Mérimée buildings get photos.")],
) -> None:
    """F2: add up to 5 Commons photos per building of a gallery (cached; query photos kept out).

    Writes commons-gallery-<gallery>.parquet and the gallery list <gallery>+F2.
    """
    from miroir.commons.gallery import (  # noqa: PLC0415 (imports httpx, Pillow)
        fetch_gallery_photos,
        gallery_buildings,
    )
    from miroir.eval.sets import read_gallery  # noqa: PLC0415
    from miroir.records.store import read_images  # noqa: PLC0415

    settings = get_settings()
    report = fetch_gallery_photos(settings, gallery)
    typer.echo(
        f"{gallery}: {report.photos} photos for {report.with_photos} of {report.buildings} "
        f"buildings; {report.requests} requests sent; left out: {dict(report.rejects)}"
    )
    full = gallery_buildings(read_images(settings.images_parquet), read_gallery(settings, "full"))
    requests, hours = report.estimate(len(full))
    typer.echo(
        f"full sample: {len(full)} buildings, about {requests} requests, "
        f"{hours:.1f} h at Wikimedia's pace with nothing cached"
    )


@testsets_app.command("vet")
def testsets_vet(
    images: Annotated[
        list[Path],
        typer.Argument(
            help="The three Vétheuil photos, in order: catalogue page (Mise au tombeau), the "
            "altarpiece today, catalogue page (Descente de croix).",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """VET: three photos of the Vétheuil altarpiece from outside POP (tune half only)."""
    from miroir.eval.parts import build_vet  # noqa: PLC0415

    settings = get_settings()
    for q in build_vet(settings, images):
        typer.echo(f"{q.query_id}: {settings.outside_image_path(q.query_image_id)}")


@testsets_app.command("parts")
def testsets_parts() -> None:
    """PARTS: a detail photo as query, its whole as answer, mined from the full POP tables."""
    from miroir.eval.parts import Kind, build_parts  # noqa: PLC0415

    report = build_parts(get_settings())
    typer.echo(f"{'kind':<14} {'tune':>5} {'sealed':>6} {'read':>6}")
    for kind in Kind:
        typer.echo(
            f"{kind.value:<14} {report.tune[kind.value]:>5} {report.sealed[kind.value]:>6}"
            f" {report.tried[kind.value]:>6}"
        )
    typer.echo(f"why: {dict(report.rules)}")


def _echo_build(report: BuildReport) -> None:
    typer.echo(
        f"{'set':<4} {'tune':>6} {'sealed':>6} {'total':>6} {'dropped':>7} {'spec':>7}"
        f" {'closed-ok':>9}"
    )
    for s in report.sets:
        flag = "" if s.meets_spec else "  differs from spec §7"
        spec = f"{'>=' if s.at_least else ''}{s.spec}"
        typer.echo(
            f"{s.set_id:<4} {s.tune:>6} {s.sealed:>6} {s.tune + s.sealed:>6} {s.dropped:>7}"
            f" {spec:>7} {s.closed_ok:>9}{flag}"
        )
    typer.echo(
        f"galleries: full {report.full}, screening {report.screening} "
        f"({report.needed} needed by queries, {report.screening - report.needed} distractors), "
        f"screening publishable {report.screening_publishable}"
    )
    if report.screening_v2:
        typer.echo(
            f"screening-v2 {report.screening_v2} ({report.delta} new), "
            f"screening-v2-publishable {report.screening_v2_publishable} "
            f"({report.delta_publishable} new)"
        )


@eval_app.command("run")
def eval_run(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R1, R2...")],
    step: Annotated[str, typer.Option(help="Step letter from experiments.md §2, e.g. A.")],
    option: Annotated[str, typer.Option(help="Result file label, e.g. A1-dinov2-vitl14-224.")],
    vectors: Annotated[str, typer.Option(help="Vector run of the gallery.")],
    test_set: Annotated[str, typer.Option("--set", help="Test set id, e.g. T1.")],
    half: Annotated[str, typer.Option(help="tune. The sealed half opens in R4 only.")] = "tune",
    query_vectors: Annotated[
        str | None, typer.Option(help="Vector run of the queries, if not the gallery's.")
    ] = None,
    gallery: Annotated[
        str | None, typer.Option(help="Gallery list, e.g. screening-20k (default: whole run).")
    ] = None,
    publishable_queries: Annotated[
        bool,
        typer.Option(
            "--publishable-queries",
            help="Only queries a closed model may see: publishable, and in --gallery if given.",
        ),
    ] = False,
) -> None:
    """Score a vector run on one test set half; write eval/results/<round>/<step>-<option>.json."""
    from miroir.eval.run import score_test_set  # noqa: PLC0415

    if half not in ("tune", "sealed"):
        raise typer.BadParameter("half must be tune or sealed")
    if half == "sealed" and round_id != "R4":
        raise typer.BadParameter("the sealed half opens in round R4 only")
    scored = score_test_set(
        get_settings(),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=vectors,
        set_id=test_set,
        half="tune" if half == "tune" else "sealed",
        query_vector_run=query_vectors,
        gallery=gallery,
        allow_sealed=round_id == "R4",
        publishable_queries=publishable_queries,
    )
    _echo_scores([scored])


@eval_app.command("crops")
def eval_crops() -> None:
    """Score the crop slice test from the tables `miroir scw crop-slice` brought back."""
    import json  # noqa: PLC0415

    from miroir.config import eval_result_path  # noqa: PLC0415
    from miroir.eval.crops import (  # noqa: PLC0415
        cases_of,
        load_scores,
        report,
        target_photo_cases,
    )
    from miroir.eval.parts import VET, VET_TARGET_REF  # noqa: PLC0415
    from miroir.jobs.crop_slice import read_spec, slice_dir, slice_queries, vet_refs  # noqa: PLC0415

    settings = get_settings()
    folder = slice_dir(settings)
    spec = read_spec((folder / "spec.json").read_bytes())
    scores = load_scores(folder, spec)
    cases = cases_of(spec, slice_queries(settings))
    refs = vet_refs(settings, spec)
    target = target_photo_cases(scores, spec, cases, set_id=VET, ref_of=refs, ref=VET_TARGET_REF)
    out = report(scores, [*cases, *target])
    path = eval_result_path("R3", "SL", "crop-slice")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    typer.echo(json.dumps(out, indent=1, ensure_ascii=False))
    typer.echo(f"written to {path}")


@eval_app.command("crop-short")
def eval_crop_short() -> None:
    """Score shorter crop vectors on the crop slice (`miroir scw crop-short-slice` first)."""
    import numpy as np  # noqa: PLC0415

    from miroir.config import eval_result_path  # noqa: PLC0415
    from miroir.eval.crop_short import Projection, load_pieces, run_measure  # noqa: PLC0415
    from miroir.eval.crops import cases_of, load_scores  # noqa: PLC0415
    from miroir.jobs.crop_embed import SLICE_NAME, crops_dir  # noqa: PLC0415
    from miroir.jobs.crop_slice import read_spec, slice_dir, slice_queries  # noqa: PLC0415
    from miroir.jobs.objects import Bucket, bucket_name, s3_client  # noqa: PLC0415

    settings = get_settings()
    folder = slice_dir(settings)
    spec = read_spec((folder / "spec.json").read_bytes())
    scores = load_scores(folder, spec)
    cases = cases_of(spec, slice_queries(settings))
    mine = crops_dir(settings, SLICE_NAME)
    with np.load(mine / "queries.npz", allow_pickle=False) as data:
        at = {str(q): i for i, q in enumerate(data["query_ids"].tolist())}
        queries = np.asarray(data["a3"], dtype=np.float32)[[at[q] for q in scores.query_ids]]
    with np.load(folder / "crops.npz", allow_pickle=False) as data:
        columns = [str(f) for f in data["files"].tolist()]
    projection_path = mine / "pop-all-a3_projection.npz"
    if not projection_path.exists():
        s3_client(settings).download_file(
            bucket_name(settings, Bucket.INDEX), "pop-all/a3_projection.npz", str(projection_path)
        )
    with np.load(projection_path, allow_pickle=False) as data:
        pop_pca = Projection(
            np.asarray(data["mean"], np.float32), np.asarray(data["components"], np.float32)
        )
    right = {e for c in cases for e in c.right}
    answer_files = {
        f for f, e in zip(spec.files, scores.row_entities, strict=True) if f and e in right
    }
    short_path = mine / "gallery-a3-pca1024.npy"
    out = run_measure(
        scores,
        cases,
        load_pieces(mine),
        queries,
        columns=columns,
        pop_pca=pop_pca,
        answer_files=answer_files,
        gallery_short=np.load(short_path) if short_path.exists() else None,
    )
    trial = mine / "trial.json"
    if trial.exists():
        out["trial"] = json.loads(trial.read_text())
    path = eval_result_path("R3", "SL", "crop-short")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    typer.echo(f"written to {path}")


@eval_app.command("compare")
def eval_compare(
    a: Annotated[Path, typer.Argument(help="Result file of run a.")],
    b: Annotated[Path, typer.Argument(help="Result file of run b.")],
    *,
    test_set: Annotated[str, typer.Option("--set")],
    half: Annotated[str, typer.Option()] = "tune",
    gallery: Annotated[str, typer.Option()] = "screening-20k",
    gallery_b: Annotated[
        str | None, typer.Option(help="Gallery of run b when it differs (F2 adds images).")
    ] = None,
) -> None:
    """Paired bootstrap: is a better than b on the same queries, beyond the error bar?"""
    from miroir.eval.run import compare  # noqa: PLC0415

    for d in compare(a, b, test_set=test_set, half=half, gallery=gallery, gallery_b=gallery_b):
        verdict = "a wins" if d.diff.low > 0 else "b wins" if d.diff.high < 0 else "tie"
        typer.echo(
            f"{d.metric:7} a {100 * d.a:5.1f}  b {100 * d.b:5.1f}  a-b {100 * d.diff.value:+5.1f} "
            f"[{100 * d.diff.low:+.1f}, {100 * d.diff.high:+.1f}]  {verdict}"
        )


@eval_app.command("stack")
def eval_stack(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")],
    step: Annotated[str, typer.Option(help="Step letter: G for fusion, E for expansion, A.")],
    option: Annotated[str, typer.Option(help="Result file label, e.g. G1-A7+A3.")],
    vectors: Annotated[list[str], typer.Option(help="Vector run of one model; repeat to fuse.")],
    gallery: Annotated[str, typer.Option(help="Gallery list, e.g. screening-20k.")],
    test_set: Annotated[
        list[str] | None, typer.Option("--set", help="Test set to rank, e.g. T1; repeat it.")
    ] = None,
    method: Annotated[
        str | None, typer.Option(help="Fusion with 2+ runs: G1 (weighted sum) or G2 (RRF).")
    ] = None,
    weight: Annotated[
        float | None,
        typer.Option(help="First run's weight; others share the rest. Default: pick from a grid."),
    ] = None,
    qe_alpha: Annotated[
        float | None, typer.Option(help="Expand queries (E5, alpha-QE) with this alpha.")
    ] = None,
    publishable_queries: Annotated[
        bool,
        typer.Option("--publishable-queries", help="Only queries a closed model may see."),
    ] = False,
    not_found: Annotated[
        bool, typer.Option("--not-found", help="Also score the not-found measure.")
    ] = False,
    not_found_set: Annotated[
        str, typer.Option(help="Set for --not-found: T8 (against T1, T5) or T8c (T2, T2b).")
    ] = "T8",
    query_vectors: Annotated[
        list[str] | None,
        typer.Option(help="Query run of each --vectors model, in order (Commons queries)."),
    ] = None,
) -> None:
    """Score one or more models, fused and maybe expanded, on tune halves; write one result."""
    from miroir.eval.fusion import Method  # noqa: PLC0415
    from miroir.eval.stack import Stack, score_stack  # noqa: PLC0415

    stack = Stack(
        runs=tuple(vectors),
        method=None if method is None else Method(method),
        first_weight=weight,
        qe_alpha=qe_alpha,
        query_runs=tuple(query_vectors or ()),
    )
    result = score_stack(
        get_settings(),
        stack,
        round_id=round_id,
        step=step,
        option=option,
        set_ids=test_set or (),
        gallery=gallery,
        publishable_queries=publishable_queries,
        not_found=not_found_set if not_found else None,
    )
    typer.echo(result.vectors.notes)
    _echo_scores(result.scores)
    for s in result.not_found:
        typer.echo(
            f"{s.test_set}/{s.half}: wrong but confident at {100 * s.match_rate:.0f}% matched "
            f"{_pct(s.wrong_confident)}; matched at {100 * s.wrong_target:.0f}% wrong "
            f"{_pct(s.match_rate_at_target)} ({len(s.in_index)} in index, "
            f"{len(s.not_found)} not found)"
        )


@eval_app.command("route")
def eval_route(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")],
    option: Annotated[str, typer.Option(help="Result file label, e.g. G4-router-A7+A3.")],
    vectors: Annotated[list[str], typer.Option(help="Vector run of one model; repeat (2+).")],
    gallery: Annotated[str, typer.Option(help="Gallery list, e.g. screening-20k.")],
    test_set: Annotated[list[str], typer.Option("--set", help="Test set to rank; repeat it.")],
    guess_vectors: Annotated[
        list[str] | None,
        typer.Option(help="Vector run(s) the building-or-object guess reads. Default: first."),
    ] = None,
    switch: Annotated[
        bool, typer.Option("--switch", help="Hard switch at p > 0.5 instead of a blend.")
    ] = False,
    publishable_queries: Annotated[
        bool,
        typer.Option("--publishable-queries", help="Only queries a closed model may see."),
    ] = False,
    not_found: Annotated[
        bool, typer.Option("--not-found", help="Also score T8, the not-found measure.")
    ] = False,
) -> None:
    """G4: fusion weights set per photo by a building-or-object guess; tune halves only."""
    from miroir.eval.router import score_router  # noqa: PLC0415

    report = score_router(
        get_settings(),
        runs=vectors,
        guess_runs=guess_vectors or vectors[:1],
        hard=switch,
        round_id=round_id,
        step="G",
        option=option,
        set_ids=test_set,
        gallery=gallery,
        publishable_queries=publishable_queries,
        not_found=not_found,
    )
    typer.echo(report.result.vectors.notes)
    _echo_scores(report.result.scores)
    for s in report.result.not_found:
        typer.echo(
            f"{s.test_set}/{s.half}: wrong but confident at {100 * s.match_rate:.0f}% matched "
            f"{_pct(s.wrong_confident)}; matched at {100 * s.wrong_target:.0f}% wrong "
            f"{_pct(s.match_rate_at_target)}"
        )


@eval_app.command("route3")
def eval_route3(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")],
    step: Annotated[str, typer.Option(help="Experiment step: G, F...")] = "G",
    option: Annotated[str, typer.Option(help="Result file label, e.g. G5-router3-A7+A3.")],
    vectors: Annotated[list[str], typer.Option(help="Gallery run: A7 first, then A3.")],
    query_vectors: Annotated[
        list[str] | None,
        typer.Option(help="Each model's query run (Commons or grey queries), same order."),
    ] = None,
    gallery: Annotated[str, typer.Option(help="Gallery list.")],
    test_set: Annotated[list[str], typer.Option("--set", help="Test set to rank; repeat it.")],
    weights: Annotated[
        str | None,
        typer.Option(help="Fixed A7 weights 'archive,colour,object' instead of picking them."),
    ] = None,
) -> None:
    """G5: A7+A3 with one A7 weight per bucket (archive building, colour building, object)."""
    from miroir.eval.router3 import Bucket, score_router3  # noqa: PLC0415

    fixed = None
    if weights is not None:
        values = [float(w) for w in weights.split(",")]
        fixed = dict(zip(Bucket, values, strict=True))
    report = score_router3(
        get_settings(),
        runs=vectors,
        query_runs=query_vectors or (),
        set_ids=test_set,
        gallery=gallery,
        round_id=round_id,
        step=step,
        option=option,
        fixed=fixed,
    )
    typer.echo(report.result.vectors.notes)
    sets = list(report.buckets)
    typer.echo("buckets: " + "; ".join(f"{s} {dict(b)}" for s, b in report.buckets.items()))
    typer.echo(f"{'hit@10':<10}" + "".join(f"{s:>22}" for s in sets))
    for p in report.policies:
        row = "".join(f"{_pct(report.scores[p.name][s].hit_at_10):>22}" for s in sets)
        typer.echo(f"{p.name:<10}{row}")
    for name, by_set in report.differences.items():
        row = "".join(
            f"{100 * d.diff.value:>+8.1f} [{100 * d.diff.low:+.1f},{100 * d.diff.high:+.1f}]"
            for d in by_set.values()
        )
        typer.echo(f"{name:<14}{row}")


@eval_app.command("location")
def eval_location(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")] = "R2",
    vectors: Annotated[list[str], typer.Option(help="Gallery run: A7 first, then A3.")],
    query_vectors: Annotated[list[str], typer.Option(help="Each model's query run, same order.")],
    gallery: Annotated[str, typer.Option(help="Gallery list.")],
    suffix: Annotated[str, typer.Option(help="Ends every option label, e.g. -full.")] = "",
) -> None:
    """H2: the photo's position (filter or prior) and known fields (département, museum...)."""
    from miroir.eval.location import score_location  # noqa: PLC0415

    report = score_location(
        get_settings(),
        runs=vectors,
        query_runs=query_vectors,
        gallery=gallery,
        round_id=round_id,
        suffix=suffix,
    )
    placed, total = report.placed
    typer.echo(f"gallery entities with a place: {placed:,} of {total:,}")
    typer.echo(
        "photos with a position: "
        + ", ".join(f"{s} {n} of {t}" for s, (n, t) in report.gps_queries.items())
    )
    for radius, (median, empty) in report.nearby.items():
        typer.echo(
            f"within {radius:g} km: median {median:g} gallery entities around a T2 photo; "
            f"{100 * empty:.0f}% of T8c photos have none"
        )
    bonus, scale = report.soft_choice
    typer.echo(
        f"soft prior picked: bonus {bonus:g}, scale {scale:g} km; split-half hit@10 "
        + ", ".join(f"{s} {100 * v:.1f}" for s, v in report.soft_split.items())
    )
    for policy in (*report.known, *report.gps):
        row = "; ".join(
            f"{s} {_pct(score.hit_at_10)} ({100 * policy.differences[s].diff.value:+.1f} "
            f"[{100 * policy.differences[s].diff.low:+.1f}, "
            f"{100 * policy.differences[s].diff.high:+.1f}])"
            for s, score in policy.scores.items()
        )
        typer.echo(f"{policy.option:<34}{row}")
        if policy.not_found is not None:
            nf = policy.not_found
            typer.echo(
                f"{'':<34}T8c wrong-but-confident {_pct(nf.wrong_confident)}, "
                f"matched at 5% wrong {_pct(nf.match_rate_at_target)}"
            )


@eval_app.command("merge-vectors")
def eval_merge_vectors(
    *,
    run: Annotated[list[str], typer.Option(help="Vector run of one model; repeat it.")],
    out: Annotated[str, typer.Option(help="Name of the merged run.")],
    gallery: Annotated[str, typer.Option(help="Gallery list the merged run covers.")],
) -> None:
    """Join runs of one model (a gallery and what a larger gallery adds) into one run."""
    from miroir.eval.run import merge_vector_runs  # noqa: PLC0415

    count = merge_vector_runs(get_settings(), run, out=out, gallery=gallery)
    typer.echo(f"{out}: {count} vectors")


@eval_app.command("overlay-vectors")
def eval_overlay_vectors(
    *,
    base: Annotated[str, typer.Option(help="Run whose rows are kept unless top has them.")],
    top: Annotated[str, typer.Option(help="Run whose rows win (e.g. grey copies, F1).")],
    out: Annotated[str, typer.Option(help="Name of the new run.")],
    gallery: Annotated[str, typer.Option(help="Gallery list the new run covers.")],
) -> None:
    """One run of a gallery's images, each from `top` when it has it, else from `base`."""
    from miroir.eval.sets import read_gallery  # noqa: PLC0415
    from miroir.eval.vectors import overlay_runs, read_vectors  # noqa: PLC0415

    settings = get_settings()
    done = overlay_runs(
        settings.vectors_path(out),
        read_vectors(settings.vectors_path(base)),
        read_vectors(settings.vectors_path(top)),
        ids=read_gallery(settings, gallery),
        gallery=gallery,
    )
    typer.echo(f"{out}: {len(done.ids)} vectors; {done.info.notes[:120]}")


@testsets_app.command("fetched")
def testsets_fetched(
    *,
    gallery: Annotated[str, typer.Option(help="Gallery list, e.g. screening-v2-publishable.")],
    vectors: Annotated[list[str], typer.Option(help="Vector run to compare; repeat it.")],
) -> None:
    """Write <gallery>-fetched: its images that every given run holds a vector for."""
    from miroir.eval.run import write_fetched_gallery  # noqa: PLC0415
    from miroir.eval.sets import read_gallery  # noqa: PLC0415

    settings = get_settings()
    ids = write_fetched_gallery(settings, gallery, vectors)
    total = len(read_gallery(settings, gallery))
    typer.echo(f"{gallery}-fetched: {len(ids)} of {total} images")


@eval_app.command("views")
def eval_views(
    result: Annotated[Path, typer.Argument(help="Result file holding a T2 score.")],
    *,
    test_set: Annotated[str, typer.Option("--set")] = "T2",
    gallery: Annotated[str, typer.Option()] = "screening-v2-publishable-fetched",
) -> None:
    """Split a Commons set's tune score by the view each photo shows (exterior, interior...)."""
    from miroir.eval.run import score_by_view  # noqa: PLC0415

    for v in score_by_view(get_settings(), result, test_set=test_set, gallery=gallery):
        typer.echo(
            f"{v.view.value:9} {v.queries:4} queries  hit@1 {_pct(v.hit_at_1)}  "
            f"hit@10 {_pct(v.hit_at_10)}"
        )


@eval_app.command("summary")
def eval_summary() -> None:
    """Rebuild the summary table in docs/results.md from every file in eval/results/."""
    from miroir.eval.results import rebuild_summary, results_dir, results_md  # noqa: PLC0415

    count = rebuild_summary(results_dir(), results_md())
    typer.echo(f"docs/results.md: summary rebuilt from {count} result file(s)")


@eval_app.command("viewer")
def eval_viewer(
    *,
    out: Annotated[Path, typer.Option(help="HTML file to write.")],
    run: Annotated[
        list[str] | None,
        typer.Option(help="Gallery vector run to show; repeat it. Default: every model found."),
    ] = None,
    per_set: Annotated[int, typer.Option(help="Queries shown per set.")] = 20,
) -> None:
    """Write one HTML page showing each model's top 5 for sample queries of T1, T5 and T3."""
    from miroir.eval.viewer import build_page, find_model_runs, pop_images  # noqa: PLC0415

    settings = get_settings()
    models = find_model_runs(settings, run or ())
    with pop_images(settings) as fetch:
        page = build_page(settings, models, fetch, per_set=per_set)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    size = out.stat().st_size
    labels = ", ".join(m.label for m in models)
    typer.echo(f"{out}: {size / 1e6:.1f} MB, models: {labels}")


@amstertime_app.command("score")
def amstertime_score(
    *,
    run: Annotated[str, typer.Option(help="Vector run name.")] = "R0-dinov2-giant-224-amstertime",
    option: Annotated[str, typer.Option(help="Result file label.")] = "dinov2-giant-224",
) -> None:
    """Score an AmsterTime run in the three directions of the official code (old, new, all)."""
    from miroir.eval.run import score_amstertime  # noqa: PLC0415

    result = score_amstertime(get_settings(), vector_run=run, option=option)
    _echo_scores(result.scores)


def _pct(i: Interval) -> str:
    return f"{100 * i.value:.1f} [{100 * i.low:.1f}, {100 * i.high:.1f}]"


def _echo_scores(scores: Sequence[SetScore]) -> None:
    for s in scores:
        cells = "  ".join(
            f"{name} {100 * i.value:.1f} [{100 * i.low:.1f}, {100 * i.high:.1f}]"
            for name, i in s.metrics().items()
        )
        typer.echo(f"{s.test_set}/{s.half} on {s.gallery} ({s.queries} queries): {cells}")


@embed_app.command("openrouter")
def embed_openrouter(
    option: Annotated[str, typer.Option(help="Option id from experiments.md: A9, A10a.")],
    gallery: Annotated[
        str | None, typer.Option(help="Gallery, e.g. screening-20k-publishable.")
    ] = None,
    queries: Annotated[
        list[str] | None,
        typer.Option(help="Test set (tune half) whose query images to embed; repeat it."),
    ] = None,
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
) -> None:
    """Embed a gallery or test sets' query images on OpenRouter. Paid: probe first."""
    from miroir.jobs import embed as jobs  # noqa: PLC0415 (keeps `miroir --help` fast)
    from miroir.jobs.factory import open_embedder  # noqa: PLC0415

    if (gallery is None) == (not queries):
        raise typer.BadParameter("give exactly one of --gallery and --queries")
    with open_embedder(get_settings(), option) as embedder:
        if gallery is not None and probe:
            result, size = jobs.probe(embedder, gallery)
            typer.echo(
                f"{option} probe: {result.images} images cost ${result.cost_usd:.4f} "
                f"(${result.usd_per_image * 1000:.3f} per 1,000). "
                f"Projected for {gallery} ({size:,} images): ${result.projected_usd(size):.2f}"
            )
            return
        if probe:
            raise typer.BadParameter("--probe works with --gallery only")
        if gallery is not None:
            done = jobs.embed_gallery(embedder, gallery)
        else:
            done = jobs.embed_queries(embedder, queries or [])
        typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("f2")
def embed_f2(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[str, typer.Option(help="Gallery whose F2 photos to embed.")],
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """F2: embed the Commons photos added to a gallery's buildings (sent from the Mac)."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.supplied import open_gallery_photos  # noqa: PLC0415
    from miroir.records.store import read_commons  # noqa: PLC0415

    settings = get_settings()
    ids = [p.image_id for p in read_commons(settings.commons_gallery_parquet(gallery))]
    purpose = f"embed {option} F2-{gallery}"
    with (
        open_gallery_photos(settings, gallery) as photos,
        open_scaleway(settings, max_hours=max_hours, purpose=purpose) as calls,
    ):
        runner = GpuRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        done = runner.embed_supplied(option, f"F2-{gallery}", ids, photos, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("extra")
def embed_extra(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """Embed the images of the records in extra/ (`miroir records add`), sent from the Mac."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.supplied import open_extra_images  # noqa: PLC0415
    from miroir.records.extra import read_extra_images  # noqa: PLC0415

    settings = get_settings()
    ids = [i.image_id for i in read_extra_images(settings)]
    with (
        open_extra_images(settings) as images,
        open_scaleway(settings, max_hours=max_hours, purpose=f"embed {option} X") as calls,
    ):
        runner = GpuRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        done = runner.embed_supplied(option, "X", ids, images, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("grey")
def embed_grey(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A3-megaloc-322.")],
    gallery: Annotated[str, typer.Option(help="Gallery whose colour images to grey.")],
    queries: Annotated[
        list[str] | None, typer.Option(help="Also grey these sets' tune query images.")
    ] = None,
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """F1: embed grey copies of every image with colour in a gallery and the query sets."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.supplied import grey_candidates, open_grey_copies  # noqa: PLC0415

    settings = get_settings()
    purpose = f"embed {option} grey-{gallery}"
    with open_grey_copies(settings) as grey:
        ids = grey.not_grey(grey_candidates(settings, gallery, queries or ()))
        with open_scaleway(settings, max_hours=max_hours, purpose=purpose) as calls:
            runner = GpuRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
            done = runner.embed_supplied(option, f"grey-{gallery}", ids, grey, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("images")
def embed_images(
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = "screening-20k-publishable",
    *,
    all_images: Annotated[
        bool,
        typer.Option(
            "--all-images",
            help="Also images of records that are not publishable (full-sample index only).",
        ),
    ] = False,
    max_hours: MaxHours = 2.0,
) -> None:
    """Fetch a gallery's images once into the images bucket, so model runs read them there."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    settings = get_settings()
    with open_scaleway(settings, max_hours=max_hours, purpose=f"images {gallery}") as calls:
        runner = GpuRunner(
            settings, calls, key_usage_usd=_openrouter_key_usage(), all_images=all_images
        )
        counts = runner.fill_images(gallery)
    typer.echo(f"{gallery}: {counts}")


@r0_app.command("run")
def r0_run(
    model: Annotated[
        str, typer.Option(help="dinov2-giant-224 (expect 48.9 mAP) or dinov3-vit7b16-224 (56.5).")
    ] = "dinov2-giant-224",
    max_hours: MaxHours = 1.0,
) -> None:
    """Download AmsterTime on a GPU machine and embed it there; vectors come back here."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    settings = get_settings()
    with open_scaleway(settings, max_hours=max_hours, purpose=f"r0 {model}") as calls:
        runner = GpuRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        done = runner.r0(model)
    run = settings.vectors_path(f"R0-{model}-amstertime").name
    typer.echo(f"{run}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")
    typer.echo(f"next: uv run miroir eval amstertime score --run {run} --option {model}")


@vlm_app.command("ask")
def vlm_ask(
    model: Annotated[str, typer.Option(help="qwen3-vl-30b-a3b or gemini-3.5-flash-lite.")],
    *,
    test_set: Annotated[
        list[str] | None, typer.Option("--set", help="Tune half to ask about; repeat it.")
    ] = None,
    gallery: Annotated[str, typer.Option(help="Keep queries whose image is here.")] = (
        DEFAULT_GALLERY
    ),
    probe: Annotated[
        bool, typer.Option("--probe", help="Ask about 100 query images; print the projection.")
    ] = False,
    max_usd: Annotated[float, typer.Option(help="Refuse or stop a run past this.")] = 1.0,
    batch: Annotated[
        bool,
        typer.Option(
            "--batch/--no-batch",
            help="Batch API when the model has one. --no-batch: standard endpoint, full price, "
            "for images the batch could not fetch.",
        ),
    ] = True,
) -> None:
    """Ask a vision model about each publishable query image. Paid: probe first."""
    from miroir.jobs import describe  # noqa: PLC0415
    from miroir.jobs.factory import open_describer  # noqa: PLC0415

    settings = get_settings()
    sets = tuple(test_set or DEFAULT_SETS)
    ids = describe.query_image_ids(settings, sets, gallery)
    with open_describer(settings, model, batch=batch) as describer:
        if probe:
            result = describe.probe(describer, ids, "+".join(sets) + "-tune")
            typer.echo(
                f"{model} probe: {result.images} images cost ${result.cost_usd:.4f}. "
                f"Projected for {len(ids)} query images: ${result.projected_usd(len(ids)):.3f}"
            )
            return
        done = describe.run(describer, ids, max_usd=max_usd)
    typer.echo(f"{model}: {done.answered} new answers, ${done.cost_usd:.4f}, skipped {done.skips}")


@vlm_app.command("score")
def vlm_score(
    model: Annotated[str, typer.Option(help="Vision model whose saved answers to search.")],
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")] = "R2",
    test_set: Annotated[list[str] | None, typer.Option("--set", help="Repeat it.")] = None,
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = DEFAULT_GALLERY,
    image_run: Annotated[
        str | None, typer.Option(help="Image vector run to mix with (H3), e.g. the A7 run.")
    ] = None,
    image_label: Annotated[str, typer.Option(help="Short name of that run in labels.")] = "A7",
) -> None:
    """Score H1 (BM25 on the model's answers) and, with --image-run, H3 (mixed with images)."""
    from miroir.eval import vlm  # noqa: PLC0415
    from miroir.jobs.describe import OPTIONS  # noqa: PLC0415

    if model not in OPTIONS:
        raise typer.BadParameter(f"known models: {', '.join(OPTIONS)}")
    settings = get_settings()
    prepared = vlm.prepare(settings, OPTIONS[model], tuple(test_set or DEFAULT_SETS), gallery)
    h1 = vlm.score_h1(prepared, round_id=round_id)
    typer.echo(f"H1 {model}: {h1.vectors.notes}")
    _echo_scores(h1.scores)
    for r in vlm.report(prepared):
        bands = ", ".join(f"{b.label}: {b.queries} q, {100 * b.hit_at_10:.0f}%" for b in r.bands)
        typer.echo(
            f"{r.set_id}: {r.answered}/{r.queries} answered; named {r.named} "
            f"(right {r.name_right}); type right {r.kind_right}; commune given "
            f"{r.commune_given}, right {r.commune_right} of {r.with_commune} with one "
            f"({r.commune_read} read in the image); "
            f"visible text {r.visible_text} (top 10 gained {r.text_helped}, lost "
            f"{r.text_hurt}); hit@10 by confidence: {bands}"
        )
    if image_run is not None:
        h3, weight = vlm.score_h3(settings, prepared, image_run, image_label, round_id=round_id)
        typer.echo(f"H3 {image_label}+H1, image weight {weight:.1f}: {h3.vectors.notes}")
        _echo_scores(h3.scores)


@vlm_app.command("h4")
def vlm_h4(
    *,
    model: Annotated[str, typer.Option(help="Vision model leading the search.")] = (
        "mistral-large-4"
    ),
    scenario: Annotated[
        list[str] | None,
        typer.Option("--scenario", help="H1, S1, S2, S3, S4; repeat it. Default: all."),
    ] = None,
    per_set: Annotated[
        int, typer.Option(help="Queries drawn from each of T1, T2, T3, T5 (seeded).")
    ] = 5,
    every_query: Annotated[
        bool,
        typer.Option("--all", help="Every tune query a closed model may see; writes results."),
    ] = False,
    photos: Annotated[
        bool,
        typer.Option(
            "--photos/--per-set",
            help="The 10-photo test (T1 2, T2 3 with 2 placed, T3 2, T5 3), or --per-set.",
        ),
    ] = True,
    label: Annotated[str | None, typer.Option(help="Report folder name.")] = None,
    max_usd: Annotated[float, typer.Option(help="Stop once this run has paid this much.")] = 2.0,
    workers: Annotated[int, typer.Option(help="Queries asked in parallel.")] = 4,
) -> None:
    """Step H4: a vision model leads the search (S1 plan, S2 plan + review, S3 free loop).

    Paid. Replies are saved and replayed, so a rerun costs nothing. Without --all, a smoke
    test: --per-set queries from each set, no result file.
    """
    from miroir.eval import h4  # noqa: PLC0415
    from miroir.jobs.factory import open_agent_caller  # noqa: PLC0415
    from miroir.jobs.h4_agent import Scenario  # noqa: PLC0415
    from miroir.jobs.spend import Budget, BudgetError, SpendLedger, spent_usd  # noqa: PLC0415

    settings = get_settings()
    scenarios = [Scenario(s) for s in (scenario or [s.value for s in Scenario])]
    budget = Budget.from_settings(settings)
    spent = spent_usd(SpendLedger(settings.spend_ledger), _openrouter_key_usage())
    if spent + max_usd > budget.usable_usd:
        msg = (
            f"spent ${spent:.2f}; ${max_usd:.2f} more would pass the ${budget.usable_usd:.2f} "
            "usable budget. Ask the project lead."
        )
        raise BudgetError(msg)
    loaded = h4.load_world(settings)
    if every_query:
        queries = h4.pick(loaded, None)
    elif photos:
        queries = h4.pick_photos(loaded)
    else:
        queries = h4.pick(loaded, per_set)
    name = label or ("full" if every_query else "photos-10" if photos else f"smoke-{per_set}")
    typer.echo(f"H4 {model}: {len(queries)} queries, {', '.join(s.value for s in scenarios)}")
    with open_agent_caller(settings, model, limit_usd=max_usd) as caller:
        caller.check_images([q.query.image_id for q in queries])
        outcomes = h4.run(caller, loaded, queries, scenarios, workers=workers)
        new_spend = caller.spent_usd
    rows, extra = h4.score(loaded, outcomes)
    counts = {s: len(qs) for s, qs in loaded.sets.items()}
    costs = h4.cost_report(outcomes, counts)
    folder = settings.h4_report_dir(name)
    h4.write_traces(folder / "traces.jsonl", outcomes)
    report = {
        "model": model,
        "queries": len(queries),
        "full_counts": counts,
        "new_spend_usd": new_spend,
        "costs": costs,
        "scores": [
            {
                "run": r.name,
                "set": r.set_id,
                "queries": r.score.queries,
                "hit@1": r.score.hit_at_1.value,
                "hit@10": r.score.hit_at_10.value,
                "hit@1 - G5": r.vs_g5["hit@1"].diff.value,
                "hit@10 - G5": r.vs_g5["hit@10"].diff.value,
                **(
                    {
                        "hit@1 - S3": r.vs_s3["hit@1"].diff.value,
                        "hit@10 - S3": r.vs_s3["hit@10"].diff.value,
                    }
                    if r.vs_s3 is not None
                    else {}
                ),
            }
            for r in rows
        ],
        **extra,
    }
    (folder / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if photos and not every_query:
        _echo_photos(loaded, outcomes, folder)
    else:
        for line in [*h4.table(rows), *h4.table_s4(rows)]:
            typer.echo(line)
    for s, c in costs.items():
        typer.echo(f"{s}: {json.dumps(c, ensure_ascii=False)}")
    typer.echo(f"paid this run: ${new_spend:.4f}; traces and report in {folder}")
    if every_query:
        from miroir.jobs.h4_agent import OPTIONS as AGENT_OPTIONS  # noqa: PLC0415

        option = AGENT_OPTIONS[model]
        for path in h4.write_results(
            loaded,
            rows,
            outcomes,
            model=option.model,
            slug=option.slug,
            h3_weight=extra.get("h3_a7_weight"),
        ):
            typer.echo(f"wrote {path}")


def _echo_photos(loaded: Loaded, got: Mapping[Scenario, Sequence[Outcome]], folder: Path) -> None:
    """Per photo: each run's rank of the right answer, cost, and short S3 and S4 traces."""
    from miroir.eval import h4  # noqa: PLC0415
    from miroir.jobs.h4_agent import Scenario  # noqa: PLC0415

    ranks = h4.photo_ranks(loaded, got)
    costs = h4.photo_costs(got)
    names = list(next(iter(ranks.values())))
    typer.echo("rank of the right answer (1 = first)")
    typer.echo(
        f"{'set':<4}{'query':<30}{'bucket':<18}{'gps':<5}" + "".join(f"{n:>13}" for n in names)
    )
    for qid, by_run in ranks.items():
        q = loaded.world.queries[qid]
        gps = "yes" if loaded.facts[qid].position else ""
        typer.echo(
            f"{q.set_id:<4}{qid:<30}{q.bucket:<18}{gps:<5}"
            + "".join(f"{by_run[n]:>13}" for n in names)
        )
    typer.echo("cost per photo: $ (tokens in / out)")
    for qid, by_s in costs.items():
        cells = "  ".join(
            f"{s} ${c['cost_usd']:.4f} ({c['prompt_tokens']:.0f}/{c['completion_tokens']:.0f})"
            for s, c in sorted(by_s.items())
        )
        typer.echo(f"{qid:<30}{cells}")
    lines: list[str] = []
    for scenario in (Scenario.S3, Scenario.S4):
        for o in got.get(scenario, ()):
            lines += h4.short_trace(o, loaded.world.describe)
    (folder / "traces-short.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (folder / "photos.json").write_text(
        json.dumps({"ranks": ranks, "costs": costs}, indent=2, ensure_ascii=False) + "\n"
    )
    typer.echo(f"short traces: {folder / 'traces-short.txt'}")


@words_app.command("ask")
def words_ask(
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Write for 100 images; print the projection.")
    ] = False,
    max_usd: Annotated[float, typer.Option(help="Refuse or stop a run past this.")] = 1.0,
) -> None:
    """T7: a vision model writes the words a visitor would type for each drawn image. Paid."""
    from miroir.eval import words  # noqa: PLC0415
    from miroir.jobs import describe  # noqa: PLC0415
    from miroir.jobs.factory import open_vision  # noqa: PLC0415

    settings = get_settings()
    draws = words.draw_from_settings(settings, _web_index_ids())
    ids = [d.query.query_image_id for d in draws]
    prompts = {d.query.query_image_id: d.prompt for d in draws}
    with open_vision(settings, words.WRITER, prompts) as writer:
        if probe:
            result = describe.probe(writer, ids, words.SET_ID)
            typer.echo(
                f"T7 writer probe: {result.images} images cost ${result.cost_usd:.4f}. "
                f"Projected for {len(ids)} images: ${result.projected_usd(len(ids)):.3f}"
            )
            return
        done = describe.run(writer, ids, max_usd=max_usd)
    typer.echo(
        f"T7 writer: {done.answered} new answers, ${done.cost_usd:.4f}, skipped {done.skips}"
    )


@words_app.command("sets")
def words_sets() -> None:
    """Write T7 and T7en (both halves) from the draw and the writer's saved answers."""
    from miroir.eval import words  # noqa: PLC0415

    settings = get_settings()
    counts = words.build_sets_from_answers(
        settings, words.draw_from_settings(settings, _web_index_ids())
    )
    for label, count in counts.items():
        typer.echo(f"{label}: {count} queries")


@words_app.command("embed")
def words_embed(
    *,
    max_usd: Annotated[float, typer.Option(help="Stop before a call once spent.")] = 0.5,
    max_hours: MaxHours = 1.0,
) -> None:
    """Embed T7/T7en tune words and the refine communes with EmbeddingGemma 2 on a GPU. Paid."""
    from miroir.eval.wordsearch import TEXT_LABEL, text_inputs  # noqa: PLC0415
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    settings = get_settings()
    ids, texts = text_inputs(settings)
    with open_scaleway(settings, max_hours=max_hours, purpose="words embed") as calls:
        runner = GpuRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        done = runner.embed_texts(TEXT_OPTION, TEXT_LABEL, ids, texts, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@words_app.command("score")
def words_score(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")] = "R2",
    refine: Annotated[
        bool, typer.Option("--refine/--no-refine", help="Also score words refining a photo.")
    ] = True,
) -> None:
    """Score word search on T7 tune with the site's ranking; one result file per option."""
    from miroir.eval import wordsearch  # noqa: PLC0415
    from miroir.eval.router3 import query_chroma  # noqa: PLC0415
    from miroir.web.catalog import CATALOG_FILE, read_catalog  # noqa: PLC0415
    from miroir.web.routing import TextWeights  # noqa: PLC0415

    settings = get_settings()
    run = f"{TEXT_OPTION}-{wordsearch.TEXT_LABEL}"
    t7 = wordsearch.score_t7(settings, run, round_id=round_id)
    typer.echo(
        "T7 hit@10 by A7 weight: " + ", ".join(f"{w:.1f} {100 * h:.1f}" for w, h in t7.grid.items())
    )
    typer.echo(f"best A7 weight {t7.best_weight:.1f}")
    for part in t7.subsets:
        line = ", ".join(
            f"{name} {100 * wordsearch.hit_at(wordsearch.ranks(part.signals, mixer)):.1f}"
            for name, mixer in (
                ("BM25", wordsearch.weighted(0.0)),
                ("A7", wordsearch.weighted(1.0)),
                ("0.5", wordsearch.weighted(0.5)),
                (f"{t7.best_weight:.1f}", wordsearch.weighted(t7.best_weight)),
            )
        )
        typer.echo(f"{part.label} ({len(part.signals.query_ids)} q) hit@10: {line}")
    if not refine:
        return
    catalog = read_catalog(settings.web_index_dir(WEB_INDEX) / CATALOG_FILE)
    # Colour matters for Commons photos of buildings (T2); POP photos route by type alone.
    places = wordsearch.place_queries(settings, catalog)
    chroma = query_chroma(settings, sorted({q.query_image_id for q in places if q.set_id == "T2"}))
    words = TextWeights(a7=t7.best_weight, bm25=round(1 - t7.best_weight, 1))
    found = wordsearch.score_refine(settings, run, words, chroma)
    for set_id, hits in found.hits.items():
        typer.echo(
            f"refine {set_id} ({found.queries[set_id]} q) hit@10 by words weight: "
            + ", ".join(f"{w:g} {100 * h:.1f}" for w, h in hits.items())
        )


def _web_index_ids() -> set[str]:
    """Image ids of the website's index (`WEB_INDEX`): T7 draws only images the site holds."""
    from miroir.web.bundle import IDS_FILE  # noqa: PLC0415

    path = get_settings().web_index_dir(WEB_INDEX) / IDS_FILE
    return set(path.read_text(encoding="utf-8").split())


@rerank_app.command("run")
def rerank_run(
    *,
    vectors: Annotated[str, typer.Option(help="Vector run whose search gives the shortlists.")],
    test_set: Annotated[str, typer.Option("--set", help="Test set id; tune half only.")],
    max_usd: Annotated[float, typer.Option(help="Stop once matching has cost this much.")],
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = (
        "screening-20k-publishable-fetched"
    ),
    max_hours: MaxHours = 2.0,
) -> None:
    """Match each tune query against its top 50 entities' best images, on a GPU. Paid."""
    from miroir.eval.rerank import tasks  # noqa: PLC0415
    from miroir.eval.run import matches_name, rerank_shortlists  # noqa: PLC0415
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    settings = get_settings()
    lists, _, _ = rerank_shortlists(settings, vector_run=vectors, set_id=test_set, gallery=gallery)
    name = matches_name(vectors, test_set)
    purpose = f"rerank {name}"
    with open_scaleway(settings, max_hours=max_hours, purpose=purpose, match=True) as calls:
        runner = GpuRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        done = runner.match(name, tasks(lists), max_usd=max_usd)
    typer.echo(f"{name}: {done} queries matched; next: uv run miroir eval rerank ...")


@eval_app.command("rerank")
def eval_rerank(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2...")],
    label: Annotated[str, typer.Option(help="Shortlist label, e.g. A7-embeddinggemma-2-512.")],
    vectors: Annotated[str, typer.Option(help="Vector run whose search gave the shortlists.")],
    test_set: Annotated[str, typer.Option("--set", help="Test set id; tune half only.")],
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = (
        "screening-20k-publishable-fetched"
    ),
) -> None:
    """Score shortlists re-ranked by keypoint matches; one result file per way (E1)."""
    from miroir.eval.run import score_rerank  # noqa: PLC0415

    scores = score_rerank(
        get_settings(),
        round_id=round_id,
        label=label,
        vector_run=vectors,
        set_id=test_set,
        gallery=gallery,
    )
    _echo_scores(scores)


@index_app.command("pca")
def index_pca(
    *,
    fit: Annotated[str, typer.Option(help="Vector run to fit on: gallery vectors only.")],
    dims: Annotated[int, typer.Option(help="Dimensions to keep.")],
    apply: Annotated[
        list[str] | None, typer.Option(help="Vector run to project; repeat. Writes <run>-pca<N>.")
    ] = None,
) -> None:
    """Fit PCA on a run's vectors, save it to the index dir, project runs with it."""
    from miroir.eval.vectors import read_vectors  # noqa: PLC0415
    from miroir.index.pca import fit_pca, save_pca, write_projected  # noqa: PLC0415

    settings = get_settings()
    pca = fit_pca(read_vectors(settings.vectors_path(fit)), dims, fit)
    path = settings.index_dir / f"pca{dims}-{fit}.npz"
    save_pca(path, pca)
    typer.echo(f"{path}: {100 * pca.explained.sum():.1f}% of variance in {dims} dims")
    for run in apply or ():
        out = f"{run}-pca{dims}"
        done = write_projected(
            pca, read_vectors(settings.vectors_path(run)), settings.vectors_path(out)
        )
        typer.echo(f"{out}: {len(done.ids)} vectors of dim {done.dim}")


@index_app.command("guess")
def index_guess(
    vectors: Annotated[str, typer.Option(help="EmbeddingGemma 2 (A7) run of the full sample.")],
) -> None:
    """Train the G4 building-or-object guess on a run; save it to the index dir as .npz."""
    from miroir.index.guess import train_and_save  # noqa: PLC0415

    settings = get_settings()
    saved = train_and_save(settings, vectors, settings.index_dir / f"guess-{vectors}.npz")
    report = saved.report
    typer.echo(
        f"{saved.path}: trained on {report.trained_on:,} images, held-out balanced accuracy "
        f"{100 * report.held_out_accuracy:.1f}%"
    )
    for set_id, count, share in report.query_accuracy:
        typer.echo(f"  {set_id} tune: {100 * share:.1f}% of {count} queries guessed right")


def _openrouter_key_usage() -> float | None:
    """What OpenRouter says its key has spent, for the project-wide cap; None without a key."""
    settings = get_settings()
    if settings.openrouter_api_key is None:
        return None
    import httpx  # noqa: PLC0415

    from miroir.jobs.openrouter import OpenRouterClient, OpenRouterError  # noqa: PLC0415

    client = OpenRouterClient(settings.openrouter_api_key)
    try:
        return client.key_spend().usage_usd
    except (OpenRouterError, httpx.HTTPError):
        typer.echo("could not read OpenRouter's spend; counting the ledger only", err=True)
        return None
    finally:
        client.close()


# --- the prototype website ---------------------------------------------------------------

WebIndex = Annotated[str, typer.Option("--index", help="Index bundle name (config.WEB_INDEX).")]


@web_app.command("build-index")
def web_build_index(index: WebIndex = WEB_INDEX) -> None:
    """Build the index bundle on the Mac: vectors, catalog, type guess, weights, cut-offs."""
    from miroir.web.bundle import build_index  # noqa: PLC0415

    folder = build_index(get_settings(), index)
    typer.echo(f"bundle written to {folder}; see report.json, then `miroir web upload`")


@web_app.command("upload")
def web_upload(index: WebIndex = WEB_INDEX) -> None:
    """Copy the bundle to the index bucket, where the demo machine and `web dev --scw` read it."""
    from miroir.web.ops import upload_index  # noqa: PLC0415

    size = upload_index(get_settings(), index)
    typer.echo(f"{index}: {size / 1e6:.0f} MB sent to the index bucket")


@web_app.command("places")
def web_places(
    index: WebIndex = WEB_INDEX,
    *,
    upload: Annotated[
        bool, typer.Option(help="Also copy places.npz alone to the index bucket.")
    ] = False,
) -> None:
    """Add entity positions (`miroir geo places`) to a built bundle, without rebuilding it."""
    from miroir.web.bundle import add_places  # noqa: PLC0415
    from miroir.web.nearby import PLACES_FILE  # noqa: PLC0415

    settings = get_settings()
    count = add_places(settings, settings.web_index_dir(index))
    typer.echo(f"{index}: {count:,} entities placed")
    if upload:
        from miroir.web.ops import upload_file  # noqa: PLC0415

        upload_file(settings, index, PLACES_FILE)
        typer.echo(f"{PLACES_FILE} uploaded; `miroir demo install` makes the site read it")


@web_app.command("publishable")
def web_publishable(
    index: WebIndex = WEB_INDEX,
    *,
    upload: Annotated[
        bool, typer.Option(help="Also copy publishable.txt.gz alone to the index bucket.")
    ] = False,
) -> None:
    """Mark which images of a built bundle may go to an outside model (the conversation page)."""
    from miroir.web.bundle import add_publishable  # noqa: PLC0415
    from miroir.web.publishable import PUBLISHABLE_FILE  # noqa: PLC0415

    settings = get_settings()
    count = add_publishable(settings, settings.web_index_dir(index))
    typer.echo(f"{index}: {count:,} images may go to an outside model")
    if upload:
        from miroir.web.ops import upload_file  # noqa: PLC0415

        upload_file(settings, index, PUBLISHABLE_FILE)
        typer.echo(f"{PUBLISHABLE_FILE} uploaded; `miroir demo install` makes the site read it")


@web_app.command("secret")
def web_secret(
    *,
    rotate: Annotated[bool, typer.Option(help="Replace the password.")] = False,
) -> None:
    """Make the shared password, kept in .env (`ENV_PASSWORD_KEY`) only."""
    from miroir.web.ops import set_secret  # noqa: PLC0415

    if set_secret(rotate=rotate):
        typer.echo(
            f"new password in .env ({ENV_PASSWORD_KEY}); `miroir demo install` sends it to the site"
        )
    else:
        typer.echo("password unchanged")


@web_app.command("dev")
def web_dev(
    index: WebIndex = WEB_INDEX,
    port: Annotated[int, typer.Option(help="Local port.")] = 8000,
    *,
    fake_llm: Annotated[
        bool,
        typer.Option(help="The conversation page uses a scripted stand-in, even with a key."),
    ] = False,
    scw: Annotated[
        bool,
        typer.Option(
            help="Search on a Scaleway GPU started for this session, deleted when it stops "
            "(index from `miroir web upload`)."
        ),
    ] = False,
    max_hours: Annotated[float, typer.Option(help="With --scw: the machine's longest life.")] = 1.0,
) -> None:
    """Run the website on this Mac with fake models (no model is loaded here).

    The conversation page (/chat) calls Mistral Large 4 on Mistral's API when MISTRAL_API_KEY
    is set, else through OpenRouter, within the same spending caps as the demo site."""
    from miroir.web.ops import serve_locally  # noqa: PLC0415

    settings = get_settings()
    if not scw:
        serve_locally(settings, name=index, port=port, fake_llm=fake_llm)
        return
    from miroir.jobs.scw_run import ScalewayBackend, open_scaleway  # noqa: PLC0415

    with open_scaleway(settings, max_hours=max_hours, purpose=f"web dev {index}") as calls:
        search = ScalewayBackend(calls, index)
        serve_locally(settings, name=index, port=port, fake_llm=fake_llm, search=search)


# --- Scaleway --------------------------------------------------------------------------------


@scw_app.command("buckets")
def scw_buckets() -> None:
    """Create the Object Storage buckets the GPU machines and the website read."""
    from miroir.jobs.objects import ensure_buckets, s3_client  # noqa: PLC0415

    settings = get_settings()
    for name, outcome in ensure_buckets(settings, s3_client(settings)).items():
        typer.echo(f"{name}: {outcome}")


def _leftovers_report() -> tuple[object, list[object]]:
    from miroir.jobs.scaleway import ZONES, InstanceApi, leftovers  # noqa: PLC0415

    settings = get_settings()
    api = InstanceApi(settings)
    zones = tuple(dict.fromkeys((*ZONES, settings.scw_zone)))
    found = leftovers(api, zones)
    typer.echo(f"zones: {', '.join(zones)}")
    for item in found:
        typer.echo(f"  {item.zone}  {item.kind:<15} {item.id}  {item.label}")
    if not found:
        typer.echo("  no machine, loose volume or loose IP")
    return api, list(found)


@scw_app.command("ps")
def scw_ps() -> None:
    """List miroir machines, and volumes and IPs attached to no machine, in every zone used."""
    from miroir.jobs.scaleway import MachineLedger  # noqa: PLC0415

    _leftovers_report()
    spent = MachineLedger(get_settings().scaleway_dir / "spend.jsonl").total_eur()
    typer.echo(f"spent on machines so far: €{spent:.2f}")


@scw_app.command("cleanup")
def scw_cleanup(
    *, yes: Annotated[bool, typer.Option("--yes", help="Delete without asking.")] = False
) -> None:
    """Delete every miroir machine (with its volumes and IP) and every loose volume and IP."""
    from miroir.jobs.scaleway import InstanceApi, Leftover, delete_leftovers  # noqa: PLC0415

    api, found = _leftovers_report()
    if not found:
        return
    if not yes and not typer.confirm(f"delete these {len(found)}?"):
        raise typer.Abort
    assert isinstance(api, InstanceApi)
    delete_leftovers(api, [f for f in found if isinstance(f, Leftover)])
    typer.echo("after cleanup:")
    _leftovers_report()


@scw_app.command("up")
def scw_up(max_hours: MaxHours = 1.0) -> None:
    """Start a GPU machine with the worker on it and keep it until Ctrl-C, then delete it."""
    import time  # noqa: PLC0415

    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    with open_scaleway(get_settings(), max_hours=max_hours, purpose="up") as calls:
        gpu = calls.gpu
        typer.echo(
            f"{gpu.name}: {gpu.offer.commercial_type} in {gpu.offer.zone} at {gpu.ip}, "
            f"€{gpu.offer.eur_per_hour:.2f}/h. Ctrl-C deletes it."
        )
        while time.time() < gpu.deadline:
            time.sleep(10)


@scw_app.command("smoke")
def scw_smoke(
    images: Annotated[int, typer.Option(help="Images to embed with each model.")] = 32,
    max_hours: MaxHours = 1.0,
) -> None:
    """Embed a few images with A7 and A3 on a GPU machine and match them to the stored vectors."""
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.scw_smoke import (  # noqa: PLC0415
        MIN_COSINE,
        run_smoke,
        save_smoke,
        smoke_images,
    )

    settings = get_settings()
    picked = smoke_images(settings, images)
    with open_scaleway(settings, max_hours=max_hours, purpose="smoke") as calls:
        results = run_smoke(calls, settings, picked)
        where = f"{calls.gpu.offer.commercial_type} {calls.gpu.offer.zone}"
    path = save_smoke(settings, where, results)
    for r in results:
        typer.echo(
            f"{r.option}: {r.embedded}/{r.asked} images, cosine min {r.min_cosine:.4f} "
            f"median {r.median_cosine:.4f}, {r.images_per_s:.1f} images/s, "
            f"model load {r.load_s:.0f} s: {'ok' if r.ok else f'FAILED (needs > {MIN_COSINE})'}"
        )
    typer.echo(f"written to {path}")
    if not all(r.ok for r in results):
        raise typer.Exit(1)


@scw_app.command("serve", hidden=True)
def scw_serve(
    port: Annotated[int, typer.Option()],
    server_id: Annotated[str, typer.Option()],
    zone: Annotated[str, typer.Option()],
    deadline: Annotated[
        float, typer.Option(help="Unix time past which the machine deletes itself.")
    ],
) -> None:
    """The worker, inside the job image on a GPU machine. Never run on the Mac."""
    from miroir.remote.worker import serve  # noqa: PLC0415 (machine only)

    serve(get_settings(), port=port, server_id=server_id, zone=zone, deadline=deadline)


@scw_app.command("watch", hidden=True)
def scw_watch(
    server_id: Annotated[str, typer.Option()],
    zone: Annotated[str, typer.Option()],
    deadline: Annotated[float, typer.Option(help="Unix time past which the machine is deleted.")],
) -> None:
    """Delete this machine at its deadline. Runs on a Scaleway machine, never on the Mac."""
    from miroir.remote.worker import watch_deadline  # noqa: PLC0415 (machine only)

    watch_deadline(get_settings(), server_id, zone, deadline)


#: The July vectors every POP image reuses when it has one (`miroir scw pop-download`).
JULY_A7_RUN = "A7-embeddinggemma-2-512-full+F2"
Shards = Annotated[str, typer.Option(help="Shards this machine runs, e.g. 0,1.")]
ShardCount = Annotated[int, typer.Option("--of", help="Shards in all.")]


@scw_app.command("download", hidden=True)
def scw_download(
    shard: Annotated[int, typer.Option()],
    of: ShardCount,
    workers: Annotated[int, typer.Option(help="Parallel downloads.")] = 16,
) -> None:
    """Store one shard of the POP images in the images bucket. Runs on a Scaleway machine."""
    from miroir.jobs.pop_download import run_download  # noqa: PLC0415

    run_download(get_settings(), shard=shard, of=of, workers=workers)


@scw_app.command("pop-download")
def scw_pop_download(
    shards: Shards,
    of: ShardCount,
    workers: Annotated[int, typer.Option(help="Parallel downloads per shard.")] = 16,
    max_hours: MaxHours = 8.0,
    *,
    upload_list: Annotated[
        bool, typer.Option(help="Write the image list to the bucket first.")
    ] = False,
) -> None:
    """Download POP images to the images bucket on a machine without a GPU (resumable)."""
    from miroir.jobs.scaleway import CPU_OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import (  # noqa: PLC0415
        download_launch,
        run_unattended,
        upload_image_list,
    )

    settings = get_settings()
    if upload_list:
        from miroir.eval.vectors import read_vectors  # noqa: PLC0415

        embedded = frozenset(read_vectors(settings.vectors_path(JULY_A7_RUN)).ids)
        typer.echo(f"image list: {upload_image_list(settings, embedded):,} files")
    picked = [int(k) for k in shards.split(",")]
    launch = download_launch(settings, picked, of, workers)
    done = run_unattended(
        settings, launch, offers=CPU_OFFERS, max_hours=max_hours, purpose="download"
    )
    for status in done:
        typer.echo(status.line())


@scw_app.command("embed", hidden=True)
def scw_embed(
    first: Annotated[int, typer.Option()],
    last: Annotated[int, typer.Option()],
    *,
    a7_batch: Annotated[int, typer.Option()] = 32,
    a7_streams: Annotated[int, typer.Option()] = 1,
    a3_batch: Annotated[int, typer.Option()] = 64,
    check: Annotated[int, typer.Option()] = 0,
    a7_trials: Annotated[str, typer.Option()] = "",
) -> None:
    """Embed pieces of the stored POP images with A7 and A3. Runs in the job image on a GPU."""
    from miroir.remote.bulk_embed import run_embed  # noqa: PLC0415 (machine only)

    run_embed(
        get_settings(),
        first=first,
        last=last,
        a7_batch=a7_batch,
        a7_streams=a7_streams,
        a3_batch=a3_batch,
        check=check,
        a7_trials=_trials(a7_trials),
    )


def _trials(text: str) -> list[tuple[int, int]]:
    """ "32:1,64:2" -> [(32, 1), (64, 2)]: A7 batch sizes and streams to time."""
    pairs = [t.split(":") for t in text.split(",") if t]
    return [(int(b), int(s)) for b, s in pairs]


@scw_app.command("pop-embed")
def scw_pop_embed(
    first: Annotated[int, typer.Option(help="First piece (0 to 4095).")],
    last: Annotated[int, typer.Option(help="Last piece, included.")],
    *,
    a7_batch: Annotated[int, typer.Option(help="A7 images per batch.")] = 32,
    a7_streams: Annotated[int, typer.Option(help="A7 batches run at once.")] = 1,
    a3_batch: Annotated[int, typer.Option(help="A3 images per batch.")] = 64,
    check: Annotated[int, typer.Option(help="July files embedded again to compare.")] = 0,
    a7_trials: Annotated[
        str, typer.Option(help='A7 "batch:streams" to time on the --check files, e.g. 32:1,64:2.')
    ] = "",
    max_hours: MaxHours = 12.0,
    upload_list: Annotated[
        bool, typer.Option(help="Write the list of files July embedded to the bucket first.")
    ] = False,
    l40s: Annotated[bool, typer.Option("--l40s", help="An L40S instead of an L4.")] = False,
) -> None:
    """Embed stored POP images with A7 and A3 on a GPU machine, unattended (resumable)."""
    from miroir.jobs.scaleway import OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import (  # noqa: PLC0415
        compare_check,
        embed_launch,
        run_unattended,
        upload_july_list,
    )

    settings = get_settings()
    if upload_list:
        from miroir.eval.vectors import read_vectors  # noqa: PLC0415

        ids = read_vectors(settings.vectors_path(JULY_A7_RUN)).ids
        embedded = frozenset(i for i in ids if not i.startswith("commons-"))
        typer.echo(f"July files: {upload_july_list(settings, embedded):,}")
    launch = embed_launch(
        settings,
        first,
        last,
        a7_batch=a7_batch,
        a7_streams=a7_streams,
        a3_batch=a3_batch,
        check=check,
        a7_trials=a7_trials,
    )
    offers = [o for o in OFFERS if o.zone == "fr-par-2" and ("L40S" in o.commercial_type) == l40s]
    done = run_unattended(settings, launch, offers=offers, max_hours=max_hours, purpose="embed")
    for status in done:
        typer.echo(status.line())
    if check:
        for option_id, (count, low, median) in compare_check(settings, launch.job).items():
            typer.echo(
                f"{option_id}: {count} July files again, cosine min {low:.4f}, median {median:.4f}"
            )


@scw_app.command("crop-slice-run", hidden=True)
def scw_crop_slice_run() -> None:
    """The crop slice test, in the job image on a GPU machine. Never run on the Mac."""
    from miroir.remote.crop_slice import run_crop_slice  # noqa: PLC0415 (machine only)

    run_crop_slice(get_settings())


@scw_app.command("crop-slice")
def scw_crop_slice(
    max_hours: MaxHours = 3.4,
    *,
    l4: Annotated[bool, typer.Option("--l4", help="An L4 when no L40S is free.")] = False,
    a7_others: Annotated[
        int, typer.Option(help="Files of the sample of others that get EmbeddingGemma 2 crops.")
    ] = 3000,
    render: Annotated[
        bool, typer.Option("--render", help="A RENDER-S (P100), MegaLoc crops only.")
    ] = False,
    gallery: Annotated[int, typer.Option(help="Gallery images.")] = 50_000,
    t2_queries: Annotated[int, typer.Option(help="T2 queries kept (0: all).")] = 0,
) -> None:
    """Crop slice test (docs/search-levels.md): crops of a 50,000-image gallery on an L40S,
    scored against VET, PARTS, T2 and T3; the score tables land in the Scaleway folder."""
    from miroir.jobs.crop_slice import (  # noqa: PLC0415
        RENDER_OFFER,
        crop_launch,
        fetch_outputs,
        prepare,
    )
    from miroir.jobs.objects import s3_client  # noqa: PLC0415
    from miroir.jobs.scaleway import OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import run_unattended  # noqa: PLC0415

    settings = get_settings()
    s3 = s3_client(settings)
    if render:
        a7_others = 0
    spec = prepare(settings, s3, a7_others=a7_others, size=gallery, t2_queries=t2_queries)
    typer.echo(spec.note)
    offers = (
        [RENDER_OFFER] if render else [o for o in OFFERS if ("L40S" in o.commercial_type) != l4]
    )
    launch = crop_launch(settings, torch_tag="cu126" if render else "")
    try:
        run_unattended(settings, launch, offers=offers, max_hours=max_hours, purpose="crops")
    finally:
        for path in fetch_outputs(settings, s3):
            typer.echo(f"got {path}")


@scw_app.command("crop-embed-run", hidden=True)
def scw_crop_embed_run(
    name: Annotated[str, typer.Option(help="The crop job (its spec is in the images bucket).")],
    first: Annotated[int, typer.Option(help="First piece.")],
    last: Annotated[int, typer.Option(help="Last piece, included.")],
    *,
    reverse: Annotated[bool, typer.Option("--reverse", help="Last piece first.")] = False,
) -> None:
    """Crop vectors for pieces of a crop job, in the job image on a GPU machine. Never run on
    the Mac."""
    from miroir.remote.crop_embed import run_crop_embed  # noqa: PLC0415 (machine only)

    run_crop_embed(get_settings(), name=name, first=first, last=last, reverse=reverse)


def _run_crop_job(
    name: str, first: int, last: int, max_hours: float, *, any_gpu: bool, reverse: bool = False
) -> None:
    """One machine for pieces `first`..`last`, the best offer in stock; deleted at the end."""
    from miroir.jobs.crop_embed import crop_launch, crop_offers, in_stock  # noqa: PLC0415
    from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415
    from miroir.jobs.scw_bulk import run_unattended  # noqa: PLC0415

    settings = get_settings()
    api = InstanceApi(settings)
    try:
        offers = in_stock(api, crop_offers())
    finally:
        api.close()
    if not any_gpu:
        offers = [o for o in offers if o.commercial_type != "RENDER-S"]
    if not offers:
        typer.echo("no GPU in stock: nothing started")
        raise typer.Exit(1)
    launch = crop_launch(settings, name, first, last, reverse=reverse)
    done = run_unattended(settings, launch, offers=offers, max_hours=max_hours, purpose="crops")
    for status in done:
        typer.echo(status.line())


@scw_app.command("crop-short-slice")
def scw_crop_short_slice(
    max_hours: MaxHours = 2.0,
    *,
    any_gpu: Annotated[
        bool, typer.Option("--any-gpu", help="Also a RENDER-S (P100) when no L4 or L40S is free.")
    ] = False,
) -> None:
    """Keep the 3x3 crop vectors of the crop slice (float32 and bfloat16, whole) and the slice
    queries' MegaLoc vectors, for `miroir eval crop-short`. Lands in the Scaleway folder."""
    from miroir.jobs.crop_embed import crops_dir, fetch_job, put_spec, slice_spec  # noqa: PLC0415
    from miroir.jobs.crop_slice import slice_dir  # noqa: PLC0415
    from miroir.jobs.objects import s3_client  # noqa: PLC0415

    settings = get_settings()
    folder = slice_dir(settings)
    ids = (folder / "ids.txt").read_text(encoding="utf-8").split()
    spec = slice_spec(folder / "spec.json", ids)
    s3 = s3_client(settings)
    put_spec(settings, s3, spec)
    typer.echo(f"{spec.note}: {len(spec.pieces())} pieces")
    try:
        _run_crop_job(spec.name, 0, len(spec.pieces()) - 1, max_hours, any_gpu=any_gpu)
    finally:
        got = fetch_job(settings, s3, spec.name)
        typer.echo(f"{len(got)} files in {crops_dir(settings, spec.name)}")


@scw_app.command("crop-embed")
def scw_crop_embed(
    first: Annotated[int, typer.Option(help="First piece (0 to 4095).")],
    last: Annotated[int, typer.Option(help="Last piece, included.")],
    max_hours: MaxHours = 12.0,
    *,
    any_gpu: Annotated[
        bool, typer.Option("--any-gpu", help="Also a RENDER-S (P100) when no L4 or L40S is free.")
    ] = False,
    reverse: Annotated[bool, typer.Option("--reverse", help="Last piece first.")] = False,
) -> None:
    """3x3 crop vectors of every stored POP file, pieces `first` to `last`, on one GPU machine
    (resumable: pieces already written are skipped; `--reverse` starts from the last, to share
    a range with a machine already on it). The spec comes from `crop-embed-spec`."""
    from miroir.jobs.crop_embed import FULL_NAME  # noqa: PLC0415

    _run_crop_job(FULL_NAME, first, last, max_hours, any_gpu=any_gpu, reverse=reverse)


@scw_app.command("crop-embed-spec")
def scw_crop_embed_spec(
    dims: Annotated[int, typer.Option(help="PCA axes kept per crop (at most 1,024).")] = 256,
    precision: Annotated[
        str, typer.Option(help="MegaLoc on the GPU: fp32, tf32, bf16, fp16.")
    ] = "fp16",
) -> None:
    """Write the spec of the crop run over all of POP: 3x3 crops, projected on the first
    `dims` axes of the `pop-all` MegaLoc PCA (copied beside the spec)."""
    from miroir.jobs.crop_embed import FULL_NAME, PREFIX, Precision, full_spec, put_spec  # noqa: PLC0415
    from miroir.jobs.objects import Bucket, bucket_name, s3_client  # noqa: PLC0415

    settings = get_settings()
    s3 = s3_client(settings)
    key = f"{PREFIX}/{FULL_NAME}/projection.npz"
    s3.copy(
        {"Bucket": bucket_name(settings, Bucket.INDEX), "Key": "pop-all/a3_projection.npz"},
        bucket_name(settings, Bucket.IMAGES),
        key,
    )
    spec = full_spec(Precision(precision), key, dims)
    put_spec(settings, s3, spec)
    typer.echo(f"{spec.note}: {len(spec.pieces())} pieces")


@scw_app.command("crop-bundle")
def scw_crop_bundle(
    dims: Annotated[int, typer.Option(help="PCA axes per crop to pick crops with.")] = 128,
    rescore: Annotated[
        int, typer.Option(help="Crops picked and scored whole per search.")
    ] = 20_000,
    shortlist: Annotated[
        int, typer.Option(help="Image rows by best crop on the shortlist.")
    ] = 2000,
    weight: Annotated[float, typer.Option(help="An image scores max(whole, weight * crop).")] = 1.0,
) -> None:
    """Build `pop-all-crops` in the index bucket: `pop-all` copied there, plus the crop files
    made from the crop run's pieces (downloaded to the Scaleway folder first). No model."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    from miroir.index.crop_bundle import (  # noqa: PLC0415
        NAME,
        SOURCE,
        copy_bundle,
        copy_whole,
        crop_file_paths,
        piece_paths,
        write_crop_files,
    )
    from miroir.jobs.crop_embed import FULL_NAME, crops_dir, fetch_job  # noqa: PLC0415
    from miroir.jobs.crop_slice import IMAGE_FILES, slice_dir  # noqa: PLC0415
    from miroir.jobs.objects import Bucket, bucket_name, s3_client, upload_files  # noqa: PLC0415
    from miroir.web.crops import CropRule  # noqa: PLC0415

    settings = get_settings()
    s3 = s3_client(settings)
    index = bucket_name(settings, Bucket.INDEX)
    fetch_job(settings, s3, FULL_NAME)
    folder = crops_dir(settings, FULL_NAME)
    ids = (slice_dir(settings) / "ids.txt").read_text(encoding="utf-8").split()
    path = settings.scaleway_dir / IMAGE_FILES
    table = pq.read_table(path, columns=["image_id", "file_id"])  # pyright: ignore[reportUnknownMemberType]
    columns = table.to_pydict()
    file_of = dict(zip(columns["image_id"], columns["file_id"], strict=True))
    out = settings.scaleway_dir / NAME / NAME
    facts = write_crop_files(
        piece_paths(folder),
        out,
        index_ids=ids,
        file_of=file_of,
        projection=folder / "projection.npz",
        dims=dims,
        rule=CropRule(rescore=rescore, shortlist=shortlist, weight=weight),
    )
    typer.echo(json.dumps(facts))
    copied = copy_bundle(s3, index, SOURCE, NAME)
    copy_whole(s3, bucket_name(settings, Bucket.IMAGES), FULL_NAME, index, NAME)
    upload_files(s3, index, out.parent, crop_file_paths(out))
    typer.echo(f"{NAME}: {copied} file(s) of {SOURCE} copied, crop files sent")


@scw_app.command("crop-measure-run", hidden=True)
def scw_crop_measure_run(index: Annotated[str, typer.Option(help="Bundle to measure.")]) -> None:
    """Search the measure's queries crops off and on. Runs on a GPU machine, never the Mac."""
    from miroir.remote.crop_measure import run_crop_measure  # noqa: PLC0415 (machine only)

    run_crop_measure(get_settings(), index)


@scw_app.command("crop-measure")
def scw_crop_measure(
    index: Annotated[str, typer.Option(help="Bundle with crops.")] = "pop-all-crops",
    max_hours: MaxHours = 2.0,
) -> None:
    """T1, T2, T2b, T3, T5, PARTS and VET (tune halves) searched on a GPU machine with the
    site's code, crops off and on; hit rates and times go to eval/results/R3."""
    from miroir.config import eval_result_path  # noqa: PLC0415
    from miroir.jobs.crop_embed import crop_offers, in_stock  # noqa: PLC0415
    from miroir.jobs.crop_measure import (  # noqa: PLC0415
        fetch_out,
        measure_launch,
        prepare,
        read_out,
        summarize,
    )
    from miroir.jobs.crop_slice import slice_dir  # noqa: PLC0415
    from miroir.jobs.objects import s3_client  # noqa: PLC0415
    from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415
    from miroir.jobs.scw_bulk import run_unattended  # noqa: PLC0415

    settings = get_settings()
    s3 = s3_client(settings)
    ids = (slice_dir(settings) / "ids.txt").read_text(encoding="utf-8").split()
    spec = prepare(settings, s3, index, ids)
    typer.echo(f"{len(spec.queries)} queries ({len(spec.skipped)} without vectors left out)")
    api = InstanceApi(settings)
    try:
        offers = [o for o in in_stock(api, crop_offers()) if o.commercial_type != "RENDER-S"]
    finally:
        api.close()
    run_unattended(
        settings, measure_launch(settings, index), offers=offers, max_hours=max_hours,
        purpose="crop measure",
    )  # fmt: skip
    out = read_out(fetch_out(settings, s3, index).read_bytes())
    table = summarize(out)
    result = {
        "index": index,
        "gpu": out.gpu,
        "sets": table,
        "photo_search_ms": {k: list(v) for k, v in out.photo_search_ms.items()},
        "memory_gb": dict(out.memory_gb),
        "skipped": list(spec.skipped),
        "vet_ranks": {r.query_id: [r.off, r.on] for r in out.ranks if r.set_id == "VET"},
    }
    path = eval_result_path("R3", "SL", index)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    typer.echo(json.dumps(table, indent=1))
    typer.echo(f"written to {path}")


@scw_app.command("build-pop-index", hidden=True)
def scw_build_pop_index(
    work: Annotated[Path, typer.Option(help="Scratch folder for the gathered vectors.")],
) -> None:
    """Gather the vectors and build the `pop-all` bundle. Runs on a Scaleway machine."""
    from miroir.index.pop_all import build_on_machine  # noqa: PLC0415

    build_on_machine(get_settings(), work)


@scw_app.command("pop-index")
def scw_pop_index(max_hours: MaxHours = 6.0) -> None:
    """Build the `pop-all` index bundle on a machine with much memory; it lands in the index
    bucket. The vectors must be in the images bucket (`pop-download`, `pop-embed`)."""
    from miroir.index.pop_all import input_files, merged_tables, upload_inputs  # noqa: PLC0415
    from miroir.jobs.objects import s3_client  # noqa: PLC0415
    from miroir.jobs.scaleway import MEMORY_OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import index_launch, run_unattended  # noqa: PLC0415

    settings = get_settings()
    merged = settings.scaleway_dir / "pop-all-tables"
    records, images = merged_tables(settings, merged)
    typer.echo(f"tables: {records:,} records, {images:,} image rows")
    sent = upload_inputs(settings, s3_client(settings), input_files(settings, merged))
    typer.echo(f"inputs: {sent} file(s) sent")
    done = run_unattended(
        settings,
        index_launch(settings),
        offers=MEMORY_OFFERS,
        max_hours=max_hours,
        purpose="pop-index",
        poll_s=120.0,
    )
    for status in done:
        typer.echo(status.line())


@scw_app.command("pop-coverage")
def scw_pop_coverage() -> None:
    """Which records of the `pop-all` index have vectors, and why the others do not."""
    from miroir.index.coverage import write_report  # noqa: PLC0415
    from miroir.jobs.objects import s3_client  # noqa: PLC0415

    settings = get_settings()
    out = settings.index_dir / "coverage-pop-all.json"
    tables = settings.scaleway_dir / "pop-all-tables"
    report = write_report(settings, s3_client(settings), tables, out)
    typer.echo(f"files with vectors: {report['files_with_vectors']:,} of {report['files']:,}")
    typer.echo(f"written to {out}")


@embed_app.command("scw")
def embed_scw(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = "screening-20k-publishable",
    queries: Annotated[
        list[str] | None,
        typer.Option(help="Embed these test sets' tune query images instead; repeat it."),
    ] = None,
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    all_images: Annotated[
        bool,
        typer.Option(
            "--all-images",
            help="Also images of records that are not publishable (full-sample index only).",
        ),
    ] = False,
    reuse: Annotated[
        list[str] | None,
        typer.Option(help="Vector run of the same model and settings to take vectors from."),
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """Embed a gallery, or test sets' query images, with an open model on a Scaleway GPU.

    Starts a machine, runs, deletes it. Paid: probe first. Commons query photos are fetched on
    the Mac and sent as bytes.
    """
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.remote_spec import option as find_option  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    find_option(option)  # fail before starting a machine
    if queries and probe:
        raise typer.BadParameter("--probe works with --gallery only")
    settings = get_settings()
    purpose = f"embed {option} {'+'.join(queries) if queries else gallery}"
    with (
        open_commons_images(settings) as commons,
        open_scaleway(settings, max_hours=max_hours, purpose=purpose) as calls,
    ):
        runner = GpuRunner(
            settings,
            calls,
            key_usage_usd=_openrouter_key_usage(),
            commons=commons,
            all_images=all_images,
        )
        if queries:
            done = runner.embed_queries(option, queries, max_usd=max_usd)
            typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")
            return
        if probe:
            result, size = runner.probe(option, gallery)
            typer.echo(
                f"{option} probe: {result.images} images ({result.per_image_s * 1000:.0f} ms "
                f"each). Projected GPU time for {gallery} ({size:,} images): "
                f"${result.projected_usd(size):.2f}"
            )
            return
        written = runner.embed_gallery(option, gallery, max_usd=max_usd, reuse=reuse or ())
    for done in written:
        typer.echo(f"{done.info.gallery} ({done.info.pooling}): {len(done.ids)} vectors")


def main() -> None:
    app()


detect_app = typer.Typer(
    no_args_is_help=True, help="Step C: object outlines on the Jeu de Paume room photos."
)
app.add_typer(detect_app, name="detect")


@detect_app.command("locate")
def detect_locate(max_hours: MaxHours = 1.0) -> None:
    """Find where each T3 tune artwork hangs in its room photos, on a GPU. Paid."""
    from miroir.eval.detect import run_locate  # noqa: PLC0415
    from miroir.jobs.gpu_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    purpose = "detect locate"
    with open_scaleway(settings, max_hours=max_hours, purpose=purpose, match=True) as calls:
        located = run_locate(settings, calls, paid)
    found = sum(x.box is not None for x in located)
    typer.echo(f"{found} of {len(located)} (room photo, artwork) pairs located")


@detect_app.command("rooms")
def detect_rooms(
    option: Annotated[str, typer.Option(help="Detector id (C1-owlv2-base...) or C0-located.")],
    max_hours: MaxHours = 1.0,
) -> None:
    """Run one detector on the room photos and search every box like the site. Paid."""
    from miroir.eval.detect import run_detector  # noqa: PLC0415
    from miroir.jobs.gpu_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    purpose = f"detect {option}"
    with open_scaleway(settings, max_hours=max_hours, purpose=purpose, match=True) as calls:
        found = run_detector(settings, calls, paid, option)
    boxes = sum(len(r.boxes) for r in found.rooms)
    typer.echo(f"{option}: {boxes} raw boxes on {len(found.rooms)} photos, ${found.cost_usd:.4f}")


@eval_app.command("detect")
def eval_detect(
    option: Annotated[str, typer.Option(help="Detector id whose boxes `detect rooms` saved.")],
    round_id: Annotated[str, typer.Option("--round")] = "R3",
) -> None:
    """Score one detector end to end on T3 tune (tapped outline in the top 5), rule tuned."""
    from miroir.eval.detect import summary, write_scores  # noqa: PLC0415

    tried = write_scores(get_settings(), option, round_id=round_id)
    for score in tried[:3]:
        typer.echo(f"every outline: {summary(score.every)}")
        typer.echo(f"first view: top 5 {score.first.hit_at(5):.3f}")


@eval_app.command("detect-order")
def eval_detect_order() -> None:
    """Score orders of the site's outlines on T3 tune: which ones the first view shows."""
    from miroir.eval.detect import write_order_report  # noqa: PLC0415

    path = write_order_report(get_settings())
    typer.echo(f"wrote {path} and C-detail/order.json beside it")


@rerank_app.command("site")
def rerank_site(
    max_usd: Annotated[float, typer.Option(help="Stop once matching has cost this much.")],
    max_hours: MaxHours = 2.0,
) -> None:
    """E1 in the website: match object searches' top 30 on a GPU with the site's code. Paid."""
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.eval.site_rerank import match_sets, search_sets  # noqa: PLC0415
    from miroir.jobs.gpu_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415
    from miroir.web.bundle import SiteBundle  # noqa: PLC0415

    settings = get_settings()
    bundle = SiteBundle.load(settings.web_index_dir(WEB_INDEX))
    searched = search_sets(settings, bundle)
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with (
        open_commons_images(settings) as photos,
        open_scaleway(settings, max_hours=max_hours, purpose="rerank site", match=True) as calls,
    ):
        cost = match_sets(settings, searched, calls, paid, photos, max_usd=max_usd)
    typer.echo(f"site re-rank matched, ${cost:.4f}; next: uv run miroir eval rerank-site")


@eval_app.command("rerank-site")
def eval_rerank_site(round_id: Annotated[str, typer.Option("--round")] = "R3") -> None:
    """Score E1 in the website on T3, T5, T2b; set the object cut-off again on T8."""
    from miroir.eval.site_rerank import search_sets, write_report  # noqa: PLC0415
    from miroir.web.bundle import SiteBundle  # noqa: PLC0415

    settings = get_settings()
    bundle = SiteBundle.load(settings.web_index_dir(WEB_INDEX))
    summary = write_report(settings, bundle, search_sets(settings, bundle), round_id=round_id)
    typer.echo(summary)


@detect_app.command("single")
def detect_single(
    option: Annotated[str, typer.Option(help="Detector id, e.g. C2-gdino-tiny.")],
    max_hours: MaxHours = 1.0,
) -> None:
    """Count the outlines the site would draw on photos of one object (T5, T2b). Paid."""
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.eval.detect import run_single_objects  # noqa: PLC0415
    from miroir.jobs.gpu_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    purpose = f"detect single {option}"
    with (
        open_commons_images(settings) as commons,
        open_scaleway(settings, max_hours=max_hours, purpose=purpose, match=True) as calls,
    ):
        counts = run_single_objects(settings, calls, paid, option, commons)
    for set_id, found in counts.items():
        none = sum(n == 0 for n in found)
        typer.echo(f"{set_id}: {none} of {len(found)} photos with no outline; counts {found}")


@web_app.command("sure")
def web_sure(round_id: Annotated[str, typer.Option("--round")] = "R3") -> None:
    """Set when an object search counts as sure (margin rule) on the tune halves."""
    from miroir.eval.site_rerank import search_sets  # noqa: PLC0415
    from miroir.eval.sure import write_sure_rule  # noqa: PLC0415
    from miroir.web.bundle import SiteBundle  # noqa: PLC0415

    settings = get_settings()
    bundle = SiteBundle.load(settings.web_index_dir(WEB_INDEX))
    chosen = write_sure_rule(settings, search_sets(settings, bundle), round_id=round_id)
    typer.echo(
        f"margin {chosen.margin}: {chosen.all_sure:.0%} sure, {chosen.all_right:.0%} right "
        f"({', '.join(f'{k} {v:.0%}' for k, v in chosen.right.items())}); T8 {chosen.t8_sure:.0%}"
    )


@web_app.command("levels")
def web_levels(round_id: Annotated[str, typer.Option("--round")] = "R3") -> None:
    """Set the level each result card shows ("Très proche", "Proche", "Ressemblance") on the
    tune halves, from stored query vectors (NumPy on the Mac, about 3 minutes the first time)."""
    from miroir.eval.levels import (  # noqa: PLC0415
        not_found_shares,
        rank_tune_sets,
        write_level_rules,
    )

    settings = get_settings()
    rows = rank_tune_sets(settings)
    reports = write_level_rules(settings, rows, round_id=round_id)
    for report in reports:
        rule = report.rule
        typer.echo(
            f"{report.route}: très proche score >= {rule.very_close_score}, gap >= "
            f"{rule.very_close_gap}; proche score >= {rule.close_score}, gap >= {rule.close_gap}"
        )
        for level, c in report.levels.items():
            typer.echo(f"  {level}: {c.searches:.1%} of searches, {c.right:.1%} right")
    for set_id, share in not_found_shares(rows, reports).items():
        typer.echo(f"{set_id} (record not in the index): 'Très proche' on {share:.1%}")
    typer.echo("levels.json written into the bundle; `miroir web upload` sends it.")


archive_app = typer.Typer(
    no_args_is_help=True,
    help="T1 archive prints: search the print without its mount, read its caption.",
)
app.add_typer(archive_app, name="archive")


@archive_app.command("remote")
def archive_remote(
    model: Annotated[
        list[str], typer.Option(help="Caption reader(s), from miroir.caption.CAPTION_MODELS.")
    ],
    max_hours: MaxHours = 1.0,
) -> None:
    """On a GPU: embed the T1 tune photos whole and cut from their mounts, read their
    captions. Paid; each part is kept and not asked again."""
    from miroir.eval.archive import run_remote  # noqa: PLC0415
    from miroir.jobs.gpu_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with open_scaleway(settings, max_hours=max_hours, purpose="archive remote") as calls:
        cost = run_remote(settings, calls, paid, model)
    typer.echo(f"done, ${cost:.4f}")


@archive_app.command("score")
def archive_score(
    model: Annotated[str, typer.Option(help="Caption reader whose captions to use.")],
    only: Annotated[
        list[str] | None, typer.Option(help="Rules to score (miroir.eval.archive.RULES).")
    ] = None,
    caption_a7: Annotated[
        list[float] | None,
        typer.Option(help="Caption words' text-vector weight to try (BM25 gets the rest)."),
    ] = None,
) -> None:
    """Score each way of using the mount and caption on T1 tune, with the site's search."""
    from miroir.eval.archive import RULES, score  # noqa: PLC0415

    rules = {k: v for k, v in RULES.items() if not only or k in only}
    rows = score(get_settings(), model, rules, caption_a7 or ())
    for row in rows:
        hits = ", ".join(f"hit@{k} {100 * v:.1f}" for k, v in row.hits.items())
        typer.echo(f"{row.label}: {hits} ({row.queries} queries)")
    named, right = rows[0].with_place, rows[0].right_place
    typer.echo(f"captions naming a known commune: {named}, the right one: {right}")


# --- the demo machine (`miroir.jobs.demo`) ---------------------------------------------------


def _need_demo() -> tuple[Demo, DemoMachine]:
    """The demo's handle and its machine; exits when `miroir demo create` has not run."""
    from miroir.jobs.demo import Demo, find_demo  # noqa: PLC0415
    from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415

    settings = get_settings()
    demo = Demo(settings, InstanceApi(settings))
    machine = find_demo(demo.api)
    if machine is None:
        typer.echo("no demo machine: `uv run miroir demo create` makes it (once)", err=True)
        raise typer.Exit(1)
    return demo, machine


def _need_running(machine: DemoMachine) -> str:
    if machine.state != "running" or machine.address is None:
        typer.echo(f"the demo machine is {machine.state}: `uv run miroir demo up`", err=True)
        raise typer.Exit(1)
    return machine.address


DemoIndex = Annotated[str, typer.Option("--index", help="Bundle in the index bucket.")]


@demo_app.command("create")
def demo_create(index: DemoIndex = DEMO_INDEX) -> None:
    """Make the demo machine (once): flexible IP, L4 machine, image, index, HTTPS. 12 min."""
    from miroir.jobs.demo import (  # noqa: PLC0415
        Demo,
        create_demo,
        demo_key,
        demo_shell,
        install,
        rest_eur_per_hour,
        wait_site,
    )
    from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415

    settings = get_settings()
    demo = Demo(settings, InstanceApi(settings))
    public = demo_key(settings).with_suffix(".pub").read_text().strip()
    machine = create_demo(demo, public)
    assert machine.address is not None
    assert machine.url is not None
    typer.echo(f"machine {machine.server_id} running at {machine.address}; installing")
    shell = demo_shell(settings, machine.address)
    shell.wait()
    install(shell, settings, machine, index)
    waited = wait_site(machine.url)
    typer.echo(f"site up at {machine.url} ({waited:.0f} s after install)")
    typer.echo(
        f"off between demos it costs €{rest_eur_per_hour() * 730:.2f} a month; "
        "`uv run miroir demo down` powers it off now"
    )


@demo_app.command("install")
def demo_install(index: DemoIndex = DEMO_INDEX) -> None:
    """Send this checkout's code to the running machine, rebuild, restart the site."""
    from miroir.jobs.demo import demo_shell, install, served, wait_site  # noqa: PLC0415

    demo, machine = _need_demo()
    address = _need_running(machine)
    install(demo_shell(demo.settings, address), demo.settings, machine, index)
    served(demo, index)
    assert machine.url is not None
    waited = wait_site(machine.url)
    typer.echo(f"site up at {machine.url} ({waited:.0f} s to load)")


@demo_app.command("crops")
def demo_crops(
    index: DemoIndex = "pop-all-crops",
    gb: Annotated[int, typer.Option(help="Size of the volume for the whole crop vectors.")] = 200,
) -> None:
    """Serve a bundle with crops on the running demo machine: a Block Storage volume for its
    whole crop vectors (made, attached and mounted the first time, about €19 a month), then
    `install` with that bundle. Back to the bundle without crops: `demo install`."""
    from miroir.jobs.demo import add_crops, demo_shell, install, served, wait_site  # noqa: PLC0415

    demo, machine = _need_demo()
    address = _need_running(machine)
    shell = demo_shell(demo.settings, address)
    state = add_crops(demo, machine, shell, index, gb)
    typer.echo(f"crop volume {state.crops_volume} ({state.extra_gb} GB) mounted")
    install(shell, demo.settings, machine, index)
    served(demo, index)
    assert machine.url is not None
    waited = wait_site(machine.url)
    typer.echo(f"site up at {machine.url} with {index} ({waited:.0f} s to load)")


@demo_app.command("up")
def demo_up(
    idle_minutes: Annotated[
        int | None,
        typer.Option(help="Power off after this long with no visitor (default from settings)."),
    ] = None,
    *,
    gpu_only: Annotated[
        bool, typer.Option("--gpu-only", help="Wait for a GPU rather than run on the CPU.")
    ] = False,
    only: Annotated[
        str | None,
        typer.Option("--type", help="Run as this type only, e.g. POP2-HM-12C-96G (a check)."),
    ] = None,
    index: Annotated[
        str | None,
        typer.Option("--index", help="Bundle in the index bucket (default: the one installed)."),
    ] = None,
) -> None:
    """Power the demo machine on, wait until the site answers, print its address.

    When no L4 is free it runs as the next machine type that is (DEMO_OFFERS): an L40S, or a
    machine without a GPU, where searches are slower and captions are not read."""
    import time  # noqa: PLC0415

    from miroir.jobs.demo import (  # noqa: PLC0415
        demo_shell,
        fit_hardware,
        idle_command,
        offer_of,
        read_state,
        wait_site,
    )

    demo, machine = _need_demo()
    start = time.monotonic()
    machine = demo.power_on(machine, allow_cpu=not gpu_only, only=only)
    address = _need_running(machine)
    booted = time.monotonic() - start
    shell = demo_shell(demo.settings, address)
    shell.wait()  # "running" comes a little before SSH answers
    bundle = index or read_state(demo.state_path).index or DEMO_INDEX
    gpu = fit_hardware(shell, machine, demo.settings, bundle)
    shell.run(idle_command(idle_minutes), timeout=60)
    assert machine.url is not None
    wait_site(machine.url)
    idle = idle_minutes or demo.settings.demo_idle_minutes
    offer = offer_of(machine.commercial_type)
    typer.echo(
        f"{machine.url}\nup in {time.monotonic() - start:.0f} s ({booted:.0f} s to boot) as "
        f"{offer.commercial_type}{'' if gpu else ' (no GPU: slower, no captions)'}, "
        f"€{offer.eur_per_hour:.2f} an hour; powers itself off after {idle} min with no visitor"
    )


@demo_app.command("down")
def demo_down() -> None:
    """Power the demo machine off. Its disk and address stay; only they are billed."""
    from miroir.jobs.demo import rest_eur_per_hour  # noqa: PLC0415

    demo, machine = _need_demo()
    machine = demo.power_off(machine)
    typer.echo(
        f"off; at rest €{rest_eur_per_hour():.3f} an hour (€{rest_eur_per_hour() * 730:.2f} "
        f"a month). Spent on Scaleway so far: €{demo.ledger.total_eur():.2f}"
    )


@demo_app.command("status")
def demo_status() -> None:
    """State, address, time on, this session's cost, and the ledger's total."""
    import time  # noqa: PLC0415

    from miroir.jobs.demo import LAST_VISIT, ROOT, demo_shell, rest_eur_per_hour  # noqa: PLC0415

    demo, machine = _need_demo()
    state = demo.settle(machine)
    typer.echo(f"{machine.state}  {machine.url}  ({machine.server_id}, {machine.zone})")
    if state.session_start is not None:
        on = (time.time() - state.session_start) / 60
        typer.echo(f"on for {on:.0f} min, €{demo.session_cost(state):.2f} so far")
    if machine.state == "running" and machine.address is not None:
        seen = demo_shell(demo.settings, machine.address).run(
            f"stat -c %Y {ROOT}/state/{LAST_VISIT} 2>/dev/null || true", timeout=30
        )
        if seen.strip():
            typer.echo(f"last visitor {(time.time() - float(seen)) / 60:.0f} min ago")
    typer.echo(
        f"at rest €{rest_eur_per_hour() * 730:.2f} a month; "
        f"spent on Scaleway: €{demo.ledger.total_eur():.2f} of €{demo.settings.scw_budget_eur:.2f}"
    )


@demo_app.command("logs")
def demo_logs(
    container: Annotated[str, typer.Option(help="miroir-site, miroir-watch or caddy.")] = (
        "miroir-site"
    ),
    lines: Annotated[int, typer.Option(help="Last lines to show.")] = 60,
) -> None:
    """The last lines a container on the running machine wrote (no secret is ever logged)."""
    from miroir.jobs.demo import CONTAINERS, demo_shell  # noqa: PLC0415

    if container not in CONTAINERS:
        raise typer.BadParameter(f"one of {', '.join(CONTAINERS)}")
    demo, machine = _need_demo()
    shell = demo_shell(demo.settings, _need_running(machine))
    typer.echo(shell.run(f"docker logs --tail {int(lines)} {container} 2>&1", timeout=60))


@demo_app.command("chat-logs")
def demo_chat_logs() -> None:
    """Copy the conversation page's logs (JSON lines, no images) from the running machine."""
    import shlex  # noqa: PLC0415

    from miroir.jobs.demo import CHAT_LOGS, ROOT, demo_shell  # noqa: PLC0415
    from miroir.web.chatlog import pull  # noqa: PLC0415

    demo, machine = _need_demo()
    shell = demo_shell(demo.settings, _need_running(machine))
    folder = f"{ROOT}/state/{CHAT_LOGS}"

    def names() -> list[str]:
        found = shell.run(f"cd {folder} 2>/dev/null && find . -name '*.jsonl' || true")
        return [line.removeprefix("./") for line in found.splitlines() if line.strip()]

    def read(name: str) -> bytes:
        return shell.run(f"cat {shlex.quote(f'{folder}/{name}')}").encode()

    target = demo.settings.chat_logs_dir / "hosted"
    written = pull(names, read, target)
    typer.echo(f"{len(written)} log file(s) new or grown in {target}")


@demo_app.command("destroy")
def demo_destroy(
    *, yes: Annotated[bool, typer.Option("--yes", help="Delete without asking.")] = False
) -> None:
    """Delete the demo machine, its disk and its address (the URL changes after that)."""
    from miroir.jobs.scaleway import delete_machine  # noqa: PLC0415

    demo, machine = _need_demo()
    if not yes and not typer.confirm(f"delete the demo machine, its disk and {machine.address}?"):
        raise typer.Abort
    if machine.state == "running":
        machine = demo.power_off(machine)
    demo.settle(machine)
    delete_machine(demo.api, machine.zone, machine.server_id)
    demo.state_path.unlink(missing_ok=True)
    typer.echo("deleted")


@demo_app.command("serve", hidden=True)
def demo_serve(
    index: Annotated[Path, typer.Option()],
    state: Annotated[Path, typer.Option()],
    port: Annotated[int, typer.Option()],
) -> None:
    """The site, inside the image on the demo machine. Never run on the Mac."""
    from miroir.remote.site import serve_site  # noqa: PLC0415 (machine only)

    serve_site(get_settings(), index=index, state=state, port=port)


@demo_app.command("watch", hidden=True)
def demo_watch(
    server_id: Annotated[str, typer.Option()],
    zone: Annotated[str, typer.Option()],
    idle_minutes: Annotated[int, typer.Option()],
    max_hours: Annotated[float, typer.Option()],
    state: Annotated[Path, typer.Option()],
) -> None:
    """Power the demo machine off when nobody uses it. Runs on the machine."""
    import time  # noqa: PLC0415

    from miroir.jobs.demo import watch_idle  # noqa: PLC0415

    watch_idle(
        get_settings(),
        server_id=server_id,
        zone=zone,
        state=state,
        idle_minutes=idle_minutes,
        max_hours=max_hours,
    )
    while True:  # powering off: Docker would only start this again
        time.sleep(3600)


@demo_app.command("fetch", hidden=True)
def demo_fetch(index: DemoIndex = DEMO_INDEX) -> None:
    """Copy the bundle and the weights from their buckets onto the machine's disk."""
    from miroir.jobs.demo import MOUNTS  # noqa: PLC0415
    from miroir.jobs.objects import (  # noqa: PLC0415
        Bucket,
        bucket_name,
        download_tree,
        link_siblings,
        s3_client,
    )

    settings = get_settings()
    s3 = s3_client(settings)
    root = Path(MOUNTS["web-index"])
    link_siblings(s3, bucket_name(settings, Bucket.INDEX), root, f"{index}/")
    got = download_tree(s3, bucket_name(settings, Bucket.INDEX), root, prefix=f"{index}/")
    weights = download_tree(s3, bucket_name(settings, Bucket.WEIGHTS), Path(MOUNTS["weights"]))
    typer.echo(f"{index}: {got} file(s) copied; weights: {weights} file(s) copied")
