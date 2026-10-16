"""The `miroir` command. Every entry point is a subcommand here: `uv run miroir --help`."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from miroir import logs
from miroir.config import ENV_PASSWORD_KEY, WEB_INDEX, get_settings

if TYPE_CHECKING:
    from miroir.eval.build_sets import BuildReport
    from miroir.eval.h4 import Loaded
    from miroir.eval.results import SetScore
    from miroir.eval.scoring import Interval
    from miroir.jobs.h4_agent import Outcome, Scenario

app = typer.Typer(no_args_is_help=True, add_completion=False)
records_app = typer.Typer(no_args_is_help=True, help="POP records: parse the exports.")
app.add_typer(records_app, name="records")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: vectors, scoring, results.")
app.add_typer(eval_app, name="eval")
testsets_app = typer.Typer(
    no_args_is_help=True, help="Test sets T1, T2, T2b, T3, T5, T8 and galleries."
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
    no_args_is_help=True, help="Full-sample index for the website: PCA, the guess, upload."
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
    no_args_is_help=True, help="The prototype website: build its index, deploy it, run it locally."
)
app.add_typer(web_app, name="web")
geo_app = typer.Typer(
    no_args_is_help=True, help="Step H2: where buildings, objects and query photos are."
)
app.add_typer(geo_app, name="geo")

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


@records_app.command("fetch-probe")
def records_fetch_probe(
    max_usd: Annotated[float, typer.Option(help="Stop before Modal time passes this.")] = 0.4,
) -> None:
    """Time about 1,000 downloads from the POP bucket on Modal CPUs (spec §12, question 2).

    Needs `records build-full` first. Takes about 5 minutes; costs a few cents.
    """
    from miroir.jobs.fetch_probe import run  # noqa: PLC0415

    settings = get_settings()
    summary = run(settings, max_usd, settings.full_fetch_probe_dir)
    typer.echo(json.dumps(summary, indent=2))


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


@embed_app.command("modal")
def embed_modal(
    option: Annotated[str, typer.Option(help="Modal option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[
        str, typer.Option(help="Gallery list. Only publishable images may go to Modal.")
    ] = "screening-20k-publishable",
    queries: Annotated[
        list[str] | None,
        typer.Option(
            help="Embed these test sets' tune query images instead of a gallery; repeat it."
        ),
    ] = None,
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    workers: Annotated[
        int, typer.Option(help="Chunks sent at once, each to its own GPU container.")
    ] = 1,
    reuse: Annotated[
        list[str] | None,
        typer.Option(help="Vector run of the same model and settings to take vectors from."),
    ] = None,
    all_images: Annotated[
        bool,
        typer.Option(
            "--all-images",
            help="Also images of records that are not publishable (full-sample index only).",
        ),
    ] = False,
) -> None:
    """Embed a gallery, or test sets' query images, with an open model on a Modal GPU.

    Paid: probe first. Commons query photos are fetched on the Mac and sent as bytes.
    """
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415
    from miroir.jobs.modal_spec import option as find_option  # noqa: PLC0415

    find_option(option)  # fail before starting Modal
    if queries and probe:
        raise typer.BadParameter("--probe works with --gallery only")
    settings = get_settings()
    with open_commons_images(settings) as commons, open_modal(settings) as calls:
        runner = ModalRunner(
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
                f"{option} probe: {result.images} images cost ${result.cost_usd:.4f} "
                f"({result.fixed_s:.0f} s start-up, {result.per_image_s * 1000:.0f} ms per image). "
                f"Projected for {gallery} ({size:,} images): ${result.projected_usd(size):.2f}"
            )
            return
        written = runner.embed_gallery(
            option, gallery, max_usd=max_usd, workers=workers, reuse=reuse or ()
        )
    for done in written:
        typer.echo(
            f"{done.info.gallery} ({done.info.pooling}): {len(done.ids)} vectors, "
            f"${done.info.cost_usd:.4f}"
        )


@embed_app.command("modal-f2")
def embed_modal_f2(
    option: Annotated[str, typer.Option(help="Modal option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[str, typer.Option(help="Gallery whose F2 photos to embed.")],
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
) -> None:
    """F2: embed the Commons photos added to a gallery's buildings (sent from the Mac)."""
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415
    from miroir.jobs.supplied import open_gallery_photos  # noqa: PLC0415
    from miroir.records.store import read_commons  # noqa: PLC0415

    settings = get_settings()
    ids = [p.image_id for p in read_commons(settings.commons_gallery_parquet(gallery))]
    with open_gallery_photos(settings, gallery) as photos, open_modal(settings) as calls:
        runner = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        done = runner.embed_supplied(option, f"F2-{gallery}", ids, photos, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("modal-grey")
def embed_modal_grey(
    option: Annotated[str, typer.Option(help="Modal option id, e.g. A3-megaloc-322.")],
    gallery: Annotated[str, typer.Option(help="Gallery whose colour images to grey.")],
    queries: Annotated[
        list[str] | None, typer.Option(help="Also grey these sets' tune query images.")
    ] = None,
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
) -> None:
    """F1: embed grey copies of every image with colour in a gallery and the query sets."""
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415
    from miroir.jobs.supplied import grey_candidates, open_grey_copies  # noqa: PLC0415

    settings = get_settings()
    with open_grey_copies(settings) as grey:
        ids = grey.not_grey(grey_candidates(settings, gallery, queries or ()))
        with open_modal(settings) as calls:
            runner = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
            done = runner.embed_supplied(option, f"grey-{gallery}", ids, grey, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("modal-images")
def embed_modal_images(
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = "screening-20k-publishable",
    *,
    all_images: Annotated[
        bool,
        typer.Option(
            "--all-images",
            help="Also images of records that are not publishable (full-sample index only).",
        ),
    ] = False,
) -> None:
    """Fetch a gallery's images once into the Modal image volume, so model runs read them there."""
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415

    settings = get_settings()
    with open_modal(settings) as calls:
        runner = ModalRunner(
            settings, calls, key_usage_usd=_openrouter_key_usage(), all_images=all_images
        )
        counts = runner.fill_images(gallery)
    typer.echo(f"{gallery}: {counts}")


@r0_app.command("run")
def r0_run(
    model: Annotated[
        str, typer.Option(help="dinov2-giant-224 (expect 48.9 mAP) or dinov3-vit7b16-224 (56.5).")
    ] = "dinov2-giant-224",
) -> None:
    """Download AmsterTime into a Modal volume and embed it there; vectors come back here."""
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415

    settings = get_settings()
    with open_modal(settings) as calls:
        runner = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
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
) -> None:
    """Embed T7/T7en tune words and the refine communes with EmbeddingGemma 2 on Modal. Paid."""
    from miroir.eval.wordsearch import TEXT_LABEL, text_inputs  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415

    settings = get_settings()
    ids, texts = text_inputs(settings)
    with open_modal(settings) as calls:
        runner = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
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


@rerank_app.command("modal")
def rerank_modal(
    *,
    vectors: Annotated[str, typer.Option(help="Vector run whose search gives the shortlists.")],
    test_set: Annotated[str, typer.Option("--set", help="Test set id; tune half only.")],
    max_usd: Annotated[float, typer.Option(help="Stop once matching has cost this much.")],
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = (
        "screening-20k-publishable-fetched"
    ),
) -> None:
    """Match each tune query against its top 50 entities' best images, on a Modal GPU. Paid."""
    from miroir.eval.rerank import tasks  # noqa: PLC0415
    from miroir.eval.run import matches_name, rerank_shortlists  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415

    settings = get_settings()
    lists, _, _ = rerank_shortlists(settings, vector_run=vectors, set_id=test_set, gallery=gallery)
    name = matches_name(vectors, test_set)
    with open_modal(settings) as calls:
        runner = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
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


@index_app.command("upload")
def index_upload(
    runs: Annotated[list[str], typer.Option("--run", help="Vector run to copy; repeat.")],
) -> None:
    """Copy vector runs and the full-sample .npz files into the `miroir-index` Modal volume."""
    from miroir.index.upload import upload  # noqa: PLC0415 (imports the modal client)

    for line in upload(get_settings(), runs):
        typer.echo(line)


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
    """Copy the bundle to the Modal volume the website reads."""
    from miroir.web.ops import upload_index  # noqa: PLC0415

    size = upload_index(get_settings(), index)
    typer.echo(f"uploaded {size / 1e6:.0f} MB")


@web_app.command("places")
def web_places(
    index: WebIndex = WEB_INDEX,
    *,
    upload: Annotated[bool, typer.Option(help="Also copy places.npz alone to the volume.")] = False,
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
        typer.echo(f"{PLACES_FILE} uploaded; `miroir web deploy` makes the site read it")


@web_app.command("secret")
def web_secret(
    *,
    rotate: Annotated[bool, typer.Option(help="Replace the password.")] = False,
) -> None:
    """Make the shared password: kept in .env (`ENV_PASSWORD_KEY`) and the Modal secret."""
    from miroir.web.ops import set_secret  # noqa: PLC0415

    changed = set_secret(rotate=rotate)
    typer.echo(f"new password in .env ({ENV_PASSWORD_KEY})" if changed else "password unchanged")


@web_app.command("deploy")
def web_deploy() -> None:
    """Deploy the `miroir-web` Modal app (GPU search sleeps when idle)."""
    from miroir.web.ops import deploy  # noqa: PLC0415

    typer.echo(f"deployed: {deploy()}")


@web_app.command("dev")
def web_dev(
    index: WebIndex = WEB_INDEX,
    port: Annotated[int, typer.Option(help="Local port.")] = 8000,
    *,
    real: Annotated[
        bool, typer.Option(help="Search with the deployed models on Modal (costs GPU time).")
    ] = False,
) -> None:
    """Run the website on this Mac with fake models (no model is loaded here)."""
    from miroir.web.ops import serve_locally  # noqa: PLC0415

    serve_locally(get_settings(), name=index, port=port, real=real)


def main() -> None:
    app()


detect_app = typer.Typer(
    no_args_is_help=True, help="Step C: object outlines on the Jeu de Paume room photos."
)
app.add_typer(detect_app, name="detect")


@detect_app.command("locate")
def detect_locate() -> None:
    """Find where each T3 tune artwork hangs in its room photos, on a Modal GPU. Paid."""
    from miroir.eval.detect import run_locate  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with open_modal(settings) as calls:
        located = run_locate(settings, calls, paid)
    found = sum(x.box is not None for x in located)
    typer.echo(f"{found} of {len(located)} (room photo, artwork) pairs located")


@detect_app.command("rooms")
def detect_rooms(
    option: Annotated[str, typer.Option(help="Detector id (C1-owlv2-base...) or C0-located.")],
) -> None:
    """Run one detector on the room photos and search every box like the site. Paid."""
    from miroir.eval.detect import run_detector  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with open_modal(settings) as calls:
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
) -> None:
    """E1 in the website: match object searches' top 30 on Modal with the site's code. Paid."""
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.eval.site_rerank import match_sets, search_sets  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415
    from miroir.web.bundle import SiteBundle  # noqa: PLC0415

    settings = get_settings()
    bundle = SiteBundle.load(settings.web_index_dir(WEB_INDEX))
    searched = search_sets(settings, bundle)
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with open_commons_images(settings) as photos, open_modal(settings) as calls:
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
) -> None:
    """Count the outlines the site would draw on photos of one object (T5, T2b). Paid."""
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.eval.detect import run_single_objects  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with open_commons_images(settings) as commons, open_modal(settings) as calls:
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
) -> None:
    """On Modal: embed the T1 tune photos whole and cut from their mounts, read their
    captions. Paid; each part is kept and not asked again."""
    from miroir.eval.archive import run_remote  # noqa: PLC0415
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import PaidCall  # noqa: PLC0415
    from miroir.jobs.spend import Budget, SpendLedger  # noqa: PLC0415

    settings = get_settings()
    paid = PaidCall(SpendLedger(settings.spend_ledger), Budget.from_settings(settings))
    with open_modal(settings) as calls:
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
