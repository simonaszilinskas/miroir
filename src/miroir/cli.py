"""The `miroir` command. Every entry point is a subcommand here: `uv run miroir --help`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from miroir import logs
from miroir.config import get_settings

if TYPE_CHECKING:
    from miroir.eval.build_sets import BuildReport
    from miroir.eval.results import SetScore
    from miroir.eval.scoring import Interval

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
r0_app = typer.Typer(no_args_is_help=True, help="Round R0: harness check on AmsterTime (T9).")
app.add_typer(r0_app, name="r0")
vlm_app = typer.Typer(
    no_args_is_help=True, help="Step H1: ask a vision model about query photos, search its answer."
)
app.add_typer(vlm_app, name="vlm")

DEFAULT_GALLERY = "screening-20k-publishable-fetched"
DEFAULT_SETS = ("T1", "T3", "T5")


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
) -> None:
    """Paired bootstrap: is a better than b on the same queries, beyond the error bar?"""
    from miroir.eval.run import compare  # noqa: PLC0415

    for d in compare(a, b, test_set=test_set, half=half, gallery=gallery):
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
        bool, typer.Option("--not-found", help="Also score T8, the not-found measure.")
    ] = False,
) -> None:
    """Score one or more models, fused and maybe expanded, on tune halves; write one result."""
    from miroir.eval.fusion import Method  # noqa: PLC0415
    from miroir.eval.stack import Stack, score_stack  # noqa: PLC0415

    stack = Stack(
        runs=tuple(vectors),
        method=None if method is None else Method(method),
        first_weight=weight,
        qe_alpha=qe_alpha,
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
        not_found=not_found,
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
        str | None, typer.Option(help="Test set (tune half) whose query images to embed.")
    ] = None,
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
) -> None:
    """Embed a gallery or a test set's query images on OpenRouter. Paid: probe first."""
    from miroir.jobs import embed as jobs  # noqa: PLC0415 (keeps `miroir --help` fast)
    from miroir.jobs.factory import open_embedder  # noqa: PLC0415

    if (gallery is None) == (queries is None):
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
            assert queries is not None
            done = jobs.embed_queries(embedder, queries)
        typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("modal")
def embed_modal(
    option: Annotated[str, typer.Option(help="Modal option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[
        str, typer.Option(help="Gallery list. Only publishable images may go to Modal.")
    ] = "screening-20k-publishable",
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
) -> None:
    """Embed a gallery with an open model on a Modal GPU. Paid: probe first."""
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415
    from miroir.jobs.modal_spec import option as find_option  # noqa: PLC0415

    find_option(option)  # fail before starting Modal
    settings = get_settings()
    with open_modal(settings) as calls:
        runner = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage())
        if probe:
            result, size = runner.probe(option, gallery)
            typer.echo(
                f"{option} probe: {result.images} images cost ${result.cost_usd:.4f} "
                f"({result.fixed_s:.0f} s start-up, {result.per_image_s * 1000:.0f} ms per image). "
                f"Projected for {gallery} ({size:,} images): ${result.projected_usd(size):.2f}"
            )
            return
        done = runner.embed_gallery(option, gallery, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@embed_app.command("modal-images")
def embed_modal_images(
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = "screening-20k-publishable",
) -> None:
    """Fetch a gallery's images once into the Modal image volume, so model runs read them there."""
    from miroir.jobs.modal_app import open_modal  # noqa: PLC0415 (imports the modal client)
    from miroir.jobs.modal_run import ModalRunner  # noqa: PLC0415

    settings = get_settings()
    with open_modal(settings) as calls:
        counts = ModalRunner(settings, calls, key_usage_usd=_openrouter_key_usage()).fill_images(
            gallery
        )
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
) -> None:
    """Ask a vision model about each publishable query image. Paid: probe first."""
    from miroir.jobs import describe  # noqa: PLC0415
    from miroir.jobs.factory import open_describer  # noqa: PLC0415

    settings = get_settings()
    sets = tuple(test_set or DEFAULT_SETS)
    ids = describe.query_image_ids(settings, sets, gallery)
    with open_describer(settings, model) as describer:
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
            f"{r.commune_given}, right {r.commune_right} of {r.with_commune} with one; "
            f"visible text {r.visible_text} (top 10 gained {r.text_helped}, lost "
            f"{r.text_hurt}); hit@10 by confidence: {bands}"
        )
    if image_run is not None:
        h3, weight = vlm.score_h3(settings, prepared, image_run, image_label, round_id=round_id)
        typer.echo(f"H3 {image_label}+H1, image weight {weight:.1f}: {h3.vectors.notes}")
        _echo_scores(h3.scores)


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


def main() -> None:
    app()
