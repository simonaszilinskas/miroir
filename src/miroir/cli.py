"""The `miroir` command. Every entry point is a subcommand here: `uv run miroir --help`."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from miroir import logs
from miroir.config import get_settings

if TYPE_CHECKING:
    from miroir.eval.results import SetScore

app = typer.Typer(no_args_is_help=True, add_completion=False)
records_app = typer.Typer(no_args_is_help=True, help="POP records: parse the exports.")
app.add_typer(records_app, name="records")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: vectors, scoring, results.")
app.add_typer(eval_app, name="eval")
testsets_app = typer.Typer(no_args_is_help=True, help="Test sets T1, T3, T5, T8 and galleries.")
app.add_typer(testsets_app, name="testsets")
amstertime_app = typer.Typer(no_args_is_help=True, help="AmsterTime (T9, round R0).")
eval_app.add_typer(amstertime_app, name="amstertime")
embed_app = typer.Typer(no_args_is_help=True, help="Embedding runs on remote models.")
app.add_typer(embed_app, name="embed")
r0_app = typer.Typer(no_args_is_help=True, help="Round R0: harness check on AmsterTime (T9).")
app.add_typer(r0_app, name="r0")


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

    report = build(get_settings())
    typer.echo(
        f"{'set':<4} {'tune':>6} {'sealed':>6} {'total':>6} {'dropped':>7} {'spec':>6}"
        f" {'closed-ok':>9}"
    )
    for s in report.sets:
        flag = "" if s.tune + s.sealed + s.dropped == s.spec else "  differs from spec §7"
        typer.echo(
            f"{s.set_id:<4} {s.tune:>6} {s.sealed:>6} {s.tune + s.sealed:>6} {s.dropped:>7}"
            f" {s.spec:>6} {s.closed_ok:>9}{flag}"
        )
    typer.echo(
        f"galleries: full {report.full}, screening {report.screening} "
        f"({report.needed} needed by queries, {report.screening - report.needed} distractors), "
        f"screening publishable {report.screening_publishable}"
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


@eval_app.command("summary")
def eval_summary() -> None:
    """Rebuild the summary table in docs/results.md from every file in eval/results/."""
    from miroir.eval.results import rebuild_summary, results_dir, results_md  # noqa: PLC0415

    count = rebuild_summary(results_dir(), results_md())
    typer.echo(f"docs/results.md: summary rebuilt from {count} result file(s)")


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
