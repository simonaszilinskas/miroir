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
    typer.echo(f"{'set':<4} {'tune':>6} {'sealed':>6} {'total':>6} {'spec':>6} {'closed-ok':>9}")
    for s in report.sets:
        flag = "" if s.tune + s.sealed == s.spec else "  differs from spec §7"
        typer.echo(
            f"{s.set_id:<4} {s.tune:>6} {s.sealed:>6} {s.tune + s.sealed:>6} {s.spec:>6}"
            f" {s.closed_ok:>9}{flag}"
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


@amstertime_app.command("download")
def amstertime_download() -> None:
    """Fetch AmsterTime (0.9 GB, CC0) into the data dir's benchmarks/amstertime."""
    from miroir.eval import amstertime  # noqa: PLC0415

    amstertime.download(get_settings().amstertime_dir)


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


def main() -> None:
    app()
