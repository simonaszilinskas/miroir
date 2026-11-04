"""`miroir eval`: check vector runs, score them, compare, and round R4."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from miroir.cli.common import MaxHours, key_usage
from miroir.config import get_settings

app = typer.Typer(no_args_is_help=True, help="Evaluation: vectors, scoring, results.")
r4_app = typer.Typer(
    no_args_is_help=True, help="Round R4: the frozen stack on the sealed halves, once."
)
app.add_typer(r4_app, name="r4")


@app.command("check-vectors")
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


@app.command("run")
def eval_run(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R1, R2...")],
    step: Annotated[str, typer.Option(help="Step letter from experiments.md §2, e.g. A.")],
    option: Annotated[str, typer.Option(help="Result file label, e.g. A1-dinov2-vitl14-224.")],
    vectors: Annotated[str, typer.Option(help="Vector run of the gallery.")],
    test_set: Annotated[str, typer.Option("--set", help="Test set id, e.g. T1.")],
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
    """Score a vector run on one test set's tune half: eval/results/<round>/<step>-<option>.json.

    The sealed halves open in round R4 only: `miroir eval r4`.
    """
    from miroir.eval.run import score_test_set  # noqa: PLC0415

    scored = score_test_set(
        get_settings(),
        round_id=round_id,
        step=step,
        option=option,
        vector_run=vectors,
        set_id=test_set,
        query_vector_run=query_vectors,
        gallery=gallery,
        publishable_queries=publishable_queries,
    )
    typer.echo(scored.line())


@r4_app.command("freeze")
def r4_freeze(
    index: Annotated[
        list[str], typer.Option(help="Website bundle to score; repeat it, the chosen one first.")
    ],
    a7_option: Annotated[str, typer.Option(help="EmbeddingGemma 2 option of the bundle.")] = (
        "A7-embeddinggemma-2-512"
    ),
    a3_option: Annotated[str, typer.Option(help="MegaLoc option of the bundle.")] = (
        "A3-megaloc-322"
    ),
) -> None:
    """Write eval/r4.json: a hash of every file that sets each bundle's ranking. Commit it."""
    from miroir.eval import r4  # noqa: PLC0415

    config = r4.freeze(get_settings(), index, a7_option=a7_option, a3_option=a3_option)
    for stack in config.stacks:
        typer.echo(f"{stack.index}: {len(stack.files)} files frozen; queries {stack.a7_query_run}")
    typer.echo(f"wrote {r4.CONFIG}: commit it before `miroir eval r4 embed` and `run`")


@r4_app.command("embed")
def r4_embed(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    max_usd: Annotated[float, typer.Option(help="Refuse a run projected above this.")],
    max_hours: MaxHours = 2.0,
) -> None:
    """Embed the sealed T2, T2b and T8c photos on a Scaleway GPU (paid; the option's probe is
    reused). The run is the one eval/r4.json names."""
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.eval import r4  # noqa: PLC0415
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.remote_spec import option as find_option  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    find_option(option)  # fail before starting a machine
    settings = get_settings()
    config = r4.read_config()
    ids, label = r4.sealed_query_images(settings, config)
    with (
        open_commons_images(settings) as commons,
        open_scaleway(settings, max_hours=max_hours, purpose=f"R4 embed {option}") as calls,
    ):
        runner = GpuRunner(settings, calls, key_usage_usd=key_usage(), commons=commons)
        done = runner.embed_query_images(option, ids, label, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@r4_app.command("run")
def r4_run() -> None:
    """Score every frozen bundle on every sealed half with the site's ranking; once."""
    from miroir.eval import r4  # noqa: PLC0415

    for scored in r4.run(get_settings(), r4.read_config()):
        for s in scored.result.scores:
            typer.echo(s.line())
        for set_id, row in scored.detail.items():
            typer.echo(f"{set_id}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()))


@app.command("crops")
def eval_crops() -> None:
    """Score the crop slice test from the tables `miroir scw crop-slice` brought back."""
    from miroir.eval.crops import score_slice  # noqa: PLC0415

    path = score_slice(get_settings())
    typer.echo(path.read_text(encoding="utf-8"))
    typer.echo(f"written to {path}")


@app.command("crop-short")
def eval_crop_short() -> None:
    """Score shorter crop vectors on the crop slice (`miroir scw crop-short-slice` first)."""
    from miroir.eval.crop_short import measure_slice  # noqa: PLC0415

    typer.echo(f"written to {measure_slice(get_settings())}")


@app.command("compare")
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


@app.command("location")
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
    typer.echo("\n".join(report.lines()))


@app.command("merge-vectors")
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


@app.command("views")
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
            f"{v.view.value:9} {v.queries:4} queries  hit@1 {v.hit_at_1.percent()}  "
            f"hit@10 {v.hit_at_10.percent()}"
        )


@app.command("summary")
def eval_summary() -> None:
    """Rebuild the summary table in docs/results.md from every file in eval/results/."""
    from miroir.eval.results import rebuild_summary, results_dir, results_md  # noqa: PLC0415

    count = rebuild_summary(results_dir(), results_md())
    typer.echo(f"docs/results.md: summary rebuilt from {count} result file(s)")
