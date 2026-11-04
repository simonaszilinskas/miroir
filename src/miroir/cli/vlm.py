"""`miroir vlm`: vision models asked about query photos (steps H1, H4)."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.cli.common import key_usage
from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Step H1: ask a vision model about query photos, search its answer.",
)

DEFAULT_GALLERY = "screening-20k-publishable-fetched"
DEFAULT_SETS = ("T1", "T3", "T5")


@app.command("ask")
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


@app.command("h4")
def vlm_h4(
    *,
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
            "--photos/--sample",
            help="The 10-photo test (T1 2, T2 3 with 2 placed, T3 2, T5 3), or a --per-set draw.",
        ),
    ] = True,
    label: Annotated[str | None, typer.Option(help="Report folder name.")] = None,
    max_usd: Annotated[float, typer.Option(help="Stop once this run has paid this much.")] = 2.0,
    workers: Annotated[int, typer.Option(help="Conversations held in parallel.")] = 4,
) -> None:
    """Steps H4 and H5 offline: the site's agentic search on the T sets, with a simulated
    visitor, scored against the site's photo search.

    Paid. Answers are saved and replayed, so a rerun costs nothing. Without --all, a smoke
    test: no result file.
    """
    from miroir.eval import h4  # noqa: PLC0415

    lines = h4.main(
        get_settings(),
        per_set=per_set,
        every_query=every_query,
        photos=photos,
        label=label,
        max_usd=max_usd,
        workers=workers,
        key_usage_usd=key_usage(),
    )
    for line in lines:
        typer.echo(line)
