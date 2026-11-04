"""`miroir words`: search by words (step B): test set T7, its vectors, its scores."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.cli.common import MaxHours, key_usage
from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Step B, search by words: test set T7, its vectors, its scores.",
)


@app.command("ask")
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
    draws = words.draw_from_settings(settings)
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


@app.command("sets")
def words_sets() -> None:
    """Write T7 and T7en (both halves) from the draw and the writer's saved answers."""
    from miroir.eval import words  # noqa: PLC0415

    settings = get_settings()
    counts = words.build_sets_from_answers(settings, words.draw_from_settings(settings))
    for label, count in counts.items():
        typer.echo(f"{label}: {count} queries")


@app.command("embed")
def words_embed(
    *,
    max_usd: Annotated[float, typer.Option(help="Stop before a call once spent.")] = 0.5,
    max_hours: MaxHours = 1.0,
) -> None:
    """Embed T7/T7en tune words and the refine communes with EmbeddingGemma 2 on a GPU. Paid."""
    from miroir.eval.wordsearch import TEXT_LABEL, TEXT_OPTION, text_inputs  # noqa: PLC0415
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    settings = get_settings()
    ids, texts = text_inputs(settings)
    with open_scaleway(settings, max_hours=max_hours, purpose="words embed") as calls:
        runner = GpuRunner(settings, calls, key_usage_usd=key_usage())
        done = runner.embed_texts(TEXT_OPTION, TEXT_LABEL, ids, texts, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@app.command("score")
def words_score(
    *,
    round_id: Annotated[str, typer.Option("--round", help="R2, R3...")] = "R2",
    refine: Annotated[
        bool, typer.Option("--refine/--no-refine", help="Also score words refining a photo.")
    ] = True,
) -> None:
    """Score word search on T7 tune with the site's ranking; one result file per option."""
    from miroir.eval.wordsearch import score_lines  # noqa: PLC0415

    typer.echo("\n".join(score_lines(get_settings(), round_id=round_id, refine=refine)))
