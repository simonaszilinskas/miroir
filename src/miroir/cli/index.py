"""`miroir index`: PCA and the type guess for the website's index."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Full-sample index for the website: PCA and the guess.",
)


@app.command("pca")
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


@app.command("guess")
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
