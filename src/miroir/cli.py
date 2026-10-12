"""The `miroir` command. Every entry point is a subcommand here: `uv run miroir --help`."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir import logs
from miroir.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
records_app = typer.Typer(no_args_is_help=True, help="POP records: parse the exports.")
app.add_typer(records_app, name="records")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluation: vectors, scoring, results.")
app.add_typer(eval_app, name="eval")


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


def main() -> None:
    app()
