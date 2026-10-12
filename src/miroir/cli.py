"""The `miroir` command. Every entry point is a subcommand here: `uv run miroir --help`."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir import logs
from miroir.config import get_settings

app = typer.Typer(no_args_is_help=True, add_completion=False)
records_app = typer.Typer(no_args_is_help=True, help="POP records: parse the exports.")
app.add_typer(records_app, name="records")


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


def main() -> None:
    app()
