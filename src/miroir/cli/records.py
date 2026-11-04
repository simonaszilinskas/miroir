"""`miroir records`: parse the POP exports into the record tables."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="POP records: parse the exports.",
)


@app.command("build")
def records_build() -> None:
    """Parse the exports into records.parquet and images.parquet in the data dir."""
    from miroir.records.build import build  # noqa: PLC0415 (keeps `miroir --help` fast)

    report = build(get_settings())
    mismatches = report.mismatches()
    if mismatches:
        typer.echo(f"{len(mismatches)} count(s) differ from the spec: see warnings.", err=True)


@app.command("add")
def records_add(
    entity: Annotated[
        list[str], typer.Option(help="Entity id to copy, e.g. PM95000723; repeat it.")
    ],
) -> None:
    """Copy entities from the whole download (full-pop/) into extra/, for the site's index."""
    from miroir.records.extra import add_entities  # noqa: PLC0415

    settings = get_settings()
    held = add_entities(settings, entity)
    typer.echo(f"{settings.extra_records_parquet}: {held} records")


@app.command("build-full")
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
