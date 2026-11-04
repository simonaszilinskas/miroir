"""`miroir geo`: positions of entities and of query photos (step H2)."""

from __future__ import annotations

import typer

from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Step H2: where buildings, objects and query photos are.",
)


@app.command("places")
def geo_places() -> None:
    """One position per entity: Mérimée or Muséofile point, else the commune centre."""
    from miroir.geo.places import build_places  # noqa: PLC0415

    typer.echo("\n".join(build_places(get_settings()).lines()))


@app.command("positions")
def geo_positions() -> None:
    """Where each Commons query photo was taken: EXIF GPS, else the file page's camera."""
    from miroir.geo.queries import build_positions  # noqa: PLC0415

    for (set_id, source), count in sorted(build_positions(get_settings()).items()):
        typer.echo(f"{set_id:<6}{source:<8}{count:>6}")
