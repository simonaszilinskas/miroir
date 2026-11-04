"""`miroir testsets`: build the test sets and galleries (technical spec §7)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Test sets T1, T2, T2b, T3, T5, T8, VET, PARTS and galleries.",
)


@app.command("build")
def testsets_build() -> None:
    """Build the test sets and galleries from records.parquet (technical spec §7)."""
    from miroir.eval.build_sets import build  # noqa: PLC0415

    typer.echo("\n".join(build(get_settings()).lines()))


@app.command("commons")
def build_commons_sets(
    sample_out: Annotated[
        Path | None,
        typer.Option(
            "--check-sample",
            help="Also write 100 tune T2/T2b queries to this JSON file, to check by eye.",
        ),
    ] = None,
) -> None:
    """Find Commons photos for T2, T2b and T8 (cached), then rebuild every set and gallery."""
    from miroir.commons.build import fetch_commons  # noqa: PLC0415 (imports httpx, Pillow)
    from miroir.commons.check import write_check_sample  # noqa: PLC0415
    from miroir.eval.build_sets import build  # noqa: PLC0415

    settings = get_settings()
    report = fetch_commons(settings)
    for s in report.sets:
        typer.echo(f"{s.set_id} {s.kind}: {s.photos} photos of {s.entities} entities")
    typer.echo("\n".join(build(settings).lines()))
    if sample_out is not None:
        count = write_check_sample(settings, sample_out)
        typer.echo(f"{sample_out}: {count} queries to check")


@app.command("commons-gallery")
def build_commons_gallery(
    gallery: Annotated[str, typer.Option(help="Gallery whose Mérimée buildings get photos.")],
) -> None:
    """F2: add up to 5 Commons photos per building of a gallery (cached; query photos kept out).

    Writes commons-gallery-<gallery>.parquet and the gallery list <gallery>+F2.
    """
    from miroir.commons.gallery import (  # noqa: PLC0415 (imports httpx, Pillow)
        fetch_gallery_photos,
        gallery_buildings,
    )
    from miroir.eval.sets import read_gallery  # noqa: PLC0415
    from miroir.records.store import read_images  # noqa: PLC0415

    settings = get_settings()
    report = fetch_gallery_photos(settings, gallery)
    typer.echo(
        f"{gallery}: {report.photos} photos for {report.with_photos} of {report.buildings} "
        f"buildings; {report.requests} requests sent; left out: {dict(report.rejects)}"
    )
    full = gallery_buildings(read_images(settings.images_parquet), read_gallery(settings, "full"))
    requests, hours = report.estimate(len(full))
    typer.echo(
        f"full sample: {len(full)} buildings, about {requests} requests, "
        f"{hours:.1f} h at Wikimedia's pace with nothing cached"
    )


@app.command("vet")
def testsets_vet(
    images: Annotated[
        list[Path],
        typer.Argument(
            help="The three Vétheuil photos, in order: catalogue page (Mise au tombeau), the "
            "altarpiece today, catalogue page (Descente de croix).",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """VET: three photos of the Vétheuil altarpiece from outside POP (tune half only)."""
    from miroir.eval.parts import build_vet  # noqa: PLC0415

    settings = get_settings()
    for q in build_vet(settings, images):
        typer.echo(f"{q.query_id}: {settings.outside_image_path(q.query_image_id)}")


@app.command("parts")
def testsets_parts() -> None:
    """PARTS: a detail photo as query, its whole as answer, mined from the full POP tables."""
    from miroir.eval.parts import build_parts  # noqa: PLC0415

    typer.echo("\n".join(build_parts(get_settings()).lines()))


@app.command("fetched")
def testsets_fetched(
    *,
    gallery: Annotated[str, typer.Option(help="Gallery list, e.g. screening-v2-publishable.")],
    vectors: Annotated[list[str], typer.Option(help="Vector run to compare; repeat it.")],
) -> None:
    """Write <gallery>-fetched: its images that every given run holds a vector for."""
    from miroir.eval.run import write_fetched_gallery  # noqa: PLC0415
    from miroir.eval.sets import read_gallery  # noqa: PLC0415

    settings = get_settings()
    ids = write_fetched_gallery(settings, gallery, vectors)
    total = len(read_gallery(settings, gallery))
    typer.echo(f"{gallery}-fetched: {len(ids)} of {total} images")
