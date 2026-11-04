"""`miroir embed`: paid embedding runs, on OpenRouter or a Scaleway GPU."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.cli.common import MaxHours, key_usage
from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Embedding runs on remote models.",
)


@app.command("openrouter")
def embed_openrouter(
    option: Annotated[str, typer.Option(help="Option id from experiments.md: A9, A10a.")],
    gallery: Annotated[
        str | None, typer.Option(help="Gallery, e.g. screening-20k-publishable.")
    ] = None,
    queries: Annotated[
        list[str] | None,
        typer.Option(help="Test set (tune half) whose query images to embed; repeat it."),
    ] = None,
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
) -> None:
    """Embed a gallery or test sets' query images on OpenRouter. Paid: probe first."""
    from miroir.jobs import embed as jobs  # noqa: PLC0415 (keeps `miroir --help` fast)
    from miroir.jobs.factory import open_embedder  # noqa: PLC0415

    if (gallery is None) == (not queries):
        raise typer.BadParameter("give exactly one of --gallery and --queries")
    with open_embedder(get_settings(), option) as embedder:
        if gallery is not None and probe:
            result, size = jobs.probe(embedder, gallery)
            typer.echo(
                f"{option} probe: {result.images} images cost ${result.cost_usd:.4f} "
                f"(${result.usd_per_image * 1000:.3f} per 1,000). "
                f"Projected for {gallery} ({size:,} images): ${result.projected_usd(size):.2f}"
            )
            return
        if probe:
            raise typer.BadParameter("--probe works with --gallery only")
        if gallery is not None:
            done = jobs.embed_gallery(embedder, gallery)
        else:
            done = jobs.embed_queries(embedder, queries or [])
        typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@app.command("f2")
def embed_f2(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[str, typer.Option(help="Gallery whose F2 photos to embed.")],
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """F2: embed the Commons photos added to a gallery's buildings (sent from the Mac)."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.supplied import open_gallery_photos  # noqa: PLC0415
    from miroir.records.store import read_commons  # noqa: PLC0415

    settings = get_settings()
    ids = [p.image_id for p in read_commons(settings.commons_gallery_parquet(gallery))]
    purpose = f"embed {option} F2-{gallery}"
    with (
        open_gallery_photos(settings, gallery) as photos,
        open_scaleway(settings, max_hours=max_hours, purpose=purpose) as calls,
    ):
        runner = GpuRunner(settings, calls, key_usage_usd=key_usage())
        done = runner.embed_supplied(option, f"F2-{gallery}", ids, photos, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@app.command("extra")
def embed_extra(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """Embed the images of the records in extra/ (`miroir records add`), sent from the Mac."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.supplied import open_extra_images  # noqa: PLC0415
    from miroir.records.extra import read_extra_images  # noqa: PLC0415

    settings = get_settings()
    ids = [i.image_id for i in read_extra_images(settings)]
    with (
        open_extra_images(settings) as images,
        open_scaleway(settings, max_hours=max_hours, purpose=f"embed {option} X") as calls,
    ):
        runner = GpuRunner(settings, calls, key_usage_usd=key_usage())
        done = runner.embed_supplied(option, "X", ids, images, max_usd=max_usd)
    typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")


@app.command("images")
def embed_images(
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = "screening-20k-publishable",
    *,
    all_images: Annotated[
        bool,
        typer.Option(
            "--all-images",
            help="Also images of records that are not publishable (full-sample index only).",
        ),
    ] = False,
    max_hours: MaxHours = 2.0,
) -> None:
    """Fetch a gallery's images once into the images bucket, so model runs read them there."""
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    settings = get_settings()
    purpose = f"images {gallery}"
    with open_scaleway(
        settings, max_hours=max_hours, purpose=purpose, all_images=all_images
    ) as calls:
        runner = GpuRunner(settings, calls, key_usage_usd=key_usage())
        counts = runner.fill_images(gallery)
    typer.echo(f"{gallery}: {counts}")


@app.command("scw")
def embed_scw(
    option: Annotated[str, typer.Option(help="Model option id, e.g. A7-embeddinggemma-2-512.")],
    gallery: Annotated[str, typer.Option(help="Gallery list.")] = "screening-20k-publishable",
    queries: Annotated[
        list[str] | None,
        typer.Option(help="Embed these test sets' tune query images instead; repeat it."),
    ] = None,
    *,
    probe: Annotated[
        bool, typer.Option("--probe", help="Embed 100 gallery images; print the projected cost.")
    ] = False,
    max_usd: Annotated[
        float | None, typer.Option(help="Refuse the run if the probe projects more than this.")
    ] = None,
    all_images: Annotated[
        bool,
        typer.Option(
            "--all-images",
            help="Also images of records that are not publishable (full-sample index only).",
        ),
    ] = False,
    reuse: Annotated[
        list[str] | None,
        typer.Option(help="Vector run of the same model and settings to take vectors from."),
    ] = None,
    max_hours: MaxHours = 2.0,
) -> None:
    """Embed a gallery, or test sets' query images, with an open model on a Scaleway GPU.

    Starts a machine, runs, deletes it. Paid: probe first. Commons query photos are fetched on
    the Mac and sent as bytes.
    """
    from miroir.commons.images import open_commons_images  # noqa: PLC0415
    from miroir.jobs.gpu_run import GpuRunner  # noqa: PLC0415
    from miroir.jobs.remote_spec import option as find_option  # noqa: PLC0415
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    find_option(option)  # fail before starting a machine
    if queries and probe:
        raise typer.BadParameter("--probe works with --gallery only")
    settings = get_settings()
    purpose = f"embed {option} {'+'.join(queries) if queries else gallery}"
    with (
        open_commons_images(settings) as commons,
        open_scaleway(
            settings, max_hours=max_hours, purpose=purpose, all_images=all_images
        ) as calls,
    ):
        runner = GpuRunner(settings, calls, key_usage_usd=key_usage(), commons=commons)
        if queries:
            done = runner.embed_queries(option, queries, max_usd=max_usd)
            typer.echo(f"{done.info.gallery}: {len(done.ids)} vectors, ${done.info.cost_usd:.4f}")
            return
        if probe:
            result, size = runner.probe(option, gallery)
            typer.echo(
                f"{option} probe: {result.images} images ({result.per_image_s * 1000:.0f} ms "
                f"each). Projected GPU time for {gallery} ({size:,} images): "
                f"${result.projected_usd(size):.2f}"
            )
            return
        written = runner.embed_gallery(option, gallery, max_usd=max_usd, reuse=reuse or ())
    for done in written:
        typer.echo(f"{done.info.gallery} ({done.info.pooling}): {len(done.ids)} vectors")
