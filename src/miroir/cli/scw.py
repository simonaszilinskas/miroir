"""`miroir scw`: Scaleway GPU machines, buckets, and the jobs that run on them.

The hidden commands run on a machine, never on the Mac: they are the only place
outside `miroir.remote` that may import it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from miroir.jobs.scaleway import InstanceApi, Leftover

from miroir.cli.common import MaxHours
from miroir.config import get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="Scaleway: GPU machines and Object Storage. A machine never outlives its command.",
)

Shards = Annotated[str, typer.Option(help="Shards this machine runs, e.g. 0,1.")]
ShardCount = Annotated[int, typer.Option("--of", help="Shards in all.")]
AnyGpu = Annotated[
    bool, typer.Option("--any-gpu", help="Also a RENDER-S (P100) when no L4 or L40S is free.")
]


def _leftovers() -> tuple[InstanceApi, list[Leftover]]:
    """Print what is left in every zone used, and return it with the API that found it."""
    from miroir.jobs.scaleway import ZONES, InstanceApi, leftovers  # noqa: PLC0415

    settings = get_settings()
    api = InstanceApi(settings)
    zones = tuple(dict.fromkeys((*ZONES, settings.scw_zone)))
    found = list(leftovers(api, zones))
    typer.echo(f"zones: {', '.join(zones)}")
    for item in found:
        typer.echo(f"  {item.zone}  {item.kind:<15} {item.id}  {item.label}")
    if not found:
        typer.echo("  no machine, loose volume or loose IP")
    return api, found


def _crop_job(name: str, first: int, last: int, max_hours: float, **kw: bool) -> None:
    """`run_crop_job`, which says so and exits when no GPU is in stock."""
    from miroir.jobs.crop_embed import run_crop_job  # noqa: PLC0415
    from miroir.jobs.scaleway import NoMachineError  # noqa: PLC0415

    try:
        done = run_crop_job(get_settings(), name, first, last, max_hours=max_hours, **kw)
    except NoMachineError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from err
    for status in done:
        typer.echo(status.line())


@app.command("buckets")
def scw_buckets() -> None:
    """Create the Object Storage buckets the GPU machines and the website read."""
    from miroir.jobs.objects import ensure_buckets, s3_client  # noqa: PLC0415

    settings = get_settings()
    for name, outcome in ensure_buckets(settings, s3_client(settings)).items():
        typer.echo(f"{name}: {outcome}")


@app.command("ps")
def scw_ps() -> None:
    """List miroir machines, and volumes and IPs attached to no machine, in every zone used."""
    from miroir.jobs.scaleway import MachineLedger  # noqa: PLC0415

    _leftovers()
    spent = MachineLedger(get_settings().scaleway_dir / "spend.jsonl").total_eur()
    typer.echo(f"spent on machines so far: €{spent:.2f}")


@app.command("cleanup")
def scw_cleanup(
    *, yes: Annotated[bool, typer.Option("--yes", help="Delete without asking.")] = False
) -> None:
    """Delete every miroir machine (with its volumes and IP) and every loose volume and IP."""
    from miroir.jobs.scaleway import delete_leftovers  # noqa: PLC0415

    api, found = _leftovers()
    if not found:
        return
    if not yes and not typer.confirm(f"delete these {len(found)}?"):
        raise typer.Abort
    delete_leftovers(api, found)
    typer.echo("after cleanup:")
    _leftovers()


@app.command("up")
def scw_up(max_hours: MaxHours = 1.0) -> None:
    """Start a GPU machine with the worker on it and keep it until Ctrl-C, then delete it."""
    import time  # noqa: PLC0415

    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415

    with open_scaleway(get_settings(), max_hours=max_hours, purpose="up") as calls:
        gpu = calls.gpu
        typer.echo(
            f"{gpu.name}: {gpu.offer.commercial_type} in {gpu.offer.zone} at {gpu.ip}, "
            f"€{gpu.offer.eur_per_hour:.2f}/h. Ctrl-C deletes it."
        )
        while time.time() < gpu.deadline:
            time.sleep(10)


@app.command("smoke")
def scw_smoke(
    images: Annotated[int, typer.Option(help="Images to embed with each model.")] = 32,
    max_hours: MaxHours = 1.0,
) -> None:
    """Embed a few images with A7 and A3 on a GPU machine and match them to the stored vectors."""
    from miroir.jobs.scw_run import open_scaleway  # noqa: PLC0415
    from miroir.jobs.scw_smoke import (  # noqa: PLC0415
        MIN_COSINE,
        run_smoke,
        save_smoke,
        smoke_images,
    )

    settings = get_settings()
    picked = smoke_images(settings, images)
    with open_scaleway(settings, max_hours=max_hours, purpose="smoke") as calls:
        results = run_smoke(calls, settings, picked)
        where = f"{calls.gpu.offer.commercial_type} {calls.gpu.offer.zone}"
    path = save_smoke(settings, where, results)
    for r in results:
        typer.echo(
            f"{r.option}: {r.embedded}/{r.asked} images, cosine min {r.min_cosine:.4f} "
            f"median {r.median_cosine:.4f}, {r.images_per_s:.1f} images/s, "
            f"model load {r.load_s:.0f} s: {'ok' if r.ok else f'FAILED (needs > {MIN_COSINE})'}"
        )
    typer.echo(f"written to {path}")
    if not all(r.ok for r in results):
        raise typer.Exit(1)


@app.command("serve", hidden=True)
def scw_serve(
    port: Annotated[int, typer.Option()],
    server_id: Annotated[str, typer.Option()],
    zone: Annotated[str, typer.Option()],
    deadline: Annotated[
        float, typer.Option(help="Unix time past which the machine deletes itself.")
    ],
) -> None:
    """The worker, inside the job image on a GPU machine. Never run on the Mac."""
    from miroir.remote.worker import serve  # noqa: PLC0415 (machine only)

    serve(get_settings(), port=port, server_id=server_id, zone=zone, deadline=deadline)


@app.command("watch", hidden=True)
def scw_watch(
    server_id: Annotated[str, typer.Option()],
    zone: Annotated[str, typer.Option()],
    deadline: Annotated[float, typer.Option(help="Unix time past which the machine is deleted.")],
) -> None:
    """Delete this machine at its deadline. Runs on a Scaleway machine, never on the Mac."""
    from miroir.remote.worker import watch_deadline  # noqa: PLC0415 (machine only)

    watch_deadline(get_settings(), server_id, zone, deadline)


@app.command("download", hidden=True)
def scw_download(
    shard: Annotated[int, typer.Option()],
    of: ShardCount,
    workers: Annotated[int, typer.Option(help="Parallel downloads.")] = 16,
) -> None:
    """Store one shard of the POP images in the images bucket. Runs on a Scaleway machine."""
    from miroir.jobs.pop_download import run_download  # noqa: PLC0415

    run_download(get_settings(), shard=shard, of=of, workers=workers)


@app.command("pop-download")
def scw_pop_download(
    shards: Shards,
    of: ShardCount,
    workers: Annotated[int, typer.Option(help="Parallel downloads per shard.")] = 16,
    max_hours: MaxHours = 8.0,
    *,
    upload_list: Annotated[
        bool, typer.Option(help="Write the image list to the bucket first.")
    ] = False,
) -> None:
    """Download POP images to the images bucket on a machine without a GPU (resumable)."""
    from miroir.jobs.scaleway import CPU_OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import (  # noqa: PLC0415
        download_launch,
        july_ids,
        run_unattended,
        upload_image_list,
    )

    settings = get_settings()
    if upload_list:
        typer.echo(f"image list: {upload_image_list(settings, july_ids(settings)):,} files")
    picked = [int(k) for k in shards.split(",")]
    launch = download_launch(settings, picked, of, workers)
    done = run_unattended(
        settings, launch, offers=CPU_OFFERS, max_hours=max_hours, purpose="download"
    )
    for status in done:
        typer.echo(status.line())


@app.command("embed", hidden=True)
def scw_embed(
    first: Annotated[int, typer.Option()],
    last: Annotated[int, typer.Option()],
    *,
    a7_batch: Annotated[int, typer.Option()] = 32,
    a7_streams: Annotated[int, typer.Option()] = 1,
    a3_batch: Annotated[int, typer.Option()] = 64,
    check: Annotated[int, typer.Option()] = 0,
    a7_trials: Annotated[str, typer.Option()] = "",
) -> None:
    """Embed pieces of the stored POP images with A7 and A3. Runs in the job image on a GPU."""
    from miroir.jobs.bulk import parse_trials  # noqa: PLC0415
    from miroir.remote.bulk_embed import run_embed  # noqa: PLC0415 (machine only)

    run_embed(
        get_settings(),
        first=first,
        last=last,
        a7_batch=a7_batch,
        a7_streams=a7_streams,
        a3_batch=a3_batch,
        check=check,
        a7_trials=parse_trials(a7_trials),
    )


@app.command("pop-embed")
def scw_pop_embed(
    first: Annotated[int, typer.Option(help="First piece (0 to 4095).")],
    last: Annotated[int, typer.Option(help="Last piece, included.")],
    *,
    a7_batch: Annotated[int, typer.Option(help="A7 images per batch.")] = 32,
    a7_streams: Annotated[int, typer.Option(help="A7 batches run at once.")] = 1,
    a3_batch: Annotated[int, typer.Option(help="A3 images per batch.")] = 64,
    check: Annotated[int, typer.Option(help="July files embedded again to compare.")] = 0,
    a7_trials: Annotated[
        str, typer.Option(help='A7 "batch:streams" to time on the --check files, e.g. 32:1,64:2.')
    ] = "",
    max_hours: MaxHours = 12.0,
    upload_list: Annotated[
        bool, typer.Option(help="Write the list of files July embedded to the bucket first.")
    ] = False,
    l40s: Annotated[bool, typer.Option("--l40s", help="An L40S instead of an L4.")] = False,
) -> None:
    """Embed stored POP images with A7 and A3 on a GPU machine, unattended (resumable)."""
    from miroir.commons.model import is_commons_image  # noqa: PLC0415
    from miroir.jobs.scaleway import OFFERS  # noqa: PLC0415
    from miroir.jobs.scw_bulk import (  # noqa: PLC0415
        compare_check,
        embed_launch,
        july_ids,
        run_unattended,
        upload_july_list,
    )

    settings = get_settings()
    if upload_list:
        files = frozenset(i for i in july_ids(settings) if not is_commons_image(i))
        typer.echo(f"July files: {upload_july_list(settings, files):,}")
    launch = embed_launch(
        settings,
        first,
        last,
        a7_batch=a7_batch,
        a7_streams=a7_streams,
        a3_batch=a3_batch,
        check=check,
        a7_trials=a7_trials,
    )
    offers = [o for o in OFFERS if o.zone == "fr-par-2" and ("L40S" in o.commercial_type) == l40s]
    done = run_unattended(settings, launch, offers=offers, max_hours=max_hours, purpose="embed")
    for status in done:
        typer.echo(status.line())
    if check:
        for option_id, (count, low, median) in compare_check(settings, launch.job).items():
            typer.echo(
                f"{option_id}: {count} July files again, cosine min {low:.4f}, median {median:.4f}"
            )


@app.command("crop-slice-run", hidden=True)
def scw_crop_slice_run() -> None:
    """The crop slice test, in the job image on a GPU machine. Never run on the Mac."""
    from miroir.remote.crop_slice import run_crop_slice  # noqa: PLC0415 (machine only)

    run_crop_slice(get_settings())


@app.command("crop-slice")
def scw_crop_slice(
    max_hours: MaxHours = 3.4,
    *,
    l4: Annotated[bool, typer.Option("--l4", help="An L4 when no L40S is free.")] = False,
    a7_others: Annotated[
        int, typer.Option(help="Files of the sample of others that get EmbeddingGemma 2 crops.")
    ] = 3000,
    render: Annotated[
        bool, typer.Option("--render", help="A RENDER-S (P100), MegaLoc crops only.")
    ] = False,
    gallery: Annotated[int, typer.Option(help="Gallery images.")] = 50_000,
    t2_queries: Annotated[int, typer.Option(help="T2 queries kept (0: all).")] = 0,
) -> None:
    """Crop slice test (docs/search-levels.md): crops of a 50,000-image gallery on an L40S,
    scored against VET, PARTS, T2 and T3; the score tables land in the Scaleway folder."""
    from miroir.jobs.crop_slice import run_slice  # noqa: PLC0415

    got = run_slice(
        get_settings(),
        max_hours=max_hours,
        l4=l4,
        render=render,
        a7_others=a7_others,
        size=gallery,
        t2_queries=t2_queries,
    )
    for path in got:
        typer.echo(f"got {path}")


@app.command("crop-embed-run", hidden=True)
def scw_crop_embed_run(
    name: Annotated[str, typer.Option(help="The crop job (its spec is in the images bucket).")],
    first: Annotated[int, typer.Option(help="First piece.")],
    last: Annotated[int, typer.Option(help="Last piece, included.")],
    *,
    reverse: Annotated[bool, typer.Option("--reverse", help="Last piece first.")] = False,
) -> None:
    """Crop vectors for pieces of a crop job, in the job image on a GPU machine. Never run on
    the Mac."""
    from miroir.remote.crop_embed import run_crop_embed  # noqa: PLC0415 (machine only)

    run_crop_embed(get_settings(), name=name, first=first, last=last, reverse=reverse)


@app.command("crop-short-slice")
def scw_crop_short_slice(
    max_hours: MaxHours = 2.0,
    *,
    any_gpu: AnyGpu = False,
) -> None:
    """Keep the 3x3 crop vectors of the crop slice (float32 and bfloat16, whole) and the slice
    queries' MegaLoc vectors, for `miroir eval crop-short`. Lands in the Scaleway folder."""
    from miroir.jobs.crop_embed import SLICE_NAME, crops_dir, run_slice_crops  # noqa: PLC0415
    from miroir.jobs.scaleway import NoMachineError  # noqa: PLC0415

    settings = get_settings()
    try:
        spec = run_slice_crops(settings, max_hours=max_hours, any_gpu=any_gpu)
    except NoMachineError as err:
        typer.echo(str(err), err=True)
        raise typer.Exit(1) from err
    typer.echo(f"{spec.note}: files in {crops_dir(settings, SLICE_NAME)}")


@app.command("crop-embed")
def scw_crop_embed(
    first: Annotated[int, typer.Option(help="First piece (0 to 4095).")],
    last: Annotated[int, typer.Option(help="Last piece, included.")],
    max_hours: MaxHours = 12.0,
    *,
    any_gpu: AnyGpu = False,
    reverse: Annotated[bool, typer.Option("--reverse", help="Last piece first.")] = False,
) -> None:
    """3x3 crop vectors of every stored POP file, pieces `first` to `last`, on one GPU machine
    (resumable: pieces already written are skipped; `--reverse` starts from the last, to share
    a range with a machine already on it). The spec comes from `crop-embed-spec`."""
    from miroir.jobs.crop_embed import FULL_NAME  # noqa: PLC0415

    _crop_job(FULL_NAME, first, last, max_hours, any_gpu=any_gpu, reverse=reverse)


@app.command("crop-embed-spec")
def scw_crop_embed_spec(
    dims: Annotated[int, typer.Option(help="PCA axes kept per crop (at most 1,024).")] = 256,
    precision: Annotated[
        str, typer.Option(help="MegaLoc on the GPU: fp32, tf32, bf16, fp16.")
    ] = "fp16",
) -> None:
    """Write the spec of the crop run over all of POP: 3x3 crops, projected on the first
    `dims` axes of the `pop-all` MegaLoc PCA (copied beside the spec)."""
    from miroir.jobs.crop_embed import Precision, write_full_spec  # noqa: PLC0415

    spec = write_full_spec(get_settings(), Precision(precision), dims)
    typer.echo(f"{spec.note}: {len(spec.pieces())} pieces")


@app.command("crop-bundle")
def scw_crop_bundle(
    dims: Annotated[int, typer.Option(help="PCA axes per crop to pick crops with.")] = 128,
    rescore: Annotated[int, typer.Option(help="Crops picked and scored whole per search.")] = 5_000,
    shortlist: Annotated[
        int, typer.Option(help="Image rows by best crop on the shortlist.")
    ] = 2000,
    weight: Annotated[float, typer.Option(help="An image scores max(whole, weight * crop).")] = 1.0,
) -> None:
    """Build `pop-all-crops` in the index bucket: `pop-all` copied there, plus the crop files
    made from the crop run's pieces (downloaded to the Scaleway folder first). No model."""
    from miroir.index.crop_bundle import build_crop_bundle  # noqa: PLC0415
    from miroir.web.crops import CropRule  # noqa: PLC0415

    rule = CropRule(rescore=rescore, shortlist=shortlist, weight=weight)
    typer.echo(json.dumps(build_crop_bundle(get_settings(), dims=dims, rule=rule)))


@app.command("crop-measure-run", hidden=True)
def scw_crop_measure_run(index: Annotated[str, typer.Option(help="Bundle to measure.")]) -> None:
    """Search the measure's queries crops off and on. Runs on a GPU machine, never the Mac."""
    from miroir.remote.crop_measure import run_crop_measure  # noqa: PLC0415 (machine only)

    run_crop_measure(get_settings(), index)


@app.command("crop-measure")
def scw_crop_measure(
    index: Annotated[str, typer.Option(help="Bundle with crops.")] = "pop-all-crops",
    max_hours: MaxHours = 2.0,
) -> None:
    """T1, T2, T2b, T3, T5, PARTS and VET (tune halves) searched on a GPU machine with the
    site's code, crops off and on; hit rates and times go to eval/results/R3."""
    from miroir.jobs.crop_measure import run_measure  # noqa: PLC0415

    path = run_measure(get_settings(), index, max_hours=max_hours)
    typer.echo(json.dumps(json.loads(path.read_text())["sets"], indent=1))
    typer.echo(f"written to {path}")


@app.command("build-pop-index", hidden=True)
def scw_build_pop_index(
    work: Annotated[Path, typer.Option(help="Scratch folder for the gathered vectors.")],
) -> None:
    """Gather the vectors and build the `pop-all` bundle. Runs on a Scaleway machine."""
    from miroir.index.pop_all import build_on_machine  # noqa: PLC0415

    build_on_machine(get_settings(), work)


@app.command("pop-index")
def scw_pop_index(max_hours: MaxHours = 6.0) -> None:
    """Build the `pop-all` index bundle on a machine with much memory; it lands in the index
    bucket. The vectors must be in the images bucket (`pop-download`, `pop-embed`)."""
    from miroir.index.pop_all import run_pop_index  # noqa: PLC0415

    for status in run_pop_index(get_settings(), max_hours=max_hours):
        typer.echo(status.line())


@app.command("pop-coverage")
def scw_pop_coverage() -> None:
    """Which records of the `pop-all` index have vectors, and why the others do not."""
    from miroir.index.coverage import write_report  # noqa: PLC0415
    from miroir.jobs.objects import s3_client  # noqa: PLC0415

    settings = get_settings()
    out = settings.index_dir / "coverage-pop-all.json"
    tables = settings.scaleway_dir / "pop-all-tables"
    report = write_report(settings, s3_client(settings), tables, out)
    typer.echo(f"files with vectors: {report['files_with_vectors']:,} of {report['files']:,}")
    typer.echo(f"written to {out}")
