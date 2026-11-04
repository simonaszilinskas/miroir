"""`miroir demo`: the demo website on its own Scaleway machine (prototype.md, Hosting)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from miroir.config import DEMO_INDEX, get_settings

if TYPE_CHECKING:
    from miroir.jobs.demo import Demo, DemoMachine
    from miroir.jobs.scw_run import Shell

app = typer.Typer(
    no_args_is_help=True,
    help="The demo website on its own Scaleway GPU machine, powered on only for demos.",
)

DemoIndex = Annotated[str, typer.Option("--index", help="Bundle in the index bucket.")]


def _demo() -> Demo:
    from miroir.jobs.demo import Demo  # noqa: PLC0415
    from miroir.jobs.scaleway import InstanceApi  # noqa: PLC0415

    settings = get_settings()
    return Demo(settings, InstanceApi(settings))


def _need_demo() -> tuple[Demo, DemoMachine]:
    """The demo's handle and its machine; exits when `miroir demo create` has not run."""
    from miroir.jobs.demo import find_demo  # noqa: PLC0415

    demo = _demo()
    machine = find_demo(demo.api)
    if machine is None:
        typer.echo("no demo machine: `uv run miroir demo create` makes it (once)", err=True)
        raise typer.Exit(1)
    return demo, machine


def _need_running() -> tuple[Demo, DemoMachine, Shell]:
    """The demo, its machine and a shell on it; exits unless the machine is running."""
    from miroir.jobs.demo import demo_shell  # noqa: PLC0415

    demo, machine = _need_demo()
    if machine.state != "running" or machine.address is None:
        typer.echo(f"the demo machine is {machine.state}: `uv run miroir demo up`", err=True)
        raise typer.Exit(1)
    return demo, machine, demo_shell(demo.settings, machine.address)


def _month_at_rest() -> str:
    from miroir.jobs.demo import HOURS_PER_MONTH, rest_eur_per_hour  # noqa: PLC0415

    return f"€{rest_eur_per_hour() * HOURS_PER_MONTH:.2f} a month"


@app.command("create")
def demo_create(index: DemoIndex = DEMO_INDEX) -> None:
    """Make the demo machine (once): flexible IP, L4 machine, image, index, HTTPS. 12 min."""
    from miroir.jobs.demo import create_site  # noqa: PLC0415

    machine, waited = create_site(_demo(), index)
    typer.echo(f"site up at {machine.url} ({waited:.0f} s after install)")
    typer.echo(
        f"off between demos it costs {_month_at_rest()}; "
        "`uv run miroir demo down` powers it off now"
    )


@app.command("install")
def demo_install(index: DemoIndex = DEMO_INDEX) -> None:
    """Send this checkout's code to the running machine, rebuild, restart the site."""
    from miroir.jobs.demo import reinstall  # noqa: PLC0415

    demo, machine, shell = _need_running()
    waited = reinstall(demo, machine, shell, index)
    typer.echo(f"site up at {machine.url} ({waited:.0f} s to load)")


@app.command("crops")
def demo_crops(
    index: DemoIndex = "pop-all-crops",
    gb: Annotated[int, typer.Option(help="Size of the volume for the whole crop vectors.")] = 200,
) -> None:
    """Serve a bundle with crops on the running demo machine: a Block Storage volume for its
    whole crop vectors (made, attached and mounted the first time, about €19 a month), then
    `install` with that bundle. Back to the bundle without crops: `demo install`."""
    from miroir.jobs.demo import add_crops, reinstall  # noqa: PLC0415

    demo, machine, shell = _need_running()
    state = add_crops(demo, machine, shell, index, gb)
    typer.echo(f"crop volume {state.crops_volume} ({state.extra_gb} GB) mounted")
    waited = reinstall(demo, machine, shell, index)
    typer.echo(f"site up at {machine.url} with {index} ({waited:.0f} s to load)")


@app.command("up")
def demo_up(
    idle_minutes: Annotated[
        int | None,
        typer.Option(help="Power off after this long with no visitor (default from settings)."),
    ] = None,
    *,
    gpu_only: Annotated[
        bool, typer.Option("--gpu-only", help="Wait for a GPU rather than run on the CPU.")
    ] = False,
    only: Annotated[
        str | None,
        typer.Option("--type", help="Run as this type only, e.g. POP2-HM-12C-96G (a check)."),
    ] = None,
    index: Annotated[
        str | None,
        typer.Option("--index", help="Bundle in the index bucket (default: the one installed)."),
    ] = None,
) -> None:
    """Power the demo machine on, wait until the site answers, print its address.

    When no L4 is free it runs as the next machine type that is (DEMO_OFFERS): an L40S, or a
    machine without a GPU, where searches are slower and captions are not read."""
    from miroir.jobs.demo import bring_up  # noqa: PLC0415

    demo, machine = _need_demo()
    up = bring_up(
        demo, machine, idle_minutes=idle_minutes, allow_cpu=not gpu_only, only=only, index=index
    )
    typer.echo(
        f"{up.machine.url}\nup in {up.total_s:.0f} s ({up.boot_s:.0f} s to boot) as "
        f"{up.offer.commercial_type}{'' if up.gpu else ' (no GPU: slower, no captions)'}, "
        f"€{up.offer.eur_per_hour:.2f} an hour; powers itself off after {up.idle_minutes} min "
        "with no visitor"
    )


@app.command("down")
def demo_down() -> None:
    """Power the demo machine off. Its disk and address stay; only they are billed."""
    from miroir.jobs.demo import rest_eur_per_hour  # noqa: PLC0415

    demo, machine = _need_demo()
    demo.power_off(machine)
    typer.echo(
        f"off; at rest €{rest_eur_per_hour():.3f} an hour ({_month_at_rest()}). "
        f"Spent on Scaleway so far: €{demo.ledger.total_eur():.2f}"
    )


@app.command("status")
def demo_status() -> None:
    """State, address, time on, this session's cost, and the ledger's total."""
    import time  # noqa: PLC0415

    from miroir.jobs.demo import demo_shell, minutes_since_visit  # noqa: PLC0415

    demo, machine = _need_demo()
    state = demo.settle(machine)
    typer.echo(f"{machine.state}  {machine.url}  ({machine.server_id}, {machine.zone})")
    if state.session_start is not None:
        on = (time.time() - state.session_start) / 60
        typer.echo(f"on for {on:.0f} min, €{demo.session_cost(state):.2f} so far")
    if machine.state == "running" and machine.address is not None:
        seen = minutes_since_visit(demo_shell(demo.settings, machine.address))
        if seen is not None:
            typer.echo(f"last visitor {seen:.0f} min ago")
    typer.echo(
        f"at rest {_month_at_rest()}; spent on Scaleway: €{demo.ledger.total_eur():.2f} "
        f"of €{demo.settings.scw_budget_eur:.2f}"
    )


@app.command("logs")
def demo_logs(
    container: Annotated[str, typer.Option(help="miroir-site, miroir-watch or caddy.")] = (
        "miroir-site"
    ),
    lines: Annotated[int, typer.Option(help="Last lines to show.")] = 60,
) -> None:
    """The last lines a container on the running machine wrote (no secret is ever logged)."""
    from miroir.jobs.demo import CONTAINERS  # noqa: PLC0415

    if container not in CONTAINERS:
        raise typer.BadParameter(f"one of {', '.join(CONTAINERS)}")
    _, _, shell = _need_running()
    typer.echo(shell.run(f"docker logs --tail {int(lines)} {container} 2>&1", timeout=60))


@app.command("chat-logs")
def demo_chat_logs() -> None:
    """Copy the conversation page's logs (JSON lines, no images) from the running machine."""
    from miroir.jobs.demo import pull_chat_logs  # noqa: PLC0415

    demo, _, shell = _need_running()
    written, target = pull_chat_logs(demo.settings, shell)
    typer.echo(f"{len(written)} log file(s) new or grown in {target}")


@app.command("destroy")
def demo_destroy(
    *, yes: Annotated[bool, typer.Option("--yes", help="Delete without asking.")] = False
) -> None:
    """Delete the demo machine, its disk and its address (the URL changes after that)."""
    from miroir.jobs.scaleway import delete_machine  # noqa: PLC0415

    demo, machine = _need_demo()
    if not yes and not typer.confirm(f"delete the demo machine, its disk and {machine.address}?"):
        raise typer.Abort
    if machine.state == "running":
        machine = demo.power_off(machine)
    demo.settle(machine)
    delete_machine(demo.api, machine.zone, machine.server_id)
    demo.state_path.unlink(missing_ok=True)
    typer.echo("deleted")


@app.command("serve", hidden=True)
def demo_serve(
    index: Annotated[Path, typer.Option()],
    state: Annotated[Path, typer.Option()],
    port: Annotated[int, typer.Option()],
) -> None:
    """The site, inside the image on the demo machine. Never run on the Mac."""
    from miroir.remote.site import serve_site  # noqa: PLC0415 (machine only)

    serve_site(get_settings(), index=index, state=state, port=port)


@app.command("watch", hidden=True)
def demo_watch(
    server_id: Annotated[str, typer.Option()],
    zone: Annotated[str, typer.Option()],
    idle_minutes: Annotated[int, typer.Option()],
    max_hours: Annotated[float, typer.Option()],
    state: Annotated[Path, typer.Option()],
) -> None:
    """Power the demo machine off when nobody uses it. Runs on the machine."""
    import time  # noqa: PLC0415

    from miroir.jobs.demo import watch_idle  # noqa: PLC0415

    watch_idle(
        get_settings(),
        server_id=server_id,
        zone=zone,
        state=state,
        idle_minutes=idle_minutes,
        max_hours=max_hours,
    )
    while True:  # powering off: Docker would only start this again
        time.sleep(3600)


@app.command("fetch", hidden=True)
def demo_fetch(index: DemoIndex = DEMO_INDEX) -> None:
    """Copy the bundle and the weights from their buckets onto the machine's disk."""
    from miroir.jobs.demo import fetch_bundle  # noqa: PLC0415

    got, weights = fetch_bundle(get_settings(), index)
    typer.echo(f"{index}: {got} file(s) copied; weights: {weights} file(s) copied")
