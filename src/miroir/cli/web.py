"""`miroir web`: the website's bundle, its secrets, and the site on this Mac."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.config import ENV_MCP_TOKENS_KEY, ENV_PASSWORD_KEY, WEB_INDEX, get_settings

app = typer.Typer(
    no_args_is_help=True,
    help="The prototype website: build its index, upload it, run it locally.",
)

WebIndex = Annotated[str, typer.Option("--index", help="Index bundle name (config.WEB_INDEX).")]


@app.command("build-index")
def web_build_index(index: WebIndex = WEB_INDEX) -> None:
    """Build the index bundle on the Mac: vectors, catalog, type guess, weights, cut-offs."""
    from miroir.web.bundle import build_index  # noqa: PLC0415

    folder = build_index(get_settings(), index)
    typer.echo(f"bundle written to {folder}; see report.json, then `miroir web upload`")


@app.command("upload")
def web_upload(index: WebIndex = WEB_INDEX) -> None:
    """Copy the bundle to the index bucket, where the demo machine and `web dev --scw` read it."""
    from miroir.web.ops import upload_index  # noqa: PLC0415

    size = upload_index(get_settings(), index)
    typer.echo(f"{index}: {size / 1e6:.0f} MB sent to the index bucket")


@app.command("places")
def web_places(
    index: WebIndex = WEB_INDEX,
    *,
    upload: Annotated[
        bool, typer.Option(help="Also copy places.npz alone to the index bucket.")
    ] = False,
) -> None:
    """Add entity positions (`miroir geo places`) to a built bundle, without rebuilding it."""
    from miroir.web.bundle import add_places  # noqa: PLC0415
    from miroir.web.nearby import PLACES_FILE  # noqa: PLC0415

    settings = get_settings()
    count = add_places(settings, settings.web_index_dir(index))
    typer.echo(f"{index}: {count:,} entities placed")
    if upload:
        from miroir.web.ops import upload_file  # noqa: PLC0415

        upload_file(settings, index, PLACES_FILE)
        typer.echo(f"{PLACES_FILE} uploaded; `miroir demo install` makes the site read it")


@app.command("publishable")
def web_publishable(
    index: WebIndex = WEB_INDEX,
    *,
    upload: Annotated[
        bool, typer.Option(help="Also copy publishable.txt.gz alone to the index bucket.")
    ] = False,
) -> None:
    """Mark which images of a built bundle may go to an outside model (the conversation page)."""
    from miroir.web.bundle import add_publishable  # noqa: PLC0415
    from miroir.web.publishable import PUBLISHABLE_FILE  # noqa: PLC0415

    settings = get_settings()
    count = add_publishable(settings, settings.web_index_dir(index))
    typer.echo(f"{index}: {count:,} images may go to an outside model")
    if upload:
        from miroir.web.ops import upload_file  # noqa: PLC0415

        upload_file(settings, index, PUBLISHABLE_FILE)
        typer.echo(f"{PUBLISHABLE_FILE} uploaded; `miroir demo install` makes the site read it")


@app.command("secret")
def web_secret(
    *,
    rotate: Annotated[bool, typer.Option(help="Replace the password.")] = False,
) -> None:
    """Make the shared password, kept in .env (`ENV_PASSWORD_KEY`) only."""
    from miroir.web.ops import set_secret  # noqa: PLC0415

    if set_secret(rotate=rotate):
        typer.echo(
            f"new password in .env ({ENV_PASSWORD_KEY}); `miroir demo install` sends it to the site"
        )
    else:
        typer.echo("password unchanged")


@app.command("mcp-token")
def web_mcp_token(
    name: Annotated[str, typer.Argument(help="Who gets it: chatgpt, claude, a person's name.")],
    *,
    rotate: Annotated[bool, typer.Option(help="Replace this name's token.")] = False,
    show: Annotated[bool, typer.Option(help="Print the connector's path with its token.")] = False,
) -> None:
    """Make an MCP connector token (ChatGPT, Claude), kept in .env (`ENV_MCP_TOKENS_KEY`)."""
    from miroir.web.mcp import MCP_PATH  # noqa: PLC0415
    from miroir.web.ops import set_mcp_token  # noqa: PLC0415

    token = set_mcp_token(name, rotate=rotate)
    typer.echo(f"token for {name} in .env ({ENV_MCP_TOKENS_KEY}); `miroir demo install` sends it")
    if show:
        typer.echo(f"connector URL: <site address>{MCP_PATH}/{token}")


@app.command("dev")
def web_dev(
    index: WebIndex = WEB_INDEX,
    port: Annotated[int, typer.Option(help="Local port.")] = 8000,
    *,
    fake_llm: Annotated[
        bool,
        typer.Option(help="The conversation page uses a scripted stand-in, even with a key."),
    ] = False,
    scw: Annotated[
        bool,
        typer.Option(
            help="Search on a Scaleway GPU started for this session, deleted when it stops "
            "(index from `miroir web upload`)."
        ),
    ] = False,
    max_hours: Annotated[float, typer.Option(help="With --scw: the machine's longest life.")] = 1.0,
) -> None:
    """Run the website on this Mac with fake models (no model is loaded here).

    The conversation page (/chat) calls Mistral Large 4 on Mistral's API when MISTRAL_API_KEY
    is set, else through OpenRouter, within the same spending caps as the demo site."""
    from miroir.web.ops import serve_locally  # noqa: PLC0415

    settings = get_settings()
    if not scw:
        serve_locally(settings, name=index, port=port, fake_llm=fake_llm)
        return
    from miroir.jobs.scw_run import ScalewayBackend, open_scaleway  # noqa: PLC0415

    with open_scaleway(settings, max_hours=max_hours, purpose=f"web dev {index}") as calls:
        search = ScalewayBackend(calls, index)
        serve_locally(settings, name=index, port=port, fake_llm=fake_llm, search=search)


@app.command("sure")
def web_sure(round_id: Annotated[str, typer.Option("--round")] = "R3") -> None:
    """Set when an object search counts as sure (margin rule) on the tune halves."""
    from miroir.eval.sure import search_sets, write_sure_rule  # noqa: PLC0415
    from miroir.web.measure import SiteBundle  # noqa: PLC0415

    settings = get_settings()
    bundle = SiteBundle.load(settings.web_index_dir(WEB_INDEX))
    chosen = write_sure_rule(settings, search_sets(settings, bundle), round_id=round_id)
    typer.echo(
        f"margin {chosen.margin}: {chosen.all_sure:.0%} sure, {chosen.all_right:.0%} right "
        f"({', '.join(f'{k} {v:.0%}' for k, v in chosen.right.items())}); T8 {chosen.t8_sure:.0%}"
    )


@app.command("levels")
def web_levels(round_id: Annotated[str, typer.Option("--round")] = "R3") -> None:
    """Set the level each result card shows ("Très proche", "Proche", "Ressemblance") on the
    tune halves, from stored query vectors (NumPy on the Mac, about 3 minutes the first time)."""
    from miroir.eval.levels import (  # noqa: PLC0415
        not_found_shares,
        rank_tune_sets,
        write_level_rules,
    )

    settings = get_settings()
    rows = rank_tune_sets(settings)
    reports = write_level_rules(settings, rows, round_id=round_id)
    for report in reports:
        rule = report.rule
        typer.echo(
            f"{report.route}: très proche score >= {rule.very_close_score}, gap >= "
            f"{rule.very_close_gap}; proche score >= {rule.close_score}, gap >= {rule.close_gap}"
        )
        for level, c in report.levels.items():
            typer.echo(f"  {level}: {c.searches:.1%} of searches, {c.right:.1%} right")
    for set_id, share in not_found_shares(rows, reports).items():
        typer.echo(f"{set_id} (record not in the index): 'Très proche' on {share:.1%}")
    typer.echo("levels.json written into the bundle; `miroir web upload` sends it.")
