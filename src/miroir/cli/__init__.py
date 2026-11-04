"""The `miroir` command. Every entry point is a subcommand: `uv run miroir --help`.

One module per command group, each with its own Typer app. A command parses its options,
calls one library function and prints what it returns; the work lives in the library. Heavy
imports stay inside the commands, so `miroir --help` is fast and needs no extra.
"""

from __future__ import annotations

from typing import Annotated

import typer

from miroir import logs
from miroir.cli import (
    demo,
    embed,
    geo,
    index,
    records,
    scw,
    testsets,
    vlm,
    web,
    words,
)
from miroir.cli import eval as evaluation

app = typer.Typer(no_args_is_help=True, add_completion=False)
for _name, _group in (
    ("records", records),
    ("eval", evaluation),
    ("testsets", testsets),
    ("embed", embed),
    ("index", index),
    ("vlm", vlm),
    ("words", words),
    ("web", web),
    ("geo", geo),
    ("scw", scw),
    ("demo", demo),
):
    app.add_typer(_group.app, name=_name)


@app.callback()
def _root(
    *,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """MIROIR: visual search over French heritage records."""
    logs.setup(verbose=verbose)


def main() -> None:
    app()
