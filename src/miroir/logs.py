"""Logging setup. Library code calls `logging.getLogger(__name__)`; only the CLI calls `setup`."""

from __future__ import annotations

import logging


def setup(*, verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
