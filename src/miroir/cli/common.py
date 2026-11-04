"""What several command groups share: option types, and OpenRouter's count of the key's spend."""

from __future__ import annotations

from typing import Annotated

import typer

from miroir.config import get_settings

MaxHours = Annotated[
    float, typer.Option(help="Longest life of the machine; its worker deletes it then.")
]


def key_usage() -> float | None:
    """What OpenRouter says the project's key has spent (`jobs.openrouter.key_usage_usd`)."""
    from miroir.jobs.openrouter import key_usage_usd  # noqa: PLC0415 (imports httpx)

    return key_usage_usd(get_settings().openrouter_api_key)
