from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from miroir.config import Settings, get_settings
from tests.helpers import FIXTURE_EXPORTS

#: Variables `Settings` reads (or remote clients pick up). Tests start without them.
_OWN_ENV = re.compile(
    r"^(MIROIR_|SCW_|MODAL_)|^(OPENROUTER_API_KEY|MISTRAL_API_KEY|HF_TOKEN|APP_PASSWORD"
    r"|SESSION_SECRET)$"
)


@pytest.fixture(autouse=True)
def _no_developer_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the developer's `.env` and environment out of every test."""
    for name in [n for n in os.environ if _OWN_ENV.search(n)]:  # noqa: TID251 (tests only clear it)
        monkeypatch.delenv(name)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that read the fixture exports and write into a temporary data dir."""
    return Settings(data_dir=tmp_path / "data", exports_dir=FIXTURE_EXPORTS)
