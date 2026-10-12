from __future__ import annotations

from pathlib import Path

import pytest

from miroir.config import Settings
from tests.helpers import FIXTURE_EXPORTS


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that read the fixture exports and write into a temporary data dir."""
    return Settings(data_dir=tmp_path / "data", exports_dir=FIXTURE_EXPORTS)
