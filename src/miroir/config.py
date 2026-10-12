"""Every setting and every filesystem path MIROIR uses.

No other module may read environment variables or build a data path by hand.
Ask for a path here; if the one you need is missing, add it here.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from miroir.records.model import Collection

#: Seed for every random draw (sampling, splits, bootstrap). Pass it explicitly.
SEED = 20261012

#: The only round allowed to read the sealed half of a test set.
SEALED_ROUND = "R4"

#: Repository root (holds `eval/sets` and `eval/results`, which are committed).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Base URL of the POP asset bucket. Export image paths are relative to it.
POP_ASSET_HOST = "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net"

#: Base URL of a record page on POP.
POP_NOTICE_URL = "https://pop.culture.gouv.fr/notice"

_EXPORT_FOLDER: dict[Collection, str] = {
    Collection.JOCONDE: "joconde",
    Collection.MEMOIRE: "memoire",
    Collection.ROSE_VALLAND: "rose-valand",  # sic: the export folder drops an "l"
}

_ROUND_RE = re.compile(r"^R\d+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

Half = Literal["tune", "sealed"]


class Settings(BaseSettings):
    """Values read from the environment (or `.env`), prefixed `MIROIR_`."""

    model_config = SettingsConfigDict(
        env_prefix="MIROIR_", env_file=REPO_ROOT / ".env", extra="ignore"
    )

    data_dir: Path = Path("~/miroir-data")
    exports_dir: Path = Path("/Users/simonaszilinskas/Downloads/exports")

    # Model work runs remotely: OpenRouter where the model is there, else Hugging Face Jobs.
    # These two keep their usual names, without the MIROIR_ prefix.
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")

    # --- inputs (read only) -------------------------------------------------

    def export_dir(self, collection: Collection) -> Path:
        """Folder of one POP export. Never write here."""
        return self.exports_dir.expanduser() / _EXPORT_FOLDER[collection]

    def export_jsonl(self, collection: Collection) -> Path:
        folder = _EXPORT_FOLDER[collection]
        return self.export_dir(collection) / f"{folder}.jsonl"

    # --- outputs ------------------------------------------------------------

    @property
    def data_root(self) -> Path:
        return self.data_dir.expanduser()

    @property
    def records_parquet(self) -> Path:
        return self.data_root / "records.parquet"

    @property
    def images_parquet(self) -> Path:
        return self.data_root / "images.parquet"

    # No local image store: remote jobs read images from their URLs in images.parquet.

    @property
    def vectors_dir(self) -> Path:
        """Vectors that remote jobs send back, one folder per model run."""
        return self.data_root / "vectors"

    def vectors_path(self, run: str) -> Path:
        _check_name(run)
        return self.vectors_dir / run


def eval_sets_dir() -> Path:
    return REPO_ROOT / "eval" / "sets"


def eval_set_path(set_id: str, half: Half, *, round_id: str) -> Path:
    """Path of one half of a test set. The sealed half opens only in round R4."""
    _check_round(round_id)
    _check_name(set_id)
    if half == "sealed" and round_id != SEALED_ROUND:
        msg = f"the sealed half of {set_id} is closed until {SEALED_ROUND} (asked from {round_id})"
        raise PermissionError(msg)
    return eval_sets_dir() / set_id / f"{half}.jsonl"


def eval_result_path(round_id: str, step: str, option: str) -> Path:
    """Where one experiment run writes its scores: eval/results/<round>/<step>-<option>.json."""
    _check_round(round_id)
    _check_name(step)
    _check_name(option)
    return REPO_ROOT / "eval" / "results" / round_id / f"{step}-{option}.json"


@cache
def get_settings() -> Settings:
    return Settings()


def _check_round(round_id: str) -> None:
    if not _ROUND_RE.match(round_id):
        msg = f"round must look like R0, R1, ...: {round_id!r}"
        raise ValueError(msg)


def _check_name(name: str) -> None:
    if not _NAME_RE.match(name):
        msg = f"not a safe path component: {name!r}"
        raise ValueError(msg)
