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

#: Repository root (holds `eval/results`, which is committed).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Base URL of the POP asset bucket. Export image paths are relative to it.
POP_ASSET_HOST = "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net"

#: IIIF image service (v3, level 2) serving the same files as the asset bucket.
#: A file's service id is this host plus its bucket path with every "/" encoded.
POP_IIIF_IMAGE_HOST = "https://iiif.prd.cloud.culture.fr/iiif/3"

#: OpenRouter REST API.
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

#: Base URL of a record page on POP.
POP_NOTICE_URL = "https://pop.culture.gouv.fr/notice"

#: Wikidata's SPARQL endpoint and the Wikimedia Commons API, read to build test sets T2, T2b
#: and the Commons part of T8 (technical spec §7).
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
#: Wikimedia asks every client to name itself and give a contact.
WIKIMEDIA_USER_AGENT = (
    "MIROIR/0.1 (https://github.com/simonaszilinskas/miroir) test-set builder, python-httpx"
)

# Paths inside Modal containers (miroir.jobs.modal_app). They exist only there.
#: Volume holding model weights (Hugging Face cache), shared by every run.
MODAL_WEIGHTS_MOUNT = "/weights"
#: Volume holding public benchmarks (AmsterTime), downloaded inside Modal.
MODAL_BENCHMARKS_MOUNT = "/benchmarks"
#: Volume holding every screening image, resized to 512 px, filled once (`miroir embed
#: modal-images`) so model runs read images there instead of from POP.
MODAL_IMAGES_MOUNT = "/images"

#: Where the Modal client keeps its token after `modal token new`.
_MODAL_TOML = Path.home() / ".modal.toml"

_EXPORT_FOLDER: dict[Collection, str] = {
    Collection.JOCONDE: "joconde",
    Collection.MEMOIRE: "memoire",
    Collection.ROSE_VALLAND: "rose-valand",  # sic: the export folder drops an "l"
}

_ROUND_RE = re.compile(r"^R\d+$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")

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
    # Modal: these two, or a profile in ~/.modal.toml (`modal token new`).
    modal_token_id: SecretStr | None = Field(default=None, validation_alias="MODAL_TOKEN_ID")
    modal_token_secret: SecretStr | None = Field(
        default=None, validation_alias="MODAL_TOKEN_SECRET"
    )
    modal_toml: Path = _MODAL_TOML

    # Money cap for all paid model work together (OpenRouter and Modal), in dollars
    # (experiments.md §1 rule 5). The names predate Modal.
    # A run may not push total spend past budget minus reserve. Probes have their own cap.
    openrouter_budget_usd: float = 50.0
    openrouter_reserve_usd: float = 5.0
    probe_budget_usd: float = 2.0

    #: Cap on the resized-image cache, in megabytes. Oldest files go first.
    image_cache_mb: int = 1024

    @property
    def modal_configured(self) -> bool:
        """True when Modal can authenticate: both token variables, or ~/.modal.toml."""
        has_env = self.modal_token_id is not None and self.modal_token_secret is not None
        return has_env or self.modal_toml.expanduser().exists()

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

    # No image store: images are read from their URLs. Only a small, bounded cache of
    # resized copies is kept, for runs that send images to OpenRouter from the Mac.

    @property
    def image_cache_dir(self) -> Path:
        return self.data_root / "cache" / "images"

    @property
    def commons_cache_dir(self) -> Path:
        """Answers from Wikidata and the Commons API, and Commons thumbnails, kept on disk so a
        rebuild of the Commons test sets sends no request. Delete it to fetch afresh."""
        return self.data_root / "cache" / "commons"

    @property
    def commons_parquet(self) -> Path:
        """Commons photos chosen as test queries (T2, T2b, T8), with licence and author."""
        return self.data_root / "commons.parquet"

    @property
    def spend_ledger(self) -> Path:
        """One JSON line per paid call (OpenRouter or Modal): what it cost, for which run.

        The file name predates Modal; it is the one project ledger.
        """
        return self.data_root / "spend" / "openrouter.jsonl"

    def checkpoint_path(self, run: str) -> Path:
        """Vectors a paid run has already received, kept until the run is written."""
        _check_name(run)
        return self.data_root / "checkpoints" / run

    def probe_path(self, option: str) -> Path:
        """Cost measured on 100 images for one option, read before a full run."""
        _check_name(option)
        return self.data_root / "probes" / f"{option}.json"

    def vlm_answers_path(self, model_slug: str) -> Path:
        """A vision model's answers about query photos (step H1), one JSON line per image."""
        _check_name(model_slug)
        return self.data_root / "vlm" / f"{model_slug}.jsonl"

    @property
    def vectors_dir(self) -> Path:
        """Vectors that remote jobs send back, one folder per model run."""
        return self.data_root / "vectors"

    def vectors_path(self, run: str) -> Path:
        _check_name(run)
        return self.vectors_dir / run

    # --- test sets (technical spec §7) ----------------------------------------

    @property
    def testsets_dir(self) -> Path:
        """Tune halves and galleries. Sealed halves live apart, in `testsets-sealed/`."""
        return self.data_root / "testsets"

    def testset_path(self, set_id: str, half: Half, *, allow_sealed: bool = False) -> Path:
        """One half of a test set. The sealed half needs `allow_sealed=True`: R4 code only.

        Read sets with `miroir.eval.sets.load_queries`, not through this path.
        """
        _check_name(set_id)
        if half == "tune":
            return self.testsets_dir / f"{set_id}.tune.jsonl"
        if not allow_sealed:
            msg = f"the sealed half of {set_id} is closed until round R4 (pass allow_sealed=True)"
            raise PermissionError(msg)
        return self.data_root / "testsets-sealed" / f"{set_id}.sealed.jsonl"

    def gallery_path(self, name: str) -> Path:
        """A gallery: one `image_id` per line (`full`, `screening-20k`, ...)."""
        _check_name(name)
        return self.testsets_dir / "galleries" / f"{name}.txt"


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
