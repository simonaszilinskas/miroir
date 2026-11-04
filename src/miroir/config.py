"""Every setting and every filesystem path MIROIR uses.

No other module may read environment variables or build a data path by hand.
Ask for a path here; if the one you need is missing, add it here.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from miroir.records.model import Collection

#: Seed for every random draw (sampling, splits, bootstrap). Pass it explicitly.
SEED = 20261012

#: Repository root (holds `eval/results`, which is committed).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The site's page: its sources (React and the DSFR, built with Vite by `make ui`) and where
#: the build lands, inside the package so the demo image ships it (`miroir.web.pagebuild`).
WEB_UI_SOURCE = REPO_ROOT / "web-ui"
WEB_UI_BUILD = REPO_ROOT / "src" / "miroir" / "web" / "ui"

#: Base URL of the POP asset bucket. Export image paths are relative to it.
POP_ASSET_HOST = "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net"

#: IIIF image service (v3, level 2) serving the same files as the asset bucket.
#: A file's service id is this host plus its bucket path with every "/" encoded.
POP_IIIF_IMAGE_HOST = "https://iiif.prd.cloud.culture.fr/iiif/3"

#: OpenRouter REST API.
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

#: Mistral's own REST API (the conversation page's model).
MISTRAL_API_BASE = "https://api.mistral.ai/v1"

#: Base URL of a record page on POP.
POP_NOTICE_URL = "https://pop.culture.gouv.fr/notice"

#: Wikidata's SPARQL endpoint and the Wikimedia Commons API, read to build test sets T2, T2b
#: and the Commons part of T8 (technical spec §7).
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
#: Commune centres (INSEE codes), from the French State's open geography API (step H2).
GEO_API_URL = "https://geo.api.gouv.fr"
#: Wikimedia asks every client to name itself and give a contact.
WIKIMEDIA_USER_AGENT = (
    "MIROIR/0.1 (https://github.com/simonaszilinskas/miroir) test-set builder, python-httpx"
)

# Folders on a Scaleway GPU machine, inside the job image (`miroir.jobs.scw_run.MOUNTS`). They
# exist only there; each mirrors a bucket (`miroir.jobs.objects`).
#: Model weights (Hugging Face and torch caches), shared by every run.
WEIGHTS_MOUNT = "/weights"
#: POP images resized to 512 px, filled once (`miroir embed images`) so model runs read
#: images there instead of from POP.
IMAGES_MOUNT = "/images"
#: The web index bundles (`miroir web upload`).
WEB_INDEX_MOUNT = "/web-index"

#: The bundle the site's eval steps and `miroir web dev` read on the Mac (a name from
#: `miroir.web.bundle.INDEXES`): the July exports, small enough to keep here.
WEB_INDEX = "full-f2"
#: The bundle the demo machine serves (`miroir demo`): all of POP, 27.4 GB, kept only in the
#: index bucket and on the machine's disk. Switching is this line, then `miroir demo install`.
DEMO_INDEX = "pop-all"

# Money caps (experiments.md §1 rule 5). Raising one is a project decision: change it here, in
# a reviewed commit, with the project lead's yes. Settings may only tighten them.
#: Dollars for all paid model work together: OpenRouter, and Modal before 2026-11-03.
BUDGET_USD = 50.0
#: Dollars kept back: a run may not push total spend past `BUDGET_USD` minus this.
BUDGET_RESERVE_USD = 5.0
#: Dollars for all probes together.
PROBE_BUDGET_USD = 2.0
#: Euros for all Scaleway GPU machines together (their own ledger).
SCW_BUDGET_EUR = 5.0

#: The variable that moves the data folder (`Settings.data_dir`), e.g. on a Scaleway machine.
DATA_DIR_VARIABLE = "MIROIR_DATA_DIR"

#: The `.env` file settings are read from. `miroir web secret` adds the app password to it.
ENV_FILE = REPO_ROOT / ".env"
#: The line of `.env` that holds the website's shared password.
ENV_PASSWORD_KEY = "MIROIR_APP_PASSWORD"  # noqa: S105 (a variable name, not a value)
#: The line of `.env` that holds the MCP connector's tokens (`miroir web mcp-token`).
ENV_MCP_TOKENS_KEY = "MIROIR_MCP_TOKENS"

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

    model_config = SettingsConfigDict(env_prefix="MIROIR_", env_file=ENV_FILE, extra="ignore")

    data_dir: Path = Path("~/miroir-data")
    exports_dir: Path = Path("~/miroir-exports")

    # Model work runs remotely: OpenRouter where the model is there, else a Scaleway GPU
    # machine. These keys keep their usual names, without the MIROIR_ prefix.
    openrouter_api_key: SecretStr | None = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    mistral_api_key: SecretStr | None = Field(default=None, validation_alias="MISTRAL_API_KEY")
    hf_token: SecretStr | None = Field(default=None, validation_alias="HF_TOKEN")
    # Scaleway (the ministry's "Miroir" project): GPU machines and Object Storage. The scw
    # CLI's own names, without the MIROIR_ prefix. The secret key also signs Object Storage.
    scw_access_key: SecretStr | None = Field(default=None, validation_alias="SCW_ACCESS_KEY")
    scw_secret_key: SecretStr | None = Field(default=None, validation_alias="SCW_SECRET_KEY")
    scw_project_id: str | None = Field(default=None, validation_alias="SCW_DEFAULT_PROJECT_ID")
    scw_region: str = Field(default="fr-par", validation_alias="SCW_DEFAULT_REGION")
    scw_zone: str = Field(default="fr-par-2", validation_alias="SCW_DEFAULT_ZONE")
    #: Object Storage bucket names start with this (`<prefix>-weights`, ...). Bucket names are
    #: shared by every Scaleway customer, so another prefix may be needed.
    scw_bucket_prefix: str = "miroir"
    #: Cap, in euros, on all Scaleway GPU machines together (their own ledger,
    #: `scaleway_dir / "spend.jsonl"`), apart from the dollar budget below. At most
    #: `SCW_BUDGET_EUR`.
    scw_budget_eur: float = SCW_BUDGET_EUR
    #: The demo machine (`miroir demo`) powers itself off after this long with no visitor,
    #: and after `demo_max_hours` on, visitors or not.
    demo_idle_minutes: int = 60
    demo_max_hours: float = 8.0

    # Money cap for paid model work, in dollars: at most `BUDGET_USD`, with at least
    # `BUDGET_RESERVE_USD` kept back; probes at most `PROBE_BUDGET_USD`. The environment may
    # lower a cap, never raise it. The OPENROUTER_ names are the old ones.
    budget_usd: float = Field(
        default=BUDGET_USD,
        validation_alias=AliasChoices("MIROIR_BUDGET_USD", "MIROIR_OPENROUTER_BUDGET_USD"),
    )
    budget_reserve_usd: float = Field(
        default=BUDGET_RESERVE_USD,
        validation_alias=AliasChoices("MIROIR_BUDGET_RESERVE_USD", "MIROIR_OPENROUTER_RESERVE_USD"),
    )
    probe_budget_usd: float = PROBE_BUDGET_USD

    # The prototype website (`miroir web`): one shared password, and the key that signs session
    # cookies. On the Mac they come from `.env` (MIROIR_...); on the demo machine from its
    # own env file (APP_PASSWORD, SESSION_SECRET), written by `miroir demo install`.
    app_password: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MIROIR_APP_PASSWORD", "APP_PASSWORD")
    )
    session_secret: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MIROIR_SESSION_SECRET", "SESSION_SECRET")
    )
    #: The MCP connector's secrets (`miroir.web.mcp`), one per client: `name=token,...`, made
    #: by `miroir web mcp-token`; `miroir demo install` sends them as MCP_TOKENS.
    mcp_tokens: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("MIROIR_MCP_TOKENS", "MCP_TOKENS")
    )

    # The conversation page (step H5): a model through OpenRouter leads the search. Spending
    # caps, in dollars, per conversation and per UTC day (all conversations together).
    chat_conversation_usd: float = 0.10
    chat_day_usd: float = 2.0
    #: Cap on the resized-image cache, in megabytes. Oldest files go first.
    image_cache_mb: int = 1024

    @model_validator(mode="after")
    def _caps_only_tighten(self) -> Self:
        caps = (
            ("budget_usd", self.budget_usd, BUDGET_USD),
            ("probe_budget_usd", self.probe_budget_usd, PROBE_BUDGET_USD),
            ("scw_budget_eur", self.scw_budget_eur, SCW_BUDGET_EUR),
        )
        raised = [f"{name} {value} > {cap}" for name, value, cap in caps if value > cap]
        if self.budget_reserve_usd < BUDGET_RESERVE_USD:
            raised.append(f"budget_reserve_usd {self.budget_reserve_usd} < {BUDGET_RESERVE_USD}")
        if raised:
            msg = (
                f"a money cap is set looser than config.py allows ({'; '.join(raised)}). "
                "Raising a cap is a project decision: change it in config.py with the project "
                "lead's yes, not in the environment"
            )
            raise ValueError(msg)
        return self

    @property
    def scaleway_configured(self) -> bool:
        """True when the Scaleway keys and project are set."""
        keys = self.scw_access_key is not None and self.scw_secret_key is not None
        return keys and self.scw_project_id is not None

    def scaleway_worker_env(self) -> dict[str, str]:
        """What the worker on a Scaleway machine reads (`miroir.jobs.scw_run`), as environment
        variables: the bucket prefix and the keys, for Object Storage and for deleting its own
        machine at its deadline. Holds secrets: never log it."""
        values = {
            "SCW_DEFAULT_PROJECT_ID": self.scw_project_id or "",
            "SCW_DEFAULT_REGION": self.scw_region,
            "MIROIR_SCW_BUCKET_PREFIX": self.scw_bucket_prefix,
        }
        secrets = {
            "SCW_ACCESS_KEY": self.scw_access_key,
            "SCW_SECRET_KEY": self.scw_secret_key,
            "HF_TOKEN": self.hf_token,
        }
        for name, secret in secrets.items():
            if secret is not None:
                values[name] = secret.get_secret_value()
        return values

    def demo_site_env(self) -> dict[str, str]:
        """What the website on the demo machine reads (`miroir.jobs.demo`), as environment
        variables: the shared password, the MCP connector's tokens, the conversation page's
        keys (OpenRouter first, `miroir.web.chat.chat_model`) and its spending caps. The
        session key is made on the machine. Holds secrets: never log it."""
        if self.app_password is None:
            msg = f"no {ENV_PASSWORD_KEY} in .env: run `uv run miroir web secret` first"
            raise RuntimeError(msg)
        values = {
            "MIROIR_CHAT_CONVERSATION_USD": str(self.chat_conversation_usd),
            "MIROIR_CHAT_DAY_USD": str(self.chat_day_usd),
        }
        secrets = {
            "APP_PASSWORD": self.app_password,
            "OPENROUTER_API_KEY": self.openrouter_api_key,
            "MISTRAL_API_KEY": self.mistral_api_key,
            "MCP_TOKENS": self.mcp_tokens,
        }
        for name, secret in secrets.items():
            if secret is not None:
                values[name] = secret.get_secret_value()
        return values

    # --- inputs (read only) -------------------------------------------------

    def export_dir(self, collection: Collection) -> Path:
        """Folder of one POP export. Never write here."""
        return self.exports_dir.expanduser() / _EXPORT_FOLDER[collection]

    def export_jsonl(self, collection: Collection) -> Path:
        folder = _EXPORT_FOLDER[collection]
        return self.export_dir(collection) / f"{folder}.jsonl"

    @property
    def full_exports_dir(self) -> Path:
        """The whole POP download (`exports-full/<base>/<base>.jsonl`), with bases the July
        exports lack (Mérimée, Palissy, Muséofile). Read only."""
        return self.data_root / "exports-full"

    def full_export_jsonl(self, base: str) -> Path:
        _check_name(base)
        return self.full_exports_dir / base / f"{base}.jsonl"

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

    @property
    def full_pop_dir(self) -> Path:
        """Tables of the whole POP download (`miroir records build-full`): `records.parquet`,
        `images.parquet`, `report.json`, image lists. Kept apart from the July tables."""
        return self.data_root / "full-pop"

    @property
    def full_records_parquet(self) -> Path:
        return self.full_pop_dir / "records.parquet"

    @property
    def full_images_parquet(self) -> Path:
        return self.full_pop_dir / "images.parquet"

    @property
    def extra_dir(self) -> Path:
        """Records copied from the whole download into the site's index, one entity at a time
        (`miroir records add`). Kept apart from the July tables, which stay as they are."""
        return self.data_root / "extra"

    @property
    def extra_records_parquet(self) -> Path:
        return self.extra_dir / "records.parquet"

    @property
    def extra_images_parquet(self) -> Path:
        return self.extra_dir / "images.parquet"

    @property
    def full_report_json(self) -> Path:
        """Counts of the full build: per base, unique images, buildings and objects."""
        return self.full_pop_dir / "report.json"

    def full_image_list(self, name: str) -> Path:
        """One `image_id` per line, one per distinct image URL (`unique`, `covers`, ...)."""
        _check_name(name)
        return self.full_pop_dir / "lists" / f"{name}.txt"

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

    def commons_gallery_parquet(self, gallery: str) -> Path:
        """Commons photos added as gallery images of the buildings of `gallery` (step F2)."""
        _check_name(gallery)
        return self.data_root / f"commons-gallery-{gallery}.parquet"

    def commons_gallery_parquets(self) -> list[Path]:
        """Every F2 table written so far, in name order."""
        return sorted(self.data_root.glob("commons-gallery-*.parquet"))

    @property
    def geo_cache_dir(self) -> Path:
        """Answers of the geography API (commune centres), kept so a rebuild sends nothing."""
        return self.data_root / "cache" / "geo"

    @property
    def places_parquet(self) -> Path:
        """One position per building or object (step H2): POP coordinates, else the centre of
        its commune."""
        return self.data_root / "places.parquet"

    @property
    def query_positions_parquet(self) -> Path:
        """Where each Commons query photo was taken, when its file says (step H2)."""
        return self.data_root / "query-positions.parquet"

    @property
    def spend_ledger(self) -> Path:
        """One JSON line per paid call (OpenRouter or Scaleway; Modal before 2026-11-03): what
        it cost, for which run.

        The file name is older than the other services; it is the one project ledger.
        """
        return self.data_root / "spend" / "openrouter.jsonl"

    def checkpoint_path(self, run: str) -> Path:
        """Vectors a paid run has already received, kept until the run is written."""
        _check_name(run)
        return self.data_root / "checkpoints" / run

    @property
    def demo_dir(self) -> Path:
        """The demo machine's SSH key and known hosts, and the times its ledger lines cover
        (`miroir demo`)."""
        return self.scaleway_dir / "demo"

    @property
    def scaleway_dir(self) -> Path:
        """Per-machine SSH keys and known hosts (`miroir scw`), the machines' ledger and the
        smoke-test results."""
        return self.data_root / "scaleway"

    def probe_path(self, option: str) -> Path:
        """Cost measured on 100 images for one option, read before a full run."""
        _check_name(option)
        return self.data_root / "probes" / f"{option}.json"

    def vlm_answers_path(self, model_slug: str) -> Path:
        """A vision model's answers about query photos (step H1), one JSON line per image."""
        _check_name(model_slug)
        return self.data_root / "vlm" / f"{model_slug}.jsonl"

    def h4_calls_path(self, model_slug: str) -> Path:
        """Every answer of the agent run offline (steps H4 and H5, `miroir.eval.h4`), one JSON
        line per call, keyed by its request, so a rerun replays them instead of paying again."""
        _check_name(model_slug)
        return self.data_root / "vlm" / "h4" / f"{model_slug}-calls.jsonl"

    def h4_report_dir(self, label: str) -> Path:
        """One offline agent run's traces (one JSON line per conversation) and its report."""
        _check_name(label)
        return self.data_root / "vlm" / "h4" / label

    def matches_path(self, name: str) -> Path:
        """A cache of searches on the site's bundle (`eval.sure`, `eval.levels`). The folder
        name dates from step E1, which wrote keypoint matches there first."""
        _check_name(name)
        return self.data_root / "matches" / f"{name}.jsonl"

    @property
    def index_dir(self) -> Path:
        """What the website needs beside the vectors: PCA matrices and the building-or-object
        guess, as small .npz files (technical spec §7.3)."""
        return self.data_root / "index"

    @property
    def vectors_dir(self) -> Path:
        """Vectors that remote jobs send back, one folder per model run."""
        return self.data_root / "vectors"

    def vectors_path(self, run: str) -> Path:
        _check_name(run)
        return self.vectors_dir / run

    # --- the prototype website ------------------------------------------------
    @property
    def chat_spend_path(self) -> Path:
        """Dollars the conversation page spent per day when run on the Mac (`miroir web dev`)."""
        return self.data_root / "spend" / "chat-days.json"

    @property
    def chat_logs_dir(self) -> Path:
        """The conversation page's logs (`miroir.web.chatlog`): the Mac's own under it, the
        demo site's copied into `hosted/` by `miroir demo chat-logs`."""
        return self.data_root / "chat-logs"

    def web_index_dir(self, name: str) -> Path:
        """One web index bundle, built on the Mac before it goes to the index bucket."""
        _check_name(name)
        return self.data_root / "web" / name

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

    def testset_entities_path(self, set_id: str) -> Path:
        """Every building or object a set uses, both halves mixed, written with the set: what
        a model trained on the gallery (the type guess) leaves out. Holds no query."""
        _check_name(set_id)
        return self.testsets_dir / "entities" / f"{set_id}.txt"

    def outside_image_path(self, image_id: str) -> Path:
        """A query image that is in no POP record, kept as given (e.g. the VET set's photos)."""
        _check_name(image_id)
        return self.testsets_dir / "outside" / f"{image_id}.jpg"

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
