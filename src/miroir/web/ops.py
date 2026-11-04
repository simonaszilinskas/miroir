"""Running the website from the Mac: upload an index, set the password, run locally.

`miroir web upload` copies a built bundle to the index bucket, where the demo machine and
`miroir web dev --scw` read it. `miroir web secret` makes one shared password, keeps it in the
repository's `.env` (`config.ENV_PASSWORD_KEY`, never committed) and never prints it;
`miroir demo install` sends it to the demo machine.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from miroir.config import ENV_FILE, ENV_MCP_TOKENS_KEY, ENV_PASSWORD_KEY, WEB_INDEX, Settings
from miroir.web import auth
from miroir.web.bundle import bundle_files

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from miroir.web.app import SearchBackend

logger = logging.getLogger(__name__)

ENV_PASSWORD = ENV_PASSWORD_KEY


def upload_index(settings: Settings, name: str = WEB_INDEX, client: S3Client | None = None) -> int:
    """Copy a built bundle to the index bucket, under `<name>/`. Returns bytes sent.

    Every file goes, even one the bucket holds at the same size (`levels.json` rewritten)."""
    return _upload(settings, name, bundle_files(settings.web_index_dir(name)), client)


def upload_file(
    settings: Settings, name: str, file_name: str, client: S3Client | None = None
) -> int:
    """Copy one file of a built bundle to the index bucket (e.g. `places.npz` alone)."""
    return _upload(settings, name, [settings.web_index_dir(name) / file_name], client)


def _upload(settings: Settings, name: str, files: list[Path], client: S3Client | None) -> int:
    from miroir.jobs.objects import Bucket, bucket_name, s3_client, upload_files  # noqa: PLC0415

    bucket = bucket_name(settings, Bucket.INDEX)
    root = settings.web_index_dir(name).parent
    upload_files(client or s3_client(settings), bucket, root, files)
    size = sum(p.stat().st_size for p in files)
    logger.info("uploaded %d file(s) (%.1f MB) to %s/%s", len(files), size / 1e6, bucket, name)
    return size


def set_secret(*, rotate: bool = False, env_file: Path = ENV_FILE) -> bool:
    """Make sure `.env` holds the site's password. True when it made a new one.

    `rotate` replaces it. The demo machine gets it with `miroir demo install`.
    """
    if read_env_password(env_file) and not rotate:
        logger.info("password already in %s: kept", env_file)
        return False
    write_env_password(env_file, auth.new_password())
    logger.info("new password written to %s", env_file)
    return True


def read_env_password(env_file: Path) -> str | None:
    return read_env_value(env_file, ENV_PASSWORD)


def write_env_password(env_file: Path, password: str) -> None:
    """Set the password line in `.env`, keeping every other line."""
    write_env_value(env_file, ENV_PASSWORD, password)


def read_env_value(env_file: Path, key: str) -> str | None:
    if not env_file.exists():
        return None
    match = re.search(rf"^{key}=(.*)$", env_file.read_text(encoding="utf-8"), re.M)
    return match.group(1).strip() if match and match.group(1).strip() else None


def write_env_value(env_file: Path, key: str, value: str) -> None:
    """Set one line of `.env`, keeping every other line."""
    text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    lines = [line for line in text.splitlines() if not line.startswith(f"{key}=")]
    lines.append(f"{key}={value}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_mcp_token(name: str, *, rotate: bool = False, env_file: Path = ENV_FILE) -> str:
    """Make sure `.env` holds an MCP connector token for client `name` (`miroir.web.mcp`);
    `rotate` replaces it. Returns the token. `miroir demo install` sends the tokens."""
    from miroir.web.mcp import format_tokens, new_token, parse_tokens  # noqa: PLC0415

    found = parse_tokens(read_env_value(env_file, ENV_MCP_TOKENS_KEY) or "")
    by_name = {n: t for t, n in found.names.items()}
    if name not in by_name or rotate:
        by_name[name] = new_token()
        parse_tokens(format_tokens(by_name))  # checks the name
        write_env_value(env_file, ENV_MCP_TOKENS_KEY, format_tokens(by_name))
        logger.info("MCP token for %s written to %s", name, env_file)
    return by_name[name]


class LedgerSpend:
    """The day's totals of `FileSpend`, and each paid call in the project's spend ledger too
    (experiments.md §1 rule 5), when the local page talks to OpenRouter."""

    def __init__(self, settings: Settings, model: str) -> None:
        from miroir.jobs.spend import SpendLedger  # noqa: PLC0415
        from miroir.web.chat import FileSpend  # noqa: PLC0415

        self._days = FileSpend(settings.chat_spend_path)
        self._ledger = SpendLedger(settings.spend_ledger)
        self._model = model

    async def spent(self, day: str) -> float:
        return await self._days.spent(day)

    async def add(self, day: str, usd: float) -> float:
        from miroir.jobs.spend import SpendKind  # noqa: PLC0415

        if usd > 0:
            self._ledger.record(
                SpendKind.RUN,
                option=CHAT_OPTION,
                run=f"{CHAT_OPTION}-local",
                model=self._model,
                cost_usd=usd,
                items=1,
            )
        return await self._days.add(day, usd)


#: The conversation page's line in the spend ledger.
CHAT_OPTION = "H5-chat"


def serve_locally(
    settings: Settings,
    *,
    name: str,
    port: int,
    fake_llm: bool = False,
    search: SearchBackend | None = None,
) -> None:
    """The site on http://localhost:<port>, with fake models over the local bundle, or with
    the `search` given (models on a Scaleway machine, `miroir web dev --scw`; the page and
    catalog stay local).

    The conversation page uses Mistral Large 4 on Mistral's API when MISTRAL_API_KEY is set,
    else through OpenRouter when OPENROUTER_API_KEY is (with the same caps as the deployed
    site; the day's spending is kept in `Settings.chat_spend_path`), and a scripted stand-in
    otherwise or with `fake_llm`."""
    import uvicorn  # noqa: PLC0415

    from miroir.web.app import create_app  # noqa: PLC0415
    from miroir.web.chat import ChatModel, ChatService, Limits, ThumbSource, chat_model  # noqa: PLC0415
    from miroir.web.chatlog import FileChatLog  # noqa: PLC0415
    from miroir.web.fake import FakeGuide, LocalBackend, fake_engine  # noqa: PLC0415
    from miroir.web.publishable import PUBLISHABLE_FILE, read_publishable  # noqa: PLC0415

    folder = settings.web_index_dir(name)
    bundle_files(folder)
    password = settings.app_password.get_secret_value() if settings.app_password else None
    if not password:
        msg = f"no {ENV_PASSWORD} in .env: run `uv run miroir web secret` first"
        raise RuntimeError(msg)
    engine = fake_engine(folder)
    backend: SearchBackend = LocalBackend(engine) if search is None else search
    model: ChatModel = (None if fake_llm else chat_model(settings)) or FakeGuide()
    publishable = read_publishable(folder / PUBLISHABLE_FILE)

    def chat(fetch: Callable[[str], Awaitable[bytes]]) -> ChatService:
        return ChatService(
            backend=backend,
            model=model,
            thumbs=ThumbSource(publishable, fetch),
            spend=LedgerSpend(settings, model.model),
            limits=Limits(settings.chat_conversation_usd, settings.chat_day_usd),
            index_size=(len(engine.catalog.images), engine.catalog.entity_count),
            catalog=engine.catalog,
            log=FileChatLog(settings.chat_logs_dir),
        )

    site = create_app(
        catalog=engine.catalog,
        backend=backend,
        password=password,
        session_key=secrets.token_bytes(32),
        secure_cookies=False,
        chat=chat,
    )
    if settings.mcp_tokens is not None:
        from miroir.web.mcp import mount_mcp, parse_tokens  # noqa: PLC0415

        tokens = parse_tokens(settings.mcp_tokens.get_secret_value())
        mount_mcp(site, tokens=tokens, publishable=publishable)
    logger.info("conversation page: %s", model.model)
    if search is None:
        logger.info("fake models: photo results are look-alikes, not a real search")
    uvicorn.run(site, host="127.0.0.1", port=port, log_level="info")
