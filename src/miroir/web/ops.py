"""Running the website from the Mac: upload an index, set the password, deploy, run locally.

`miroir web secret` makes one shared password, keeps it in the repository's `.env`
(`config.ENV_PASSWORD_KEY`, never committed) and in the `miroir-app` Modal secret, and never
prints it.
"""

from __future__ import annotations

import logging
import re
import secrets
from collections.abc import Awaitable, Callable
from pathlib import Path

from miroir.config import ENV_FILE, ENV_PASSWORD_KEY, WEB_INDEX, Settings
from miroir.web import auth
from miroir.web.bundle import bundle_files

logger = logging.getLogger(__name__)

ENV_PASSWORD = ENV_PASSWORD_KEY


def upload_index(settings: Settings, name: str = WEB_INDEX) -> int:
    """Copy a built bundle to the index volume, under /<name>. Returns bytes sent."""
    import modal  # noqa: PLC0415 (only this command needs it)

    from miroir.web.modal_app import INDEX_VOLUME  # noqa: PLC0415

    files = bundle_files(settings.web_index_dir(name))
    volume = modal.Volume.from_name(INDEX_VOLUME, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        for path in files:
            batch.put_file(path, f"/{name}/{path.name}")  # pyright: ignore[reportUnknownMemberType]
    size = sum(p.stat().st_size for p in files)
    logger.info("uploaded %d files (%.0f MB) to %s:/%s", len(files), size / 1e6, INDEX_VOLUME, name)
    return size


def upload_file(settings: Settings, name: str, file_name: str) -> int:
    """Copy one file of a built bundle to the index volume (e.g. `places.npz` alone)."""
    import modal  # noqa: PLC0415

    from miroir.web.modal_app import INDEX_VOLUME  # noqa: PLC0415

    path = settings.web_index_dir(name) / file_name
    volume = modal.Volume.from_name(INDEX_VOLUME, create_if_missing=True)
    with volume.batch_upload(force=True) as batch:
        batch.put_file(path, f"/{name}/{file_name}")  # pyright: ignore[reportUnknownMemberType]
    logger.info(
        "uploaded %s (%.1f MB) to %s:/%s", file_name, path.stat().st_size / 1e6, INDEX_VOLUME, name
    )
    return path.stat().st_size


def set_secret(*, rotate: bool = False, env_file: Path = ENV_FILE) -> bool:
    """Make sure `.env` and the Modal secret hold the same password. True when it changed.

    The Modal secret cannot be read back, so a password already in `.env` is trusted to be
    the one in Modal; `rotate` replaces both.
    """
    import modal  # noqa: PLC0415

    from miroir.web.modal_app import APP_SECRET_NAME  # noqa: PLC0415

    existing = read_env_password(env_file)
    names = {s.name for s in modal.Secret.objects.list()}
    if existing and APP_SECRET_NAME in names and not rotate:
        logger.info("password already in .env and in the %s secret: kept", APP_SECRET_NAME)
        return False
    password = auth.new_password()
    modal.Secret.objects.delete(APP_SECRET_NAME, allow_missing=True)
    modal.Secret.objects.create(
        APP_SECRET_NAME, {"APP_PASSWORD": password, "SESSION_SECRET": secrets.token_urlsafe(32)}
    )
    write_env_password(env_file, password)
    logger.info("new password written to %s and to the %s secret", env_file, APP_SECRET_NAME)
    return True


def read_env_password(env_file: Path) -> str | None:
    if not env_file.exists():
        return None
    match = re.search(rf"^{ENV_PASSWORD}=(.*)$", env_file.read_text(encoding="utf-8"), re.M)
    return match.group(1).strip() if match and match.group(1).strip() else None


def write_env_password(env_file: Path, password: str) -> None:
    """Set the password line in `.env`, keeping every other line."""
    text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    lines = [line for line in text.splitlines() if not line.startswith(f"{ENV_PASSWORD}=")]
    lines.append(f"{ENV_PASSWORD}={password}")
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_chat_secret(settings: Settings) -> None:
    """Copy OPENROUTER_API_KEY from `.env` into the Modal secret the conversation page reads.
    The key is never printed or logged."""
    import modal  # noqa: PLC0415

    from miroir.web.modal_app import CHAT_SECRET_NAME  # noqa: PLC0415

    if settings.openrouter_api_key is None:
        msg = "no OPENROUTER_API_KEY in .env or the environment"
        raise RuntimeError(msg)
    modal.Secret.objects.delete(CHAT_SECRET_NAME, allow_missing=True)
    modal.Secret.objects.create(
        CHAT_SECRET_NAME, {"OPENROUTER_API_KEY": settings.openrouter_api_key.get_secret_value()}
    )
    logger.info("OpenRouter key written to the %s Modal secret", CHAT_SECRET_NAME)


def deploy() -> str:
    """Deploy the `miroir-web` app. Returns the site's URL."""
    import modal  # noqa: PLC0415

    from miroir.web.modal_app import app, web  # noqa: PLC0415  # pyright: ignore[reportUnknownVariableType]

    with modal.enable_output():
        app.deploy()
    url = web.get_web_url()
    return url or "(no URL: check `modal app list`)"


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
    settings: Settings, *, name: str, port: int, real: bool = False, fake_llm: bool = False
) -> None:
    """The site on http://localhost:<port>, with fake models over the local bundle, or with
    `real` the deployed GPU search (models on Modal; the page and catalog stay local).

    The conversation page uses Mistral Large 4 through OpenRouter when OPENROUTER_API_KEY is
    set (with the same caps as the deployed site; the day's spending is kept in
    `Settings.chat_spend_path`), and a scripted stand-in otherwise or with `fake_llm`."""
    import uvicorn  # noqa: PLC0415

    from miroir.web.app import SearchBackend, create_app  # noqa: PLC0415
    from miroir.web.chat import (  # noqa: PLC0415
        ChatModel,
        ChatService,
        Limits,
        OpenRouterModel,
        ThumbSource,
    )
    from miroir.web.fake import FakeGuide, LocalBackend, fake_engine  # noqa: PLC0415
    from miroir.web.publishable import PUBLISHABLE_FILE, read_publishable  # noqa: PLC0415

    folder = settings.web_index_dir(name)
    bundle_files(folder)
    password = settings.app_password.get_secret_value() if settings.app_password else None
    if not password:
        msg = f"no {ENV_PASSWORD} in .env: run `uv run miroir web secret` first"
        raise RuntimeError(msg)
    engine = fake_engine(folder)
    backend: SearchBackend = LocalBackend(engine)
    if real:
        from miroir.web.modal_app import ModalBackend  # noqa: PLC0415

        backend = ModalBackend(deployed=True)
    key = None if fake_llm else settings.openrouter_api_key
    model: ChatModel = FakeGuide() if key is None else OpenRouterModel(key)
    publishable = read_publishable(folder / PUBLISHABLE_FILE)

    def chat(fetch: Callable[[str], Awaitable[bytes]]) -> ChatService:
        return ChatService(
            backend=backend,
            model=model,
            thumbs=ThumbSource(publishable, fetch),
            spend=LedgerSpend(settings, model.model),
            limits=Limits(settings.chat_conversation_usd, settings.chat_day_usd),
            index_size=(len(engine.catalog.images), engine.catalog.entity_count),
        )

    site = create_app(
        catalog=engine.catalog,
        backend=backend,
        password=password,
        session_key=secrets.token_bytes(32),
        secure_cookies=False,
        chat=chat,
    )
    logger.info("conversation page: %s", model.model)
    if not real:
        logger.info("fake models: photo results are look-alikes, not a real search")
    uvicorn.run(site, host="127.0.0.1", port=port, log_level="info")
