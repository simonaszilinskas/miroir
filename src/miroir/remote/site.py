# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""The website on the demo machine (`miroir demo serve`, inside the job image).

One process holds everything: the page, login, record pages, the conversation page, and the
models over the index bundle on the GPU (`miroir.remote.search.SearchService`). Caddy, in
its own container, takes HTTPS on ports 80 and 443 and passes requests to this process on
the machine's loopback. The models load before the site listens, so `miroir demo up` waits
until the page answers.

Files, all on the machine's disk (`miroir.jobs.demo` mounts them):

- the index bundle, read only;
- `state/`: the conversation page's spending per day (`chat-days.json`), its logs
  (`chat-logs/`, copied to the Mac by `miroir demo chat-logs`) and the last visit
  (`last-visit`, read by the watcher that powers the machine off).

Secrets (the site password, the session key, the model keys) come from the environment, which
Docker reads from a root-only file (`miroir.jobs.demo.site_env`).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from miroir.config import Settings
from miroir.jobs.demo import CHAT_DAYS, CHAT_LOGS, LAST_VISIT

if TYPE_CHECKING:
    from miroir.outlines import Outlines
    from miroir.remote.search import SearchService
    from miroir.web.chat import ChatService
    from miroir.web.engine import SearchRequest, SearchResponse
    from miroir.web.toolsearch import ToolHits, ToolQuery

logger = logging.getLogger(__name__)


class InProcessBackend:
    """The search backend (`miroir.web.app.SearchBackend`) with the models in this process.

    One GPU call at a time: a search, an outline or a conversation's search waits for the one
    before it, so two visitors never load the GPU's memory twice."""

    def __init__(self, service: SearchService) -> None:
        self._service = service
        self._gpu = threading.Lock()

    def _locked[T](self, work: Callable[[], T]) -> T:
        with self._gpu:
            return work()

    async def search(self, request: SearchRequest) -> SearchResponse:
        engine = self._service.engine
        return await asyncio.to_thread(self._locked, lambda: engine.search(request))

    async def warm(self) -> None:
        return None  # the models are loaded before the site listens

    async def detect(self, jpeg: bytes) -> Outlines:
        return await asyncio.to_thread(self._locked, lambda: self._service.detect(jpeg))

    async def tool_search(self, query: ToolQuery) -> ToolHits:
        return await asyncio.to_thread(self._locked, lambda: self._service.tool_search(query))


def serve_site(settings: Settings, *, index: Path, state: Path, port: int) -> None:
    """Load the models over `index`, then serve the site on the loopback `port`."""
    import uvicorn  # noqa: PLC0415

    from miroir.remote.search import SearchService  # noqa: PLC0415
    from miroir.web.app import create_app  # noqa: PLC0415
    from miroir.web.chat import ChatService, FileSpend, Limits, ThumbSource, chat_model  # noqa: PLC0415
    from miroir.web.chatlog import FileChatLog  # noqa: PLC0415
    from miroir.web.publishable import PUBLISHABLE_FILE, read_publishable  # noqa: PLC0415
    from miroir.web.visits import VisitMark  # noqa: PLC0415

    if settings.app_password is None or settings.session_secret is None:
        msg = "the site needs APP_PASSWORD and SESSION_SECRET in its environment"
        raise RuntimeError(msg)
    service = SearchService(index)
    catalog = service.engine.catalog
    backend = InProcessBackend(service)
    model = chat_model(settings)
    publishable = read_publishable(index / PUBLISHABLE_FILE)
    limits = Limits(settings.chat_conversation_usd, settings.chat_day_usd)
    log = FileChatLog(state / CHAT_LOGS)
    spend = FileSpend(state / CHAT_DAYS)

    def chat(fetch: Callable[[str], Awaitable[bytes]]) -> ChatService:
        assert model is not None
        return ChatService(
            backend=backend,
            model=model,
            thumbs=ThumbSource(publishable, fetch),
            spend=spend,
            limits=limits,
            index_size=(len(catalog.images), catalog.entity_count),
            catalog=catalog,
            log=log,
        )

    site = create_app(
        catalog=catalog,
        backend=backend,
        password=settings.app_password.get_secret_value(),
        session_key=settings.session_secret.get_secret_value().encode(),
        chat=None if model is None else chat,
        on_visit=VisitMark(state / LAST_VISIT).touch,
    )
    logger.info("conversation page: %s", "off (no key)" if model is None else model.model)
    # Caddy, on the same machine, is the only client: it sets X-Forwarded-For.
    uvicorn.run(
        site,
        host="127.0.0.1",
        port=port,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )
