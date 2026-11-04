"""MIROIR as a remote MCP server, so ChatGPT or Claude can search POP through it on their own.

Mounted on the site's app by `mount_mcp` (the demo machine and `miroir web dev` call it when
connector tokens are set): Streamable HTTP, stateless, at `/mcp/<token>`, or at `/mcp` with
`Authorization: Bearer <token>`. ChatGPT's developer mode takes OAuth or no authentication,
so the token rides in the URL there; Claude takes the URL or a bearer header (prototype.md,
"Utiliser MIROIR depuis ChatGPT ou Claude"). Tokens are named (`name=token,...`,
`Settings.mcp_tokens`, made by `miroir web mcp-token`): each name has its own rate limits and
agent conversations. The site's password and session cookie are not used here.

Tools (`miroir.web.mcp_tools` does the work): `search` (the site's simple search), `fetch`
(one record in full) and, when the site's agent is on, `delegate_search` (a turn of MIROIR's
own agent). The access log hides the tokens in paths (`_HideTokens`).
"""

from __future__ import annotations

import hmac
import logging
import re
import secrets
from collections.abc import AsyncGenerator, Awaitable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import FastAPI
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from miroir.web.app import MAX_PHOTO_BYTES, Site
from miroir.web.mcp_photo import PhotoFetcher
from miroir.web.mcp_tools import (
    AGENT_WAIT_S,
    MAX_AGENT_WAIT_S,
    MAX_TOP_K,
    AgentAnswer,
    CollectionKey,
    Document,
    FilterArgs,
    McpTools,
    PeriodName,
    SearchResult,
    ToolCallError,
)
from miroir.web.nearby import Position
from miroir.web.publishable import Publishable

logger = logging.getLogger(__name__)

#: Where the connector answers, under the site's address.
MCP_PATH = "/mcp"
#: The key in the request scope that names the caller (the token's name).
CALLER = "miroir.mcp_caller"
#: Shortest token accepted, and names a token may have.
MIN_TOKEN = 32
_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,30}$")
#: Largest request: a photo inline as base64, and room for the rest.
MAX_BODY = MAX_PHOTO_BYTES * 4 // 3 + 64 * 1024

INSTRUCTIONS = """\
MIROIR searches POP (pop.culture.gouv.fr), the French Ministry of Culture's platform of \
heritage records: museum works (Joconde), heritage photographs (Mémoire), listed buildings \
(Mérimée), listed objects (Palissy), works looted in 1940-45 (Rose Valland), museums \
(Muséofile). Records are in French.

- `search`: fast (under a second). A photo (as a public https link) and/or words, with \
optional filters and position; closest buildings and objects first, each with a level: \
"tres-proche" (most likely the very record), "proche" (maybe), "ressemblance" (looks alike). \
Only the first result can be above "ressemblance". Words work best as names, places and \
French terms ("retable", "Pont du Gard").
- `fetch`: one result in full (description, facts, its POP records). Use it before you state \
facts about a record.
- `delegate_search` (when offered): hands the whole search to MIROIR's own agent, which \
searches several ways, looks at the images and explains each pick. Slower (20 s to 2 min) and \
it may ask a question: relay it to the user and call again with their answer.

Cite results by their POP url. Images in results may be shown to the user.\
"""

Text = Annotated[str, Field(max_length=500)]
ImageUrl = Annotated[
    str | None,
    Field(
        description="The photo as a public https link (JPEG, PNG, WebP; at most 8 MB). MIROIR "
        "downloads it. A photo the user uploaded into the chat cannot be read from here: ask "
        "for a link.",
        max_length=2000,
    ),
]
ImageBase64 = Annotated[
    str | None,
    Field(description="The photo as base64 (or a data: URL), for clients that hold its bytes."),
]
Collections = Annotated[
    list[CollectionKey] | None,
    Field(
        description="Only these POP bases (joconde, memoire, merimee, palissy, …).", max_length=6
    ),
]
Region = Annotated[
    str | None, Field(description="A French region as of 2016, e.g. 'Occitanie', 'Grand Est'.")
]
Departement = Annotated[
    str | None, Field(description="A département by name, e.g. 'Côte-d'Or', 'Haute-Garonne'.")
]
Domain = Annotated[
    str | None,
    Field(description="A POP domain in French, e.g. 'Peinture', 'Architecture', 'Sculpture'."),
]
Period = Annotated[PeriodName | None, Field(description="A broad period.")]
Latitude = Annotated[
    float | None,
    Field(
        ge=-90,
        le=90,
        description="With longitude: where the photo was taken or the user stands. Nearby "
        "buildings and objects rank higher; far ones stay.",
    ),
]
Longitude = Annotated[float | None, Field(ge=-180, le=180)]


# --- tokens -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Tokens:
    """Connector secrets, each with the name of the client it was given to."""

    names: Mapping[str, str]
    """Token to name."""

    def name_of(self, token: str | None) -> str | None:
        if not token:
            return None
        found: str | None = None
        for secret, name in self.names.items():  # compare with each, in constant time
            if hmac.compare_digest(secret.encode(), token.encode()):
                found = name
        return found


def parse_tokens(text: str) -> Tokens:
    """`name=token,name=token` (`Settings.mcp_tokens`)."""
    names: dict[str, str] = {}
    for item in (p.strip() for p in text.split(",")):
        if not item:
            continue
        name, _, token = item.partition("=")
        if not _NAME.match(name) or len(token) < MIN_TOKEN or token in names:
            msg = f"bad MCP token entry {name!r}: want name=token, token of {MIN_TOKEN}+ chars"
            raise ValueError(msg)
        names[token] = name
    return Tokens(names)


def format_tokens(by_name: Mapping[str, str]) -> str:
    """The setting's text for tokens given by name."""
    return ",".join(f"{name}={token}" for name, token in sorted(by_name.items()))


def new_token() -> str:
    return secrets.token_urlsafe(32)


# --- the server ---------------------------------------------------------------------------------


def build_server(tools: McpTools) -> MCPServer:
    """The MCP server and its tools; `delegate_search` only when the site's agent is on."""
    server = MCPServer(
        "miroir",
        title="MIROIR: recherche dans POP",
        instructions=INSTRUCTIONS,
        website_url="https://pop.culture.gouv.fr",
    )
    looks = ToolAnnotations(read_only_hint=True, open_world_hint=False, idempotent_hint=True)

    @server.tool(
        name="search",
        title="Chercher dans POP",
        annotations=looks,
        description="Search POP records by photo and/or words, as MIROIR's site does. Returns "
        "buildings and objects, closest first, each with its MIROIR id (for `fetch`), title, "
        "POP url, base, place, period, authors, level and a thumbnail link. Give `query` "
        "(French works best), `image_url`, or both; words then refine the photo.",
    )
    async def search(
        ctx: Context,
        *,
        query: Annotated[
            str,
            Field(
                description="Words: a name, a place, what it shows ('nature morte au lièvre').",
                max_length=300,
            ),
        ] = "",
        image_url: ImageUrl = None,
        image_base64: ImageBase64 = None,
        collections: Collections = None,
        region: Region = None,
        departement: Departement = None,
        domain: Domain = None,
        period: Period = None,
        latitude: Latitude = None,
        longitude: Longitude = None,
        top_k: Annotated[int, Field(ge=1, le=MAX_TOP_K, description="Results wanted.")] = 10,
    ) -> SearchResult:
        caller = _caller(ctx)
        filters = FilterArgs(tuple(collections or ()), region, departement, domain, period)
        return await _run(
            tools.search(
                caller,
                query=query,
                image_url=image_url,
                image_base64=image_base64,
                filters=filters,
                position=_position(latitude, longitude),
                top_k=top_k,
            )
        )

    @server.tool(
        name="fetch",
        title="Lire une notice",
        annotations=looks,
        description="One building or object in full, by the id `search` or `delegate_search` "
        "gave (a POP record reference such as PA00105596 works too): title, description and "
        "facts as text, its POP url, and metadata (its POP records, images that may be shown).",
    )
    async def fetch(
        ctx: Context,
        id: Annotated[str, Field(description="The result's id.", max_length=100)],
    ) -> Document:
        return await _run(tools.fetch(_caller(ctx), id))

    if tools.agent is not None:
        _add_delegate(server, tools)
    return server


def _add_delegate(server: MCPServer, tools: McpTools) -> None:
    @server.tool(
        name="delegate_search",
        title="Confier la recherche à l’agent de MIROIR",
        annotations=ToolAnnotations(read_only_hint=True, open_world_hint=True),
        description="Hand a whole search to MIROIR's own agent (Mistral Large), which runs "
        "several searches, crops or greys the photo, looks at the candidates' images, reads "
        "their records and returns its picks with a reason each (French). Use it when `search` "
        "is not enough. It takes 20 s to 2 min. It may ask a question (status 'question'): "
        "relay it to the user, then call again with `conversation_id` and their answer as "
        "`reply`. Status 'running': call again with only `conversation_id`. The photo and "
        "words go to the agent's model through OpenRouter (or Mistral's API).",
    )
    async def delegate_search(
        ctx: Context,
        *,
        request: Annotated[
            str,
            Field(
                description="What the user wants, in their words: 'what is this and where?', "
                "'who painted it?', 'more like this'. Needed to start; French or English.",
                max_length=500,
            ),
        ] = "",
        conversation_id: Annotated[
            str | None, Field(description="To go on with an earlier call.", max_length=64)
        ] = None,
        reply: Annotated[
            str,
            Field(description="With conversation_id: the user's answer or new message."),
        ] = "",
        image_url: ImageUrl = None,
        image_base64: ImageBase64 = None,
        region_image_url: Annotated[
            str | None,
            Field(
                description="With conversation_id, after status 'frame': a https link to the "
                "part of the photo that matters.",
                max_length=2000,
            ),
        ] = None,
        collections: Collections = None,
        region: Region = None,
        departement: Departement = None,
        domain: Domain = None,
        period: Period = None,
        latitude: Latitude = None,
        longitude: Longitude = None,
        allow_questions: Annotated[
            bool, Field(description="False: the agent asks nothing and answers in one go.")
        ] = True,
        wait_seconds: Annotated[
            int,
            Field(
                ge=5,
                le=MAX_AGENT_WAIT_S,
                description="How long to wait for the agent before answering 'running'.",
            ),
        ] = AGENT_WAIT_S,
    ) -> Annotated[CallToolResult, AgentAnswer]:
        caller = _caller(ctx)
        done = 0

        async def progress(label: str) -> None:
            nonlocal done
            done += 1
            await ctx.report_progress(done, None, label)

        answer = await _run(
            tools.delegate(
                caller,
                request=request,
                conversation_id=conversation_id,
                reply=reply,
                image_url=image_url,
                image_base64=image_base64,
                region_image_url=region_image_url,
                filters=FilterArgs(tuple(collections or ()), region, departement, domain, period),
                position=_position(latitude, longitude),
                allow_questions=allow_questions,
                wait_seconds=wait_seconds,
                progress=progress,
            )
        )
        return CallToolResult(
            content=[TextContent(type="text", text=render_answer(answer))],
            structured_content=answer.model_dump(mode="json"),
        )


def render_answer(answer: AgentAnswer) -> str:
    """The agent's answer as short text for the calling model."""
    lines = [f"MIROIR agent, conversation {answer.conversation_id}: {answer.status}"]
    if answer.summary:
        lines.append(answer.summary)
    for n, hit in enumerate(answer.results, start=1):
        where = f", {hit.place}" if hit.place else ""
        lines.append(f"{n}. {hit.title}{where} ({hit.collection}, {hit.level_label}) {hit.url}")
        if hit.reason:
            lines.append(f"   {hit.reason}")
    if answer.question:
        choices = f" (suggestions: {' / '.join(answer.choices)})" if answer.choices else ""
        lines.append(f"Question for the user: {answer.question}{choices}")
    if answer.status == "running":
        lines.append("Still working: call delegate_search again with only conversation_id.")
    return "\n".join(lines)


async def _run[T](work: Awaitable[T]) -> T:
    """Await a tool's work; its failures become errors the calling model reads."""
    try:
        return await work
    except ToolCallError as err:
        raise ToolError(str(err)) from err


def _caller(ctx: Context) -> str:
    request = getattr(ctx.request_context, "request", None)
    scope: object = getattr(request, "scope", None)
    name = scope.get(CALLER) if isinstance(scope, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if not isinstance(name, str):
        msg = "not authenticated"
        raise ToolError(msg)
    return name


def _position(latitude: float | None, longitude: float | None) -> Position | None:
    if latitude is None and longitude is None:
        return None
    if latitude is None or longitude is None:
        msg = "give both latitude and longitude, or neither"
        raise ToolError(msg)
    return Position(latitude, longitude)


# --- on the site's app ------------------------------------------------------------------------


class _Gate:
    """The token check in front of the MCP transport: the path's token, or a bearer header."""

    def __init__(self, server: MCPServer, tokens: Tokens, site: Site) -> None:
        self._server = server
        self._tokens = tokens
        self._site = site

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        params: object = scope.get("path_params")
        token = params.get("token") if isinstance(params, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        if not isinstance(token, str):
            token = _bearer(scope)
        name = self._tokens.name_of(token)
        if name is None:
            refused = JSONResponse({"error": "unknown or missing MIROIR connector token"}, 401)
            await refused(scope, receive, send)
            return
        self._site.on_visit()
        await self._server.session_manager.handle_request({**scope, CALLER: name}, receive, send)


def _bearer(scope: Scope) -> str | None:
    for key, value in scope.get("headers", ()):
        if key == b"authorization":
            kind, _, token = bytes(value).decode("latin-1").partition(" ")
            return token.strip() if kind.lower() == "bearer" else None
    return None


class _HideTokens(logging.Filter):
    """Masks `/mcp/<token>` in log lines (uvicorn's access log writes every path)."""

    _path = re.compile(rf"{MCP_PATH}/[^/?\s\"]+")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._path.sub(f"{MCP_PATH}/…", a) if isinstance(a, str) else a for a in record.args
            )
        return True


def mount_mcp(
    app: FastAPI,
    *,
    tokens: Tokens,
    publishable: Publishable,
    photos: PhotoFetcher | None = None,
) -> MCPServer | None:
    """Serve the MCP connector on the site's app (`miroir.web.app.create_app`), at
    `/mcp/<token>` and `/mcp`. `publishable` says which images results may show. No token,
    no connector (None)."""
    if not tokens.names:
        logger.info("MCP connector off: no token (`miroir web mcp-token`)")
        return None
    site: Site = app.state.site
    tools = McpTools(site=site, publishable=publishable, photos=photos or PhotoFetcher())
    server = build_server(tools)
    gate = _Gate(server, tokens, site)
    app.router.routes.append(Route(MCP_PATH, gate))
    app.router.routes.append(Route(f"{MCP_PATH}/{{token}}", gate))
    previous = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(host: FastAPI) -> AsyncGenerator[Mapping[str, object]]:
        # A session manager runs once: each start of the app makes a new one. Auth is the
        # token, on a public host name, so no localhost-only check against DNS rebinding.
        server.streamable_http_app(
            stateless_http=True,
            max_request_body_size=MAX_BODY,
            transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
        )
        async with previous(host) as state, server.session_manager.run():
            yield dict(state or {})

    app.router.lifespan_context = lifespan
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _HideTokens) for f in access.filters):
        access.addFilter(_HideTokens())
    logger.info("MCP connector on %s for %s", MCP_PATH, ", ".join(sorted(tokens.names.values())))
    return server
