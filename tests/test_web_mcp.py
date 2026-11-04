"""The MCP connector (`miroir.web.mcp`): tokens, tools over HTTP, the agent, image links."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

import httpx
import httpx2
import pytest
from fastapi import FastAPI
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent
from PIL import Image as PILImage
from pydantic import SecretStr

from miroir.config import Settings
from miroir.jobs.openrouter import JSON, ChatAnswer
from miroir.web import catalog as catalog_module
from miroir.web import mcp_photo
from miroir.web.app import Site, create_app
from miroir.web.catalog import COLLECTION_LABELS
from miroir.web.chat import ChatModel, ChatService, Limits, MemorySpend, ThumbSource
from miroir.web.facets import PERIODS
from miroir.web.fake import FakeGuide, LocalBackend
from miroir.web.mcp import (
    _HideTokens,
    build_server,
    format_tokens,
    mount_mcp,
    new_token,
    parse_tokens,
)
from miroir.web.mcp_photo import PhotoError, PhotoFetcher, checked_photo, public_address
from miroir.web.mcp_tools import FilterArgs, McpTools, ToolCallError
from miroir.web.ops import read_env_value, set_mcp_token
from miroir.web.publishable import Publishable
from tests import web_helpers as h
from tests.helpers import FIXTURE_EXPORTS

TOKEN = "a" * 40
OTHER = "b" * 40
PUBLIC_IP = "93.184.216.34"
#: The fake catalog's images that may be shown (PA1's second photo and OBJ1's back may not).
SHOWN = ("m1", "j1", "r1")


@pytest.fixture(autouse=True)
def every_region_in_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny test catalog has fewer records per region than the menu asks for."""
    monkeypatch.setattr(catalog_module, "MIN_REGION_RECORDS", 1)


def run[T](work: Awaitable[T]) -> T:
    async def wrapped() -> T:
        return await work

    return asyncio.run(wrapped())


def photo_server(
    pages: Mapping[str, httpx.Response], seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    """Answers by the request's Host header and path; 404 elsewhere."""

    def answer(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return pages.get(f"{request.headers['host']}{request.url.path}", httpx.Response(404))

    return httpx.MockTransport(answer)


def resolver(addresses: Mapping[str, Sequence[str]]) -> mcp_photo.Resolver:
    async def resolve(host: str, port: int) -> list[str]:
        assert port == 443
        if host not in addresses:
            raise OSError("unknown host")
        return list(addresses[host])

    return resolve


def png(shade: int = 90) -> httpx.Response:
    return httpx.Response(200, content=h.photo(shade), headers={"content-type": "image/png"})


def make_app(
    *,
    agent: bool = True,
    model: ChatModel | None = None,
    limits: Limits | None = None,
    pages: Mapping[str, httpx.Response] | None = None,
) -> FastAPI:
    engine, _ = h.engine(building=False)
    backend = LocalBackend(engine)

    async def fetch(url: str) -> bytes:
        del url
        return h.photo(90)

    def chat(given: Callable[[str], Awaitable[bytes]]) -> ChatService:
        return ChatService(
            backend=backend,
            model=model or FakeGuide(),
            thumbs=ThumbSource(Publishable(SHOWN), given),
            spend=MemorySpend(),
            limits=limits or Limits(0.10, 2.0),
            index_size=(6, 3),
            catalog=engine.catalog,
            today=lambda: "2026-11-04",
        )

    app = create_app(
        catalog=engine.catalog,
        backend=backend,
        password="mot-de-passe",
        session_key=b"k" * 32,
        secure_cookies=False,
        fetch=fetch,
        chat=chat if agent else None,
    )
    photos = PhotoFetcher(
        transport=photo_server(pages or {"photos.example/p.png": png()}),
        resolve=resolver({"photos.example": [PUBLIC_IP]}),
    )
    mount_mcp(
        app,
        tokens=parse_tokens(f"chatgpt={TOKEN},claude={OTHER}"),
        publishable=Publishable(SHOWN),
        photos=photos,
    )
    return app


@asynccontextmanager
async def connected(
    app: FastAPI, path: str = f"/mcp/{TOKEN}", headers: Mapping[str, str] | None = None
) -> AsyncIterator[Client]:
    async with app.router.lifespan_context(app):
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url="https://miroir.test",
            headers=dict(headers or {}),
            timeout=30.0,
        )
        transport = streamable_http_client(f"https://miroir.test{path}", http_client=http)
        async with Client(transport) as client:
            yield client


async def call(app: FastAPI, tool: str, args: Mapping[str, object]) -> CallToolResult:
    async with connected(app) as client:
        return await client.call_tool(tool, dict(args))


def structured(result: CallToolResult) -> dict[str, object]:
    assert not result.is_error, text_of(result)
    assert result.structured_content is not None
    return result.structured_content


def text_of(result: CallToolResult) -> str:
    return "\n".join(c.text for c in result.content if isinstance(c, TextContent))


def results(result: CallToolResult) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", structured(result)["results"])


# --- tokens and access -------------------------------------------------------------------------


def test_tokens_are_named_and_checked() -> None:
    tokens = parse_tokens(f" chatgpt={TOKEN}, claude={OTHER} ,")
    assert tokens.name_of(TOKEN) == "chatgpt"
    assert tokens.name_of(OTHER) == "claude"
    assert tokens.name_of("a" * 39) is None
    assert tokens.name_of(None) is None
    assert parse_tokens(format_tokens({"x": TOKEN})).names == {TOKEN: "x"}
    assert len(new_token()) >= 40


@pytest.mark.parametrize("text", ["chatgpt=short", f"Bad Name={TOKEN}", f"a={TOKEN},b={TOKEN}"])
def test_bad_token_entries_are_refused(text: str) -> None:
    with pytest.raises(ValueError, match="bad MCP token"):
        parse_tokens(text)


def test_mcp_token_command_keeps_other_lines(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("MIROIR_APP_PASSWORD=x\n")
    first = set_mcp_token("chatgpt", env_file=env)
    assert set_mcp_token("chatgpt", env_file=env) == first
    second = set_mcp_token("claude", env_file=env)
    assert set_mcp_token("chatgpt", rotate=True, env_file=env) != first
    text = env.read_text()
    assert "MIROIR_APP_PASSWORD=x" in text
    tokens = parse_tokens(read_env_value(env, "MIROIR_MCP_TOKENS") or "")
    assert sorted(tokens.names.values()) == ["chatgpt", "claude"]
    assert tokens.name_of(second) == "claude"


def test_the_demo_site_gets_the_tokens() -> None:
    values = {"app_password": SecretStr("p"), "mcp_tokens": SecretStr(f"chatgpt={TOKEN}")}
    settings = Settings(exports_dir=FIXTURE_EXPORTS).model_copy(update=values)
    assert settings.demo_site_env()["MCP_TOKENS"] == f"chatgpt={TOKEN}"


@pytest.mark.parametrize(
    ("path", "headers"),
    [
        ("/mcp", {}),
        ("/mcp/" + "c" * 40, {}),
        ("/mcp", {"Authorization": f"Basic {TOKEN}"}),
        ("/mcp", {"Authorization": "Bearer " + "c" * 40}),
    ],
)
def test_calls_without_a_good_token_are_refused(path: str, headers: dict[str, str]) -> None:
    app = make_app()

    async def post() -> httpx2.Response:
        async with app.router.lifespan_context(app):
            http = httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app), base_url="https://miroir.test"
            )
            body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            return await http.post(path, json=body, headers=headers)

    response = run(post())
    assert response.status_code == 401
    assert "token" in response.json()["error"]


def test_a_token_in_the_path_or_as_bearer_lists_the_tools() -> None:
    app = make_app()

    async def names(path: str, headers: Mapping[str, str] | None = None) -> list[str]:
        async with connected(app, path, headers) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert run(names(f"/mcp/{TOKEN}")) == ["search", "fetch", "delegate_search"]
    assert run(names("/mcp", {"Authorization": f"Bearer {OTHER}"})) == [
        "search",
        "fetch",
        "delegate_search",
    ]


def test_a_visit_is_marked_on_each_call() -> None:
    app = make_app()
    visits: list[int] = []
    site: Site = app.state.site
    object.__setattr__(site, "on_visit", lambda: visits.append(1))
    run(call(app, "search", {"query": "église"}))
    assert visits


def test_tokens_are_hidden_in_the_access_log() -> None:
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s "%s %s"',
        ("1.2.3.4", "POST", f"/mcp/{TOKEN}"),
        None,
    )
    assert _HideTokens().filter(record)
    assert TOKEN not in record.getMessage()
    assert "/mcp/…" in record.getMessage()


def test_no_token_no_connector() -> None:
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog, backend=LocalBackend(engine), password="p", session_key=b"k" * 32
    )
    assert mount_mcp(app, tokens=parse_tokens(""), publishable=Publishable(())) is None
    assert not [r for r in app.routes if getattr(r, "path", "").startswith("/mcp")]


def test_without_the_agent_only_search_and_fetch_are_offered() -> None:
    app = make_app(agent=False)

    async def names() -> list[str]:
        async with connected(app) as client:
            return [t.name for t in (await client.list_tools()).tools]

    assert run(names()) == ["search", "fetch"]


def test_a_call_that_did_not_come_through_the_gate_is_refused() -> None:
    app = make_app()
    site: Site = app.state.site
    server = build_server(McpTools(site=site, publishable=Publishable(()), photos=PhotoFetcher()))

    async def go() -> CallToolResult:
        async with Client(server) as client:
            return await client.call_tool("search", {"query": "église"})

    result = run(go())
    assert result.is_error
    assert "not authenticated" in text_of(result)


# --- tool schemas ------------------------------------------------------------------------------


def test_tool_schemas_are_written_for_a_model() -> None:
    app = make_app()

    async def listed() -> dict[str, dict[str, object]]:
        async with connected(app) as client:
            return {
                t.name: t.model_dump(mode="json", by_alias=True)
                for t in (await client.list_tools()).tools
            }

    tools = run(listed())
    search = tools["search"]
    props = cast(
        "dict[str, dict[str, object]]",
        cast("dict[str, object]", search["inputSchema"])["properties"],
    )
    assert set(props) >= {"query", "image_url", "collections", "region", "period", "top_k"}
    assert props["top_k"]["maximum"] == 20
    assert "ctx" not in props
    text = str(props)
    for key in COLLECTION_LABELS:
        assert key in text
    for period in PERIODS:
        assert period in text
    assert cast("dict[str, object]", search["annotations"])["readOnlyHint"] is True
    assert "outputSchema" in search
    fetch_props = cast(
        "dict[str, object]", cast("dict[str, object]", tools["fetch"]["inputSchema"])["properties"]
    )
    assert list(fetch_props) == ["id"]
    delegate = cast("dict[str, object]", tools["delegate_search"]["inputSchema"])
    assert {"request", "conversation_id", "reply", "region_image_url"} <= set(
        cast("dict[str, object]", delegate["properties"])
    )
    assert "OpenRouter" in str(tools["delegate_search"]["description"])


# --- search and fetch --------------------------------------------------------------------------


def test_search_by_words_returns_records_and_json_text() -> None:
    result = run(call(make_app(), "search", {"query": "Saint-Pierre", "top_k": 2}))
    found = results(result)
    assert len(found) == 2
    church = next(r for r in found if r["id"] == "PA1")
    assert church["url"] in (
        "https://pop.culture.gouv.fr/notice/memoire/AP1",
        "https://pop.culture.gouv.fr/notice/memoire/AP2",
    )
    assert church["collection"] == "Mémoire"
    assert church["place"] == "Toulouse, Haute-Garonne"
    assert church["level"] in ("tres-proche", "proche", "ressemblance")
    assert '"results"' in text_of(result)  # deep research reads the JSON in the text


def test_thumbnails_are_publishable_images_only() -> None:
    found = {r["id"]: r for r in results(run(call(make_app(), "search", {"query": "vase"})))}
    assert found["PA1"]["thumbnail_url"] == "https://example.org/m1.jpg"
    assert found["OBJ1"]["thumbnail_url"] == "https://example.org/j1.jpg"
    for hit in found.values():
        assert hit["thumbnail_url"] in (None, *(f"https://example.org/{i}.jpg" for i in SHOWN))


def test_a_search_with_nothing_to_search_says_what_to_give() -> None:
    result = run(call(make_app(), "search", {}))
    assert result.is_error
    assert "give a query" in text_of(result)


def test_filters_match_menu_values_whatever_the_case() -> None:
    found = results(run(call(make_app(), "search", {"query": "église", "region": "occitanie"})))
    assert [r["id"] for r in found] == ["PA1"]
    found = results(run(call(make_app(), "search", {"query": "vase", "collections": ["joconde"]})))
    assert [r["id"] for r in found] == ["OBJ1"]


def test_an_unknown_filter_value_lists_valid_ones() -> None:
    result = run(call(make_app(), "search", {"query": "église", "region": "Atlantide"}))
    assert result.is_error
    assert "Normandie" in text_of(result)
    bad = run(call(make_app(), "search", {"query": "x", "collections": ["louvre"]}))
    assert bad.is_error  # not in the schema's list


def test_search_by_image_link_downloads_it_safely() -> None:
    seen: list[httpx.Request] = []
    app = make_app(pages={"photos.example/p.png": png()})
    # the fetcher answers; check what it sent
    site: Site = app.state.site
    tools = McpTools(
        site=site,
        publishable=Publishable(SHOWN),
        photos=PhotoFetcher(
            transport=photo_server({"photos.example/p.png": png()}, seen),
            resolve=resolver({"photos.example": [PUBLIC_IP]}),
        ),
    )
    found = run(tools.search("t", image_url="https://photos.example/p.png"))
    assert found.mode == "photo"
    assert found.results
    assert str(seen[0].url) == f"https://{PUBLIC_IP}/p.png"
    assert seen[0].headers["host"] == "photos.example"
    assert seen[0].extensions["sni_hostname"] == "photos.example"
    over_http = run(call(app, "search", {"image_url": "https://photos.example/p.png"}))
    assert structured(over_http)["mode"] == "photo"


def test_search_by_inline_image() -> None:
    data = base64.b64encode(h.photo(200)).decode()
    result = run(call(make_app(), "search", {"image_base64": data, "query": "Toulouse"}))
    assert structured(result)["mode"] == "photo+text"
    both = run(
        call(make_app(), "search", {"image_base64": data, "image_url": "https://x.example/a"})
    )
    assert both.is_error


def test_position_needs_both_numbers() -> None:
    result = run(call(make_app(), "search", {"query": "église", "latitude": 43.6}))
    assert result.is_error
    assert "longitude" in text_of(result)


def test_calls_are_limited_per_token() -> None:
    app = make_app()
    site: Site = app.state.site
    tools = McpTools(site=site, publishable=Publishable(()), photos=PhotoFetcher())
    tools.call_limit.limit = 2
    run(tools.search("chatgpt", query="église"))
    run(tools.fetch("chatgpt", "PA1"))
    with pytest.raises(ToolCallError, match="too many calls"):
        run(tools.search("chatgpt", query="église"))
    run(tools.search("claude", query="église"))  # each token has its own count


def test_fetch_gives_the_record_in_full() -> None:
    document = structured(run(call(make_app(), "fetch", {"id": "PA1"})))
    assert document["id"] == "PA1"
    assert document["title"] == "Église Saint-Pierre"
    assert document["url"] == "https://pop.culture.gouv.fr/notice/memoire/AP1"
    assert "Toulouse" in str(document["text"])
    metadata = cast("dict[str, object]", document["metadata"])
    assert [r["ref"] for r in cast("list[dict[str, str]]", metadata["records"])] == ["AP1", "AP2"]
    assert metadata["images"] == ["https://example.org/m1.jpg"]  # m2 may not be shown


def test_fetch_takes_a_pop_reference_and_refuses_an_unknown_id() -> None:
    assert structured(run(call(make_app(), "fetch", {"id": "AP2"})))["id"] == "PA1"
    missing = run(call(make_app(), "fetch", {"id": "XX9"}))
    assert missing.is_error
    assert "use an id from `search`" in text_of(missing)


def test_filters_are_checked_before_the_search() -> None:
    app = make_app()
    site: Site = app.state.site
    tools = McpTools(site=site, publishable=Publishable(()), photos=PhotoFetcher())
    found = tools.filters(FilterArgs(("memoire",), "OCCITANIE", "haute-garonne", "architecture"))
    assert (found.region, found.departement, found.domain) == (
        "Occitanie",
        "Haute-Garonne",
        "Architecture",
    )
    with pytest.raises(ToolCallError, match="unknown departement"):
        tools.filters(FilterArgs(departement="Atlantide"))


# --- delegate_search ---------------------------------------------------------------------------


def test_the_agent_answers_words_with_its_picks_and_reasons() -> None:
    result = run(call(make_app(), "delegate_search", {"request": "église Saint-Pierre"}))
    answer = structured(result)
    assert answer["status"] == "done"
    picks = cast("list[dict[str, object]]", answer["results"])
    assert picks
    assert picks[0]["reason"] == "Choix du modèle factice"
    assert answer["steps"]
    text = text_of(result)
    assert "done" in text
    assert "Choix du modèle factice" in text


def test_the_agent_asks_and_the_caller_answers() -> None:
    app = make_app()
    photo = {"image_url": "https://photos.example/p.png", "request": "Qu’est-ce que c’est ?"}

    async def go() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        async with connected(app) as client:
            first = structured(await client.call_tool("delegate_search", photo))
            cid = first["conversation_id"]
            again = structured(await client.call_tool("delegate_search", {"conversation_id": cid}))
            reply = {"conversation_id": cid, "reply": "Un bâtiment"}
            second = structured(await client.call_tool("delegate_search", reply))
            return first, again, second

    first, again, second = run(go())
    assert first["status"] == "question"
    assert first["question"] == "Que montre votre photo, en quelques mots ?"
    assert first["choices"] == ["Un bâtiment", "Un tableau", "Un objet"]
    assert again["status"] == "question"  # collecting again changes nothing
    assert second["status"] == "done"
    assert second["conversation_id"] == first["conversation_id"]
    assert second["results"]


def test_without_questions_the_agent_answers_in_one_go() -> None:
    args = {"image_url": "https://photos.example/p.png", "allow_questions": False}
    answer = structured(run(call(make_app(), "delegate_search", args)))
    assert answer["status"] == "done"
    assert answer["question"] is None


def test_a_conversation_belongs_to_its_token() -> None:
    app = make_app()
    answer = structured(run(call(app, "delegate_search", {"request": "église"})))

    async def other() -> CallToolResult:
        headers = {"Authorization": f"Bearer {OTHER}"}
        async with connected(app, "/mcp", headers) as client:
            args = {"conversation_id": answer["conversation_id"], "reply": "oui"}
            return await client.call_tool("delegate_search", args)

    refused = run(other())
    assert refused.is_error
    assert "unknown or expired conversation_id" in text_of(refused)


def test_the_agent_keeps_the_spending_caps() -> None:
    app = make_app(limits=Limits(conversation_usd=0.0, day_usd=2.0))
    answer = structured(run(call(app, "delegate_search", {"request": "église"})))
    assert answer["status"] == "stopped"
    assert answer["results"] == []


def test_a_long_turn_answers_running_then_its_results() -> None:
    gate = asyncio.Event()

    class Slow:
        """The fake guide, held back until the test lets it go."""

        model = "slow"

        async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
            await gate.wait()
            return await FakeGuide().complete(body)

    app = make_app(model=Slow())

    async def go() -> tuple[CallToolResult, dict[str, object]]:
        async with connected(app) as client:
            args = {"request": "église", "wait_seconds": 5}
            first = await client.call_tool("delegate_search", args)
            gate.set()
            then = {"conversation_id": structured(first)["conversation_id"]}
            return first, structured(await client.call_tool("delegate_search", then))

    first, second = run(go())
    assert structured(first)["status"] == "running"
    assert "call delegate_search again" in text_of(first)
    assert second["status"] == "done"
    assert second["results"]


def test_agent_turns_are_limited_per_token() -> None:
    app = make_app()
    site: Site = app.state.site
    tools = McpTools(site=site, publishable=Publishable(()), photos=PhotoFetcher())
    tools.agent_limit.limit = 1
    run(tools.delegate("chatgpt", request="église"))
    with pytest.raises(ToolCallError, match="too many calls"):
        run(tools.delegate("chatgpt", request="église"))


# --- image links -------------------------------------------------------------------------------


def fetcher(
    pages: Mapping[str, httpx.Response], addresses: Mapping[str, Sequence[str]]
) -> PhotoFetcher:
    return PhotoFetcher(transport=photo_server(pages), resolve=resolver(addresses))


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://photos.example/p.png", "https://"),
        ("ftp://photos.example/p.png", "https://"),
        ("https://user:pw@photos.example/p.png", "user name"),
        ("https://photos.example:8443/p.png", "port"),
        ("https://inside.example/p.png", "public"),
        ("https://loop.example/p.png", "public"),
        ("https://mapped.example/p.png", "public"),
        ("https://mixed.example/p.png", "public"),
        ("https://nowhere.example/p.png", "not known"),
        ("https://photos.example/missing.png", "HTTP 404"),
    ],
)
def test_image_links_that_are_refused(url: str, message: str) -> None:
    addresses = {
        "photos.example": [PUBLIC_IP],
        "inside.example": ["10.0.0.7"],
        "loop.example": ["127.0.0.1"],
        "mapped.example": ["::ffff:169.254.169.254"],
        "mixed.example": [PUBLIC_IP, "192.168.1.1"],
    }
    with pytest.raises(PhotoError, match=message):
        run(fetcher({"photos.example/p.png": png()}, addresses).fetch(url))


def test_redirects_are_checked_at_each_hop() -> None:
    pages = {
        "photos.example/a": httpx.Response(302, headers={"location": "/p.png"}),
        "photos.example/p.png": png(),
        "photos.example/evil": httpx.Response(302, headers={"location": "https://meta.example/"}),
        "photos.example/loop": httpx.Response(302, headers={"location": "/loop"}),
    }
    addresses = {"photos.example": [PUBLIC_IP], "meta.example": ["169.254.169.254"]}
    got = fetcher(pages, addresses)
    assert run(got.fetch("https://photos.example/a")) == h.photo(90)
    with pytest.raises(PhotoError, match="public"):
        run(got.fetch("https://photos.example/evil"))
    with pytest.raises(PhotoError, match="redirects more than"):
        run(got.fetch("https://photos.example/loop"))


def test_large_downloads_stop_at_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_photo, "MAX_PHOTO_BYTES", 1000)
    declared = httpx.Response(200, content=b"x" * 10, headers={"content-length": "5000"})
    streamed = httpx.Response(200, content=b"x" * 5000)
    pages = {"photos.example/declared": declared, "photos.example/streamed": streamed}
    got = fetcher(pages, {"photos.example": [PUBLIC_IP]})
    for path in ("declared", "streamed"):
        with pytest.raises(PhotoError, match="larger than"):
            run(got.fetch(f"https://photos.example/{path}"))


def test_a_slow_link_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_photo, "FETCH_TIMEOUT_S", 0.05)

    async def slow(host: str, port: int) -> list[str]:
        del host, port
        await asyncio.sleep(1)
        return [PUBLIC_IP]

    with pytest.raises(PhotoError, match="did not answer"):
        run(PhotoFetcher(resolve=slow).fetch("https://photos.example/p.png"))


def test_only_readable_images_of_bounded_size_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    assert checked_photo(h.photo(90)) == h.photo(90)
    with pytest.raises(PhotoError, match="not an image"):
        checked_photo(b"<html>no</html>")
    monkeypatch.setattr(mcp_photo, "MAX_PIXELS", 100)
    out = io.BytesIO()
    PILImage.new("RGB", (20, 20)).save(out, format="PNG")
    with pytest.raises(PhotoError, match="million pixels"):
        checked_photo(out.getvalue())


@pytest.mark.parametrize(
    ("address", "public"),
    [
        (PUBLIC_IP, True),
        ("2606:4700:4700::1111", True),
        ("10.1.2.3", False),
        ("172.16.0.1", False),
        ("100.64.0.1", False),
        ("169.254.169.254", False),
        ("::1", False),
        ("fd00::1", False),
        ("::ffff:10.0.0.1", False),
        ("224.0.0.1", False),
        ("0.0.0.0", False),
        ("not-an-ip", False),
    ],
)
def test_public_addresses(address: str, public: bool) -> None:
    assert public_address(address) is public
