"""The website's FastAPI app: the DSFR page, login, search, record pages, example photos.

`create_app` takes a search backend: the Modal GPU class when deployed
(`miroir.web.modal_app`), a local engine with fake models for `miroir web dev` and tests.
Every /api route but login needs a session cookie; /api/search is rate-limited per session.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from miroir.config import WIKIMEDIA_USER_AGENT
from miroir.images.fetch import to_jpeg
from miroir.web import auth
from miroir.web.catalog import (
    COLLECTION_HINTS,
    COLLECTION_LABELS,
    MATCHED_LABELS,
    PAGE_IMAGES,
    PAGE_SIDE,
    Catalog,
    FacetOptions,
    Filters,
    Source,
    display_url,
)
from miroir.web.engine import MAX_TEXT, SearchRequest, SearchResponse
from miroir.web.examples import EXAMPLES

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
#: Largest photo accepted, after base64 decoding. The page shrinks photos before sending.
MAX_PHOTO_BYTES = 8 * 1024 * 1024
#: Longest wait on the search backend before the page is told to retry (Modal cuts at 150 s).
SEARCH_WAIT_S = 100.0
#: Example photos are served at most this size (the page shrinks them again to search).
EXAMPLE_SIDE = 1280


class SearchBackend(Protocol):
    async def search(self, request: SearchRequest) -> SearchResponse: ...

    async def warm(self) -> None:
        """Start the search container if it sleeps, without waiting for it."""
        ...


class LoginBody(BaseModel):
    password: str = Field(max_length=200)


class FiltersBody(BaseModel):
    collections: list[str] = Field(default_factory=list[str], max_length=5)
    region: str | None = Field(default=None, max_length=100)
    departement: str | None = Field(default=None, max_length=100)
    domain: str | None = Field(default=None, max_length=100)
    period: str | None = Field(default=None, max_length=50)

    def filters(self) -> Filters:
        return Filters(
            collections=tuple(c for c in self.collections if c in COLLECTION_LABELS),
            region=self.region or None,
            departement=self.departement or None,
            domain=self.domain or None,
            period=self.period or None,
        )


class SearchBody(BaseModel):
    image: str | None = Field(default=None, description="The photo as base64 (JPEG or PNG).")
    text: str = Field(default="", max_length=MAX_TEXT)
    filters: FiltersBody = Field(default_factory=FiltersBody)


@dataclass(frozen=True, slots=True)
class PageImage:
    url: str
    fallback_url: str
    label: str | None
    credit: str | None
    credit_url: str | None
    ref: str | None


@dataclass(frozen=True, slots=True)
class Fact:
    label: str
    value: str


@dataclass(frozen=True, slots=True)
class LinkedRecord:
    ref: str
    title: str
    pop_url: str


@dataclass(frozen=True, slots=True)
class EntityPage:
    entity_id: str
    title: str
    place: str
    collection: str
    collection_label: str
    collection_hint: str
    pop_url: str
    description: str
    facts: tuple[Fact, ...]
    images: tuple[PageImage, ...]
    records: tuple[LinkedRecord, ...]
    photo_count: int


@dataclass(frozen=True, slots=True)
class ExampleCard:
    id: str
    label: str
    kind: str
    image_url: str
    credit: str
    frame: bool


@dataclass(frozen=True, slots=True)
class Stats:
    images: int
    records: int
    entities: int


@dataclass(frozen=True, slots=True)
class Site:
    """What the routes share, kept in `app.state.site`."""

    catalog: Catalog
    backend: SearchBackend
    password: str
    sessions: auth.Sessions
    secure_cookies: bool
    search_limit: auth.RateLimit
    login_limit: auth.RateLimit
    facet_options: FacetOptions
    fetch: Callable[[str], Awaitable[bytes]]
    example_cache: dict[str, bytes]


def get_site(request: Request) -> Site:
    site: Site = request.app.state.site
    return site


def require_session(request: Request, site: Annotated[Site, Depends(get_site)]) -> str:
    found = site.sessions.check(request.cookies.get(auth.COOKIE))
    if found is None:
        raise HTTPException(status_code=401, detail="Connectez-vous pour continuer.")
    return found


SiteDep = Annotated[Site, Depends(get_site)]
SessionId = Annotated[str, Depends(require_session)]
api = APIRouter(prefix="/api")


def create_app(
    *,
    catalog: Catalog,
    backend: SearchBackend,
    password: str,
    session_key: bytes,
    secure_cookies: bool = True,
    fetch: Callable[[str], Awaitable[bytes]] | None = None,
) -> FastAPI:
    """The app. `fetch` downloads an example photo (tests pass a fake)."""
    if not password:
        msg = "the site needs a password (`miroir web secret` sets it)"
        raise ValueError(msg)
    app = FastAPI(title="Miroir", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.site = Site(
        catalog=catalog,
        backend=backend,
        password=password,
        sessions=auth.Sessions(session_key),
        secure_cookies=secure_cookies,
        search_limit=auth.RateLimit(auth.SEARCHES_PER_MINUTE),
        login_limit=auth.RateLimit(auth.LOGINS_PER_MINUTE),
        facet_options=catalog.facet_options(),
        fetch=fetch or _fetch,
        example_cache={},
    )
    app.middleware("http")(_headers)
    app.add_api_route("/", _index, methods=["GET"], include_in_schema=False)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(api)
    return app


async def _headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["X-Frame-Options"] = "DENY"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


async def _index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@api.post("/login")
async def login(body: LoginBody, request: Request, site: SiteDep) -> JSONResponse:
    client = request.client.host if request.client else "unknown"
    if not site.login_limit.allow(client):
        raise HTTPException(status_code=429, detail="Trop d’essais. Réessayez dans une minute.")
    if not auth.password_ok(body.password, site.password):
        raise HTTPException(status_code=401, detail="Mot de passe incorrect.")
    response = JSONResponse({"ok": True})
    response.set_cookie(
        auth.COOKIE,
        site.sessions.issue(),
        max_age=site.sessions.lifetime_s,
        httponly=True,
        secure=site.secure_cookies,
        samesite="lax",
    )
    return response


@api.post("/logout")
async def logout() -> JSONResponse:
    response = JSONResponse({"ok": True})
    response.delete_cookie(auth.COOKIE)
    return response


@api.get("/me")
async def me(_: SessionId) -> dict[str, bool]:
    return {"ok": True}


@api.post("/warm")
async def warm(_: SessionId, site: SiteDep) -> dict[str, bool]:
    await site.backend.warm()
    return {"ok": True}


@api.get("/stats")
async def stats(_: SessionId, site: SiteDep) -> Stats:
    catalog = site.catalog
    return Stats(len(catalog.images), len(catalog.records), catalog.entity_count)


@api.get("/facets")
async def facets(_: SessionId, site: SiteDep) -> FacetOptions:
    return site.facet_options


@api.get("/examples")
async def example_list(_: SessionId) -> list[ExampleCard]:
    return [
        ExampleCard(e.id, e.label, e.kind, f"/api/examples/{e.id}/image", e.credit, e.frame)
        for e in EXAMPLES
    ]


@api.get("/examples/{example_id}/image")
async def example_image(example_id: str, _: SessionId, site: SiteDep) -> Response:
    example = next((e for e in EXAMPLES if e.id == example_id), None)
    if example is None:
        raise HTTPException(status_code=404, detail="Exemple inconnu.")
    if example_id not in site.example_cache:
        original = await site.fetch(example.url)
        site.example_cache[example_id] = await asyncio.to_thread(to_jpeg, original, EXAMPLE_SIDE)
    return Response(
        site.example_cache[example_id],
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=86400"},
    )


@api.post("/search")
async def search(body: SearchBody, session_id: SessionId, site: SiteDep) -> SearchResponse:
    if not site.search_limit.allow(session_id):
        raise HTTPException(
            status_code=429, detail="Trop de recherches en une minute. Patientez un instant."
        )
    image = _decode_photo(body.image)
    if image is None and not body.text.strip():
        raise HTTPException(status_code=422, detail="Ajoutez une photo ou des mots.")
    request = SearchRequest(image=image, text=body.text, filters=body.filters.filters())
    start = time.monotonic()
    # Modal answers a request still open after 150 s with a redirect that loses the body, so a
    # search waiting on a starting GPU container gives up first with 503 and the page retries.
    # The call itself goes on (shielded), so the container keeps starting.
    call = asyncio.ensure_future(site.backend.search(request))
    try:
        result = await asyncio.wait_for(asyncio.shield(call), SEARCH_WAIT_S)
    except TimeoutError as err:
        raise HTTPException(
            status_code=503,
            detail="Le moteur de recherche démarre. Nouvel essai dans un instant.",
            headers={"Retry-After": "2"},
        ) from err
    except ValueError as err:
        logger.warning("search refused: %s", err)
        raise HTTPException(status_code=422, detail="Cette image n’a pas pu être lue.") from err
    _log_search(request, result, time.monotonic() - start)
    return result


@api.get("/entity/{entity_id}")
async def entity(entity_id: str, _: SessionId, site: SiteDep) -> EntityPage:
    page = entity_page(site.catalog, entity_id)
    if page is None:
        raise HTTPException(status_code=404, detail="Fiche introuvable.")
    return page


def entity_page(catalog: Catalog, entity_id: str) -> EntityPage | None:
    """Record page (screen 4): the entity's images, key facts and POP links."""
    row = catalog.entity_row.get(entity_id)
    if row is None:
        return None
    records = catalog.entity_records(row)
    if not records:
        return None
    rows = [int(r) for r in catalog.entity_image_rows(row)]
    rows.sort(key=lambda r: (not catalog.images[r].is_cover, r))
    images: list[PageImage] = []
    for r in rows[:PAGE_IMAGES]:
        image = catalog.images[r]
        commons = image.source is Source.COMMONS
        images.append(
            PageImage(
                url=display_url(image, PAGE_SIDE),
                fallback_url=image.url,
                label="Photo Wikimedia Commons"
                if commons
                else (None if image.is_cover else MATCHED_LABELS.get(image.image_type)),
                credit=_credit(image.author, image.licence) if commons else None,
                credit_url=(image.page_url or None) if commons else None,
                ref=image.ref,
            )
        )
    main = records[0]
    authors = sorted({a for r in records for a in r.authors})
    facts = [
        Fact("Dénomination", ", ".join(dict.fromkeys(d for r in records for d in r.denomination))),
        Fact("Domaine", ", ".join(dict.fromkeys(d for r in records for d in r.domain))),
        Fact("Auteur" if len(authors) < 2 else "Auteurs", ", ".join(authors[:6])),
        Fact("Période", ", ".join(dict.fromkeys(p for p in main.period))),
        Fact("Commune", main.commune or ""),
        Fact("Département", main.departement or ""),
        Fact("Région", main.region or ""),
        Fact("Référence", entity_id if entity_id != main.ref else main.ref),
    ]
    return EntityPage(
        entity_id=entity_id,
        title=main.display_title,
        place=main.place,
        collection=main.collection,
        collection_label=COLLECTION_LABELS.get(main.collection, main.collection),
        collection_hint=COLLECTION_HINTS.get(main.collection, ""),
        pop_url=main.pop_url,
        description=main.description,
        facts=tuple(f for f in facts if f.value.strip()),
        images=tuple(images),
        records=tuple(LinkedRecord(r.ref, r.display_title, r.pop_url) for r in records[:40]),
        photo_count=len(rows),
    )


def _credit(author: str, licence: str) -> str:
    return ", ".join(p for p in (author, licence) if p)


def _decode_photo(data: str | None) -> bytes | None:
    if not data:
        return None
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    if len(data) > MAX_PHOTO_BYTES * 4 // 3 + 4:
        raise HTTPException(status_code=413, detail="Photo trop lourde.")
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as err:
        raise HTTPException(status_code=422, detail="Photo illisible.") from err


def _log_search(request: SearchRequest, result: SearchResponse, seconds: float) -> None:
    """Technical spec §9: the query (image hash, text, filters), what came back, the time."""
    digest = hashlib.sha256(request.image).hexdigest()[:12] if request.image else "-"
    logger.info(
        "search image=%s text=%r filters=%s mode=%s match=%s top=%s %.0f ms",
        digest,
        request.text[:80],
        request.filters,
        result.mode,
        result.match,
        [c.entity_id for c in result.cards[:5]],
        1000 * seconds,
    )


async def _fetch(url: str) -> bytes:
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers={"User-Agent": WIKIMEDIA_USER_AGENT}
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content
