"""The website's API: login, sessions, rate limit, search, record pages, example photos."""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from miroir.config import Settings
from miroir.images import upload
from miroir.web import app as app_module
from miroir.web import auth
from miroir.web.app import create_app
from miroir.web.bundle import MANIFEST_FILE
from miroir.web.chat import ChatService
from miroir.web.engine import SearchRequest, SearchResponse
from miroir.web.fake import LocalBackend
from miroir.web.ops import (
    read_env_password,
    set_secret,
    upload_file,
    upload_index,
    write_env_password,
)
from tests import web_helpers as h

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

PASSWORD = "test-mot-de-passe"


async def _fake_fetch(url: str) -> bytes:
    del url
    return h.photo(90)


PAGE = "<!doctype html><title>Miroir</title><div id=root></div>"


def built_page(folder: Path) -> Path:
    """A stand-in for the page `make ui` builds (tests run without Node)."""
    (folder / "assets").mkdir(parents=True)
    (folder / "index.html").write_text(PAGE)
    (folder / "assets" / "index.js").write_text("console.log('miroir')")
    return folder


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=LocalBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
        fetch=_fake_fetch,
        ui_dir=built_page(tmp_path / "ui"),
    )
    return TestClient(app)


def login(client: TestClient) -> None:
    assert client.post("/api/login", json={"password": PASSWORD}).status_code == 200


def test_the_built_page_loads_without_login(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.text == PAGE
    assert page.headers["cache-control"] == "no-cache"
    script = client.get("/static/assets/index.js")
    assert script.status_code == 200
    assert script.headers["cache-control"] == "no-cache"
    assert client.get("/static/assets/nothing.js").status_code == 404


def test_every_answer_carries_the_security_headers(client: TestClient) -> None:
    for answer in (client.get("/"), client.get("/static/assets/index.js"), client.get("/api/me")):
        policy = answer.headers["content-security-policy"]
        assert "script-src 'self';" in policy
        assert "frame-ancestors 'none'" in policy
        assert "https://iiif.prd.cloud.culture.fr " in policy  # an origin, not the IIIF path
        assert answer.headers["x-frame-options"] == "DENY"
        assert answer.headers["x-content-type-options"] == "nosniff"


def test_a_page_not_built_says_how_to_build_it(tmp_path: Path) -> None:
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=LocalBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        ui_dir=tmp_path / "missing",
    )
    page = TestClient(app).get("/")
    assert page.status_code == 503
    assert "make ui" in page.text


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/me"),
        ("GET", "/api/facets"),
        ("POST", "/api/search"),
        ("POST", "/api/detect"),
        ("GET", "/api/entity/PA1"),
    ],
)
def test_the_api_needs_a_session(client: TestClient, method: str, path: str) -> None:
    response = client.request(method, path, json={"text": "église"})
    assert response.status_code == 401


def test_a_wrong_password_is_refused(client: TestClient) -> None:
    response = client.post("/api/login", json={"password": "non"})
    assert response.status_code == 401
    assert auth.COOKIE not in response.cookies


def test_logins_are_rate_limited(client: TestClient) -> None:
    codes = [client.post("/api/login", json={"password": "non"}).status_code for _ in range(7)]
    assert codes[: auth.LOGINS_PER_MINUTE] == [401] * auth.LOGINS_PER_MINUTE
    assert codes[-1] == 429


def test_login_sets_a_session_cookie(client: TestClient) -> None:
    login(client)
    assert client.get("/api/me").json() == {"ok": True}
    client.post("/api/logout")
    client.cookies.clear()
    assert client.get("/api/me").status_code == 401


def test_a_tampered_or_expired_token_is_refused() -> None:
    sessions = auth.Sessions(b"secret")
    token = sessions.issue(now=1000.0)
    assert sessions.check(token, now=1001.0) is not None
    assert sessions.check(token[:-2] + "xx", now=1001.0) is None
    assert sessions.check(token, now=1000.0 + auth.SESSION_SECONDS + 1) is None
    assert auth.Sessions(b"other").check(token, now=1001.0) is None
    assert sessions.check("nonsense") is None


def test_searches_are_rate_limited_per_session(client: TestClient) -> None:
    login(client)
    codes = [
        client.post("/api/search", json={"text": "église"}).status_code
        for _ in range(auth.SEARCHES_PER_MINUTE + 1)
    ]
    assert set(codes[:-1]) == {200}
    assert codes[-1] == 429


def test_rate_limit_window_slides() -> None:
    limit = auth.RateLimit(2, window_s=60)
    assert limit.allow("a", now=0)
    assert limit.allow("a", now=1)
    assert not limit.allow("a", now=2)
    assert limit.allow("b", now=2)
    assert limit.allow("a", now=61)


def test_photo_search_returns_cards_and_a_type_guess(client: TestClient) -> None:
    login(client)
    image = base64.b64encode(h.photo(130)).decode()
    body = client.post(
        "/api/search", json={"image": f"data:image/png;base64,{image}", "filters": {}}
    ).json()
    assert body["mode"] == "photo"
    assert body["guessed_type"] == "object"
    assert body["cards"][0]["entity_id"] == "OBJ1"
    assert body["cards"][0]["matched"] == "Revers"
    assert body["cards"][0]["level"] in {"tres-proche", "proche", "ressemblance"}
    assert "match" not in body
    assert "credit" not in body["cards"][0]


def test_filters_reach_the_search(client: TestClient) -> None:
    login(client)
    image = base64.b64encode(h.photo(130)).decode()
    body = client.post(
        "/api/search", json={"image": image, "filters": {"collections": ["rose-valland"]}}
    ).json()
    assert [c["entity_id"] for c in body["cards"]] == ["OBJ2"]


def test_detect_returns_outlines_in_shares_of_the_photo(client: TestClient) -> None:
    login(client)
    image = base64.b64encode(h.photo(130)).decode()
    response = client.post("/api/detect", json={"image": f"data:image/jpeg;base64,{image}"})
    assert response.status_code == 200
    boxes = response.json()["boxes"]
    assert len(boxes) == 3
    for b in boxes:
        assert 0 <= b["x0"] < b["x1"] <= 1
        assert 0 <= b["y0"] < b["y1"] <= 1
        assert b["label"] == "painting"


def test_detect_needs_a_readable_photo_and_is_rate_limited(client: TestClient) -> None:
    login(client)
    assert client.post("/api/detect", json={"image": ""}).status_code == 422
    assert client.post("/api/detect", json={"image": "pas du base64!"}).status_code == 422
    image = base64.b64encode(h.photo(90)).decode()
    codes = [
        client.post("/api/detect", json={"image": image}).status_code
        for _ in range(auth.SEARCHES_PER_MINUTE)
    ]
    assert codes[-1] == 429
    assert client.post("/api/search", json={"text": "église"}).status_code == 200


def test_a_broken_photo_is_refused(client: TestClient) -> None:
    login(client)
    assert client.post("/api/search", json={"image": "pas du base64!"}).status_code == 422
    garbage = base64.b64encode(b"not an image").decode()
    assert client.post("/api/search", json={"image": garbage}).status_code == 422
    assert client.post("/api/search", json={"text": "  "}).status_code == 422


def test_record_page_groups_an_entity(client: TestClient) -> None:
    login(client)
    page = client.get("/api/entity/PA1").json()
    assert page["title"] == "Église Saint-Pierre"
    assert [r["ref"] for r in page["records"]] == ["AP1", "AP2"]
    assert page["photo_count"] == 2  # POP images only: the Commons photo is not shown
    assert [i["ref"] for i in page["images"]] == ["AP1", "AP2"]
    assert all("credit" not in i for i in page["images"])
    assert client.get("/api/entity/NOPE").status_code == 404


def test_example_photos_are_served_after_login(client: TestClient) -> None:
    assert client.get("/api/examples").status_code == 401
    login(client)
    examples = client.get("/api/examples").json()
    assert examples
    image = client.get(examples[0]["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_the_password_line_in_env_is_replaced_not_duplicated(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENROUTER_API_KEY=x\nMIROIR_APP_PASSWORD=old\n", encoding="utf-8")
    write_env_password(env, "new")
    assert env.read_text(encoding="utf-8") == "OPENROUTER_API_KEY=x\nMIROIR_APP_PASSWORD=new\n"
    assert read_env_password(env) == "new"
    assert read_env_password(tmp_path / "missing") is None


def test_the_password_stays_in_env_until_rotated(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    assert set_secret(env_file=env)
    first = read_env_password(env)
    assert first
    assert not set_secret(env_file=env)
    assert read_env_password(env) == first
    assert set_secret(rotate=True, env_file=env)
    assert read_env_password(env) not in (None, first)


class FakeS3:
    """Records what `upload_files` sends; no network."""

    def __init__(self) -> None:
        self.sent: dict[tuple[str, str], bytes] = {}

    def upload_file(self, filename: str, bucket: str, key: str, **_: object) -> None:
        self.sent[bucket, key] = Path(filename).read_bytes()


def test_web_upload_sends_every_bundle_file_to_the_index_bucket(settings: Settings) -> None:
    folder = settings.web_index_dir("small")
    folder.mkdir(parents=True)
    (folder / MANIFEST_FILE).write_text("{}", encoding="utf-8")
    (folder / "places.npz").write_bytes(b"places")
    fake = FakeS3()
    size = upload_index(settings, "small", cast("S3Client", fake))
    bucket = f"{settings.scw_bucket_prefix}-index"
    assert set(fake.sent) == {(bucket, f"small/{MANIFEST_FILE}"), (bucket, "small/places.npz")}
    assert size == 2 + len(b"places")

    again = FakeS3()
    upload_file(settings, "small", "places.npz", cast("S3Client", again))
    assert again.sent == {(bucket, "small/places.npz"): b"places"}


def test_generated_passwords_are_long_and_differ() -> None:
    first, second = auth.new_password(), auth.new_password()
    assert first != second
    assert len(first.replace("-", "")) == 12


def test_a_position_reaches_the_search_and_is_checked(client: TestClient) -> None:
    login(client)
    image = base64.b64encode(h.photo(130)).decode()
    body = {"image": image, "position": {"latitude": 43.6, "longitude": 1.44}}
    answer = client.post("/api/search", json=body)
    assert answer.status_code == 200
    assert answer.json()["nearby"] is None  # the test engine holds no places
    body["position"] = {"latitude": 123.0, "longitude": 1.44}
    assert client.post("/api/search", json=body).status_code == 422


def test_a_huge_body_is_refused_before_it_is_read(client: TestClient) -> None:
    """No session needed to send a body: the cap holds before login and before parsing."""
    too_big = app_module.MAX_BODY_BYTES + 1
    response = client.post(
        "/api/search", content=b"x" * too_big, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_a_huge_body_sent_in_chunks_is_cut_off(client: TestClient) -> None:
    chunk = b"x" * (1 << 20)
    count = app_module.MAX_BODY_BYTES // len(chunk) + 2

    def chunks() -> Iterator[bytes]:
        for _ in range(count):
            yield chunk

    response = client.post(
        "/api/login", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


class _BrokenBackend(LocalBackend):
    async def search(self, request: SearchRequest) -> SearchResponse:
        del request
        msg = "a7.npy has 5 rows, the catalog 6"
        raise ValueError(msg)


def test_a_server_fault_is_a_500_not_an_unreadable_photo(tmp_path: Path) -> None:
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=_BrokenBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
        ui_dir=built_page(tmp_path / "ui"),
    )
    broken = TestClient(app, raise_server_exceptions=False)
    login(broken)
    image = base64.b64encode(h.photo(130)).decode()
    response = broken.post("/api/search", json={"image": image})
    assert response.status_code == 500
    assert "lue" not in response.json()["detail"]


class _SlowBackend(LocalBackend):
    """Each search takes `SLOW_S`, and is counted."""

    calls = 0

    async def search(self, request: SearchRequest) -> SearchResponse:
        type(self).calls += 1
        await asyncio.sleep(SLOW_S)
        return await super().search(request)


SLOW_S = 0.75


def test_a_retry_waits_for_the_search_already_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "SEARCH_WAIT_S", 0.5)
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=_SlowBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
        ui_dir=built_page(tmp_path / "ui"),
    )
    body = {"image": base64.b64encode(h.photo(130)).decode()}
    with TestClient(app) as slow:  # one event loop: the first call goes on between requests
        login(slow)
        assert slow.post("/api/search", json=body).status_code == 503
        assert slow.post("/api/search", json=body).status_code == 200  # joined the first call
        assert _SlowBackend.calls == 1
        other = {"image": base64.b64encode(h.photo(90)).decode()}
        assert slow.post("/api/search", json=other).status_code == 503
        assert _SlowBackend.calls == 2


class _Conversation:
    id = "c1"


class _QuietChat:
    """Opens conversations whose events stream ends at once."""

    def open(self, *args: object, **kwargs: object) -> _Conversation:
        del args, kwargs
        return _Conversation()

    def get(self, session: str, conversation_id: str) -> _Conversation:
        del session, conversation_id
        return _Conversation()

    async def stream(self, conversation: object, after: int) -> AsyncIterator[dict[str, object]]:
        del conversation, after
        for event in ():
            yield event


def _chat_client(tmp_path: Path) -> TestClient:
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=LocalBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
        chat=lambda _: cast("ChatService", _QuietChat()),
        ui_dir=built_page(tmp_path / "ui"),
    )
    return TestClient(app)


def test_new_conversations_are_capped_per_session_per_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "CONVERSATIONS_PER_DAY", 2)
    monkeypatch.setattr(auth, "SEARCHES_PER_MINUTE", 100)
    chat = _chat_client(tmp_path)
    login(chat)
    codes = [chat.post("/api/chat/open", json={"text": "un pont"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    chat.cookies.clear()
    login(chat)  # another session has its own count
    assert chat.post("/api/chat/open", json={"text": "un pont"}).status_code == 200


def test_event_streams_and_wake_ups_are_rate_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "CALLS_PER_MINUTE", 3)
    chat = _chat_client(tmp_path)
    login(chat)
    body = {"conversation_id": "c1"}
    codes = [chat.post("/api/chat/events", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 200]
    assert chat.post("/api/chat/events", json=body).status_code == 429
    assert chat.post("/api/warm").status_code == 429


def test_a_photo_too_large_to_decode_is_unreadable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    login(client)
    monkeypatch.setattr(upload, "MAX_UPLOAD_PIXELS", 1000)
    image = base64.b64encode(h.photo(130)).decode()  # 64 x 48 = 3,072 px
    response = client.post("/api/search", json={"image": image})
    assert response.status_code == 422
    assert response.json()["detail"] == "Cette image n’a pas pu être lue."


def test_a_photo_in_another_format_is_unreadable(client: TestClient) -> None:
    login(client)
    out = io.BytesIO()
    PILImage.new("RGB", (64, 48), "grey").save(out, format="GIF")
    image = base64.b64encode(out.getvalue()).decode()
    assert client.post("/api/search", json={"image": image}).status_code == 422
    assert client.post("/api/detect", json={"image": image}).status_code == 200  # fake outlines


@pytest.mark.parametrize("path", ["/api/search", "/api/detect", "/api/chat/open"])
def test_a_photo_field_has_a_length_cap(client: TestClient, path: str) -> None:
    login(client)
    body = {"image": "A" * (app_module.MAX_PHOTO_CHARS + 4)}
    assert client.post(path, json=body).status_code == 422
