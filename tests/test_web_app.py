"""The website's API: login, sessions, rate limit, search, record pages, example photos."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from miroir.web import auth
from miroir.web.app import create_app
from miroir.web.fake import LocalBackend
from miroir.web.ops import read_env_password, write_env_password
from tests import web_helpers as h

PASSWORD = "test-mot-de-passe"


async def _fake_fetch(url: str) -> bytes:
    del url
    return h.photo(90)


@pytest.fixture
def client() -> TestClient:
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=LocalBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
        fetch=_fake_fetch,
    )
    return TestClient(app)


def login(client: TestClient) -> None:
    assert client.post("/api/login", json={"password": PASSWORD}).status_code == 200


def test_the_page_loads_without_login(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Miroir" in page.text
    assert "@gouvfr/dsfr@1.15.4" in page.text
    assert client.get("/static/app.js").status_code == 200


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
    assert page["photo_count"] == 3
    commons = [i for i in page["images"] if i["credit"]]
    assert commons[0]["credit"] == "Une autrice, CC BY-SA 4.0"
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
