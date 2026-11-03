"""The demo machine (`miroir.jobs.demo`): power, money, the idle watcher, what goes to the
machine, and the site's visit mark. A fake API in memory; no network, no SSH."""

# ruff: noqa: ARG002 (the fake takes the API's arguments and ignores the zone)

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from miroir.config import Settings
from miroir.jobs import demo as demo_module
from miroir.jobs import scaleway
from miroir.jobs.demo import (
    DEMO_OFFER,
    Demo,
    DemoMachine,
    WatchRule,
    caddyfile,
    container_commands,
    create_demo,
    find_demo,
    hostname,
    read_state,
    rest_eur_per_hour,
    should_power_off,
    site_env,
    watch_idle,
)
from miroir.jobs.scaleway import (
    JSON,
    InstanceApi,
    Leftover,
    MachineLedger,
    ScalewayError,
    delete_leftovers,
    leftovers,
)
from miroir.web.app import create_app
from miroir.web.fake import LocalBackend
from miroir.web.visits import VisitMark, last_visit
from tests import web_helpers as h
from tests.helpers import FIXTURE_EXPORTS

PASSWORD = "test-mot-de-passe"
HOUR = 3600.0


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    values = {
        "scw_access_key": SecretStr("SCWTESTACCESS"),
        "scw_secret_key": SecretStr("test-secret"),
        "scw_project_id": "project",
        "scw_budget_eur": 50.0,
        "app_password": SecretStr(PASSWORD),
        "openrouter_api_key": SecretStr("sk-or-test"),
        "mistral_api_key": None,
    }
    base = Settings(data_dir=tmp_path / "data", exports_dir=FIXTURE_EXPORTS)
    settings = base.model_copy(update=values)
    settings.demo_dir.mkdir(parents=True)
    (settings.demo_dir / "id_ed25519").write_text("private")
    (settings.demo_dir / "id_ed25519.pub").write_text("ssh-ed25519 AAAA demo\n")
    return settings


class Clock:
    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeApi:
    """One zone's servers and IPs in memory."""

    def __init__(self, *, stock: bool = True) -> None:
        self.stock = stock
        self.by_id: dict[str, JSON] = {}
        self.reserved: list[JSON] = []
        self.actions: list[tuple[str, str]] = []
        self.made_with_ip: str | None = None

    def create_ip(self, zone: str, tags: list[str]) -> JSON:
        ip: JSON = {"id": "ip1", "address": "51.15.1.2", "tags": tags, "server": None}
        self.reserved.append(ip)
        return ip

    def create_server(
        self, offer: scaleway.Offer, name: str, tags: list[str], *, ip_id: str | None = None
    ) -> JSON:
        self.made_with_ip = ip_id
        server: JSON = {
            "id": "s1",
            "name": name,
            "state": "stopped",
            "tags": tags,
            "public_ips": [{"id": ip_id, "address": "51.15.1.2", "family": "inet"}],
            "modification_date": "2026-11-03T10:00:00+00:00",
            "volumes": {"0": {"id": "v1", "volume_type": "sbs_volume"}},
        }
        self.by_id["s1"] = server
        return server

    def set_user_data(self, zone: str, server_id: str, key: str, text: str) -> None:
        assert "ssh-ed25519" in text

    def set_tags(self, zone: str, server_id: str, tags: list[str]) -> None:
        self.by_id[server_id]["tags"] = tags

    def action(self, zone: str, server_id: str, action: str) -> None:
        self.actions.append((server_id, action))
        if action == "poweron":
            if not self.stock:
                raise ScalewayError(412, "out_of_stock", "out of stock")
            self.by_id[server_id]["state"] = "running"
        elif action == "poweroff":
            self.by_id[server_id]["state"] = "stopped"

    def server(self, zone: str, server_id: str) -> JSON | None:
        return self.by_id.get(server_id)

    def servers(self, zone: str) -> list[JSON]:
        return list(self.by_id.values())

    def block_volumes(self, zone: str) -> list[JSON]:
        return [{"id": "v1", "references": [{"id": "s1"}], "size": 100 * 10**9}]

    def instance_volumes(self, zone: str) -> list[JSON]:
        return []

    def ips(self, zone: str) -> list[JSON]:
        return [{**ip, "server": {"id": "s1"}} for ip in self.reserved]


def _api(fake: FakeApi) -> InstanceApi:
    return cast("InstanceApi", fake)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaleway, "POLL_S", 0.0)
    monkeypatch.setattr(demo_module, "POLL_S", 0.0)


def made(settings: Settings, fake: FakeApi, clock: Clock) -> tuple[Demo, DemoMachine]:
    demo = Demo(settings, _api(fake), clock)
    machine = create_demo(demo, "ssh-ed25519 AAAA test")
    return demo, machine


def ledger(settings: Settings) -> list[dict[str, object]]:
    path = settings.scaleway_dir / "spend.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- making it ---------------------------------------------------------------------------------


def test_create_reserves_a_flexible_ip_and_tags_the_machine_as_the_demo(
    settings: Settings,
) -> None:
    fake = FakeApi()
    _, machine = made(settings, fake, Clock())
    assert fake.made_with_ip == "ip1"
    assert machine.state == "running"
    assert machine.url == "https://51-15-1-2.sslip.io"
    tags = set(cast("list[str]", fake.by_id["s1"]["tags"]))
    assert tags == {"miroir", "miroir-demo", "AUTHORIZED_KEY=ssh-ed25519_AAAA_test"}
    found = find_demo(_api(fake))
    assert found is not None
    assert found.server_id == "s1"


def test_cleanup_lists_the_demo_machine_but_never_deletes_it(settings: Settings) -> None:
    fake = FakeApi()
    made(settings, fake, Clock())
    found = leftovers(_api(fake), ("fr-par-2",))
    assert [f.kind for f in found] == ["demo"]  # its volume and IP are attached: not loose
    delete_leftovers(_api(fake), [Leftover("fr-par-2", "demo", "s1", "miroir-demo")])
    assert "s1" in fake.by_id
    assert ("s1", "terminate") not in fake.actions


# --- power and money ---------------------------------------------------------------------------


def test_a_session_goes_to_the_ledger_when_powered_off(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, machine = made(settings, fake, clock)
    clock.now += 90 * 60
    demo.power_off(machine)
    lines = ledger(settings)
    session = next(line for line in lines if line["purpose"] == "demo")
    assert session["minutes"] == 90
    assert session["cost_eur"] == pytest.approx(1.5 * DEMO_OFFER.eur_per_hour, abs=1e-3)
    storage = next(line for line in lines if line["purpose"] == "demo storage")
    assert storage["cost_eur"] == pytest.approx(1.5 * rest_eur_per_hour(), abs=1e-4)
    assert read_state(demo.state_path).session_start is None


def test_a_machine_that_powered_itself_off_is_billed_until_it_stopped(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, _ = made(settings, fake, clock)
    session_start = read_state(demo.state_path).session_start
    assert session_start is not None
    # The watcher powered it off 70 minutes in; the Mac looks a day later.
    fake.by_id["s1"]["state"] = "stopped"
    stopped = session_start + 70 * 60
    fake.by_id["s1"]["modification_date"] = _iso(stopped)
    clock.now += 24 * HOUR
    found = find_demo(_api(fake))
    assert found is not None
    demo.settle(found)
    session = next(line for line in ledger(settings) if line["purpose"] == "demo")
    assert session["minutes"] == 70


def test_power_on_adds_the_key_tag_the_boot_needs(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, machine = made(settings, fake, clock)
    machine = demo.power_off(machine)
    fake.by_id["s1"]["tags"] = ["miroir", "miroir-demo"]  # made before the tag existed
    demo.power_on(machine)
    assert "AUTHORIZED_KEY=ssh-ed25519_AAAA_demo" in cast("list[str]", fake.by_id["s1"]["tags"])


def test_power_on_is_refused_past_the_cap(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, machine = made(settings, fake, clock)
    machine = demo.power_off(machine)
    MachineLedger(settings.scaleway_dir / "spend.jsonl").record_cost(
        "earlier", "fr-par-2", "L40S", minutes=1, cost=46.0, purpose="embed"
    )
    with pytest.raises(RuntimeError, match="Scaleway cap"):
        demo.power_on(machine)
    assert fake.by_id["s1"]["state"] == "stopped"


def test_no_gpu_free_is_tried_again_then_said(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, machine = made(settings, fake, clock)
    machine = demo.power_off(machine)
    fake.stock = False
    waits: list[float] = []
    demo = Demo(settings, _api(fake), clock, sleep=waits.append)
    with pytest.raises(RuntimeError, match="no free L4-1-24G"):
        demo.power_on(machine)
    assert len(waits) == demo_module.STOCK_TRIES - 1


def test_a_gpu_freed_while_waiting_starts_the_machine(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, machine = made(settings, fake, clock)
    machine = demo.power_off(machine)
    fake.stock = False

    def wait(_: float) -> None:
        fake.stock = True

    demo = Demo(settings, _api(fake), clock, sleep=wait)
    assert demo.power_on(machine).state == "running"


def test_status_counts_the_open_session(settings: Settings) -> None:
    fake, clock = FakeApi(), Clock()
    demo, machine = made(settings, fake, clock)
    clock.now += 30 * 60 + 1
    state = demo.settle(machine)
    assert demo.session_cost(state) == pytest.approx(31 * 60 * DEMO_OFFER.eur_per_s)


# --- the watcher -------------------------------------------------------------------------------


RULE = WatchRule(idle_s=HOUR, max_s=8 * HOUR)


@pytest.mark.parametrize(
    ("now", "visit", "why"),
    [
        (30 * 60, None, None),  # booted half an hour ago, nobody yet
        (61 * 60, None, "no visitor for 60 min"),
        (2 * HOUR, 1.5 * HOUR, None),  # a visitor half an hour ago
        (2 * HOUR, -5 * HOUR, "no visitor for 60 min"),  # a visit from an earlier session
        (8 * HOUR, 8 * HOUR - 60, "on for 8 h"),
    ],
)
def test_the_watcher_powers_off_when_idle_or_on_too_long(
    now: float, visit: float | None, why: str | None
) -> None:
    assert should_power_off(RULE, now=now, booted=0.0, last_visit=visit) == why


def test_the_watcher_calls_power_off_and_reads_the_sessions_idle_limit(
    settings: Settings, tmp_path: Path
) -> None:
    fake, clock = FakeApi(), Clock(0.0)
    made(settings, fake, Clock())
    state = tmp_path / "state"
    state.mkdir()
    (state / "idle-minutes").write_text("5\n")

    def sleep(seconds: float) -> None:
        clock.now += seconds

    why = watch_idle(
        settings,
        server_id="s1",
        zone="fr-par-2",
        state=state,
        idle_minutes=60,
        max_hours=8,
        clock=clock,
        sleep=sleep,
        booted=0.0,
        api=_api(fake),
    )
    assert why == "no visitor for 5 min"
    assert clock.now == 5 * 60
    assert fake.actions[-1] == ("s1", "poweroff")


# --- what goes to the machine ------------------------------------------------------------------


def test_secrets_go_only_into_the_site_env_file(settings: Settings) -> None:
    machine = DemoMachine("s1", "fr-par-2", "running", "51.15.1.2", None)
    env = site_env(settings)
    assert f"APP_PASSWORD={PASSWORD}\n" in env
    assert "OPENROUTER_API_KEY=sk-or-test\n" in env
    assert "MISTRAL_API_KEY" not in env
    assert "SCW_SECRET_KEY" not in env  # the site never holds the Scaleway key
    commands = " ".join(container_commands(machine, settings))
    for secret in (PASSWORD, "sk-or-test", "test-secret"):
        assert secret not in commands
    assert "--env-file /opt/miroir/site.env" in commands
    assert "--network host" in commands


def test_caddy_serves_the_sslip_name_and_streams_the_conversation() -> None:
    text = caddyfile("51.15.1.2")
    assert text.startswith(f"{hostname('51.15.1.2')} {{")
    assert "reverse_proxy 127.0.0.1:8000" in text
    assert "flush_interval -1" in text


# --- the site's visit mark ---------------------------------------------------------------------


def test_the_visit_mark_is_touched_at_most_every_30_seconds(tmp_path: Path) -> None:
    clock = Clock(0.0)
    mark = VisitMark(tmp_path / "last-visit", clock=clock)
    assert last_visit(mark.path) is None
    mark.touch()
    os.utime(mark.path, (1.0, 1.0))
    clock.now = 10.0
    mark.touch()  # too soon: untouched
    assert last_visit(mark.path) == 1.0
    clock.now = 31.0
    mark.touch()
    assert last_visit(mark.path) != 1.0


def test_only_requests_with_a_session_count_as_visits() -> None:
    visits: list[int] = []
    engine, _ = h.engine(building=False)
    app = create_app(
        catalog=engine.catalog,
        backend=LocalBackend(engine),
        password=PASSWORD,
        session_key=b"k" * 32,
        secure_cookies=False,
        on_visit=lambda: visits.append(1),
    )
    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/api/me").status_code == 401
    assert not visits
    assert client.post("/api/login", json={"password": PASSWORD}).status_code == 200
    assert client.get("/api/me").status_code == 200
    assert visits == [1]


def _iso(when: float) -> str:
    from datetime import UTC, datetime  # noqa: PLC0415

    return datetime.fromtimestamp(when, UTC).isoformat()


def test_tool_search_gathers_its_filter_facts_ahead() -> None:
    from miroir.web.toolsearch import ToolSearch  # noqa: PLC0415

    engine, _ = h.engine(building=False)
    tools = ToolSearch(engine)
    tools.warm()
    facts = tools._facts
    assert facts is not None
    assert len(facts.bases) == engine.catalog.entity_count
