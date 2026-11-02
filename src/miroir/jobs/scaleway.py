"""Scaleway GPU machines, driven from the Mac: start one, use it, always delete it.

A machine is a GPU OS instance (Ubuntu with the NVIDIA driver, Docker and the NVIDIA container
toolkit) with one block volume and a dynamic routed IPv4, made through the Instance API.
`gpu_machine` creates one and deletes it, with its volume and IP, when its block ends: on
success, on error, on Ctrl-C and on SIGTERM or SIGHUP. GPUs are often short ("scarce"), so it
tries each of `OFFERS` in turn. Every machine carries the `miroir` tag: `leftovers` lists what
a crashed Mac left behind in `ZONES`, and `delete_leftovers` removes it. The worker on the
machine also deletes it at its deadline (`miroir.remote.worker`).

Money: the price list below, in euros per hour without VAT (Scaleway's public prices of
2026-11; the API gives none). Scaleway bills a machine by the minute while it exists; each one
is written, with its whole life, to its own ledger (`Settings.scaleway_dir`), apart from the
OpenRouter and Modal budget, and a machine is refused when its longest life (`max_hours`)
would take the ledger past `scw_budget_eur`.
"""

from __future__ import annotations

import json
import logging
import math
import signal
import threading
import time
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import cast

import httpx

from miroir.config import Settings

logger = logging.getLogger(__name__)

API_URL = "https://api.scaleway.com"
#: Ubuntu 24.04 with the NVIDIA driver, Docker and the NVIDIA container toolkit.
IMAGE_LABEL = "ubuntu_noble_gpu_os_12"
#: Root volume: the Docker image (about 15 GB) and model weights.
ROOT_GB = 100
TAG = "miroir"
#: How long a machine may take to reach "running" once powered on.
BOOT_TIMEOUT_S = 15 * 60
#: How long deleting a machine may take before it is reported as left behind.
DELETE_TIMEOUT_S = 10 * 60
POLL_S = 5.0


@dataclass(frozen=True, slots=True)
class Offer:
    zone: str
    commercial_type: str
    eur_per_hour: float

    @property
    def eur_per_s(self) -> float:
        return self.eur_per_hour / 3600


#: Tried in order until one starts. L4 (24 GB) holds every model the jobs use.
OFFERS = (
    Offer("fr-par-2", "L4-1-24G", 0.75),
    Offer("pl-waw-2", "L4-1-24G", 0.75),
    Offer("fr-par-2", "L40S-1-48G", 1.40),
)
#: Zones `leftovers` looks in: every zone an offer names.
ZONES = tuple(dict.fromkeys(o.zone for o in OFFERS))
#: Euros to dollars, rounded up, for the run costs written beside Modal's (`RunInfo.cost_usd`).
EUR_TO_USD = 1.20

JSON = dict[str, object]


class ScalewayError(RuntimeError):
    """The API refused a request. `kind` is its error type, e.g. "out_of_stock"."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(f"Scaleway API {status} {kind}: {message}")
        self.status = status
        self.kind = kind


class NoMachineError(RuntimeError):
    """No offer could start a machine."""


def _out_of_stock(err: ScalewayError) -> bool:
    text = str(err).lower()
    return err.kind == "out_of_stock" or "out of stock" in text or "no capacity" in text


class InstanceApi:
    """The few Instance and Block Storage calls a machine needs. The key never leaves headers."""

    def __init__(self, settings: Settings, http: httpx.Client | None = None) -> None:
        if not settings.scaleway_configured:
            msg = "Scaleway is not set up: SCW_ACCESS_KEY, SCW_SECRET_KEY, SCW_DEFAULT_PROJECT_ID"
            raise RuntimeError(msg)
        assert settings.scw_secret_key is not None
        self.project = cast("str", settings.scw_project_id)
        self._http = http or httpx.Client(timeout=60.0)
        self._headers = {"X-Auth-Token": settings.scw_secret_key.get_secret_value()}

    def close(self) -> None:
        self._http.close()

    def _call(
        self,
        method: str,
        path: str,
        *,
        body: JSON | None = None,
        params: Mapping[str, str | int] | None = None,
        content: bytes | None = None,
    ) -> JSON:
        headers = dict(self._headers)
        if content is not None:
            headers["Content-Type"] = "text/plain"
        response = self._http.request(
            method, f"{API_URL}/{path}", json=body, params=params, content=content, headers=headers
        )
        if response.status_code >= 400:
            try:
                error = cast("JSON", response.json())
            except ValueError:
                error = {"message": response.text[:300]}
            kind = str(error.get("type", "unknown"))
            message = str(error.get("message", ""))
            if error.get("details"):
                message += f" {error['details']}"
            raise ScalewayError(response.status_code, kind, message)
        if not response.content:
            return {}
        return cast("JSON", response.json())

    def _pages(self, path: str, key: str, params: Mapping[str, str | int]) -> Iterator[JSON]:
        page = 1
        while True:
            found = cast(
                "list[JSON]", self._call("GET", path, params={**params, "page": page})[key]
            )
            yield from found
            if len(found) < int(params.get("per_page", 50)):
                return
            page += 1

    # --- servers ----------------------------------------------------------------------------

    def create_server(self, offer: Offer, name: str, tags: Sequence[str]) -> JSON:
        body: JSON = {
            "name": name,
            "commercial_type": offer.commercial_type,
            "image": IMAGE_LABEL,
            "project": self.project,
            "tags": list(tags),
            "dynamic_ip_required": True,
            "routed_ip_enabled": True,
            "volumes": {"0": {"boot": True, "size": ROOT_GB * 10**9, "volume_type": "sbs_volume"}},
        }
        path = f"instance/v1/zones/{offer.zone}/servers"
        return cast("JSON", self._call("POST", path, body=body)["server"])

    def set_user_data(self, zone: str, server_id: str, key: str, text: str) -> None:
        path = f"instance/v1/zones/{zone}/servers/{server_id}/user_data/{key}"
        self._call("PATCH", path, content=text.encode())

    def action(self, zone: str, server_id: str, action: str) -> None:
        path = f"instance/v1/zones/{zone}/servers/{server_id}/action"
        self._call("POST", path, body={"action": action})

    def server(self, zone: str, server_id: str) -> JSON | None:
        try:
            found = self._call("GET", f"instance/v1/zones/{zone}/servers/{server_id}")
        except ScalewayError as err:
            if err.status == 404:
                return None
            raise
        return cast("JSON", found["server"])

    def delete_server(self, zone: str, server_id: str) -> None:
        self._call("DELETE", f"instance/v1/zones/{zone}/servers/{server_id}")

    def servers(self, zone: str) -> list[JSON]:
        params = {"project": self.project, "per_page": 50}
        return list(self._pages(f"instance/v1/zones/{zone}/servers", "servers", params))

    # --- IPs and volumes --------------------------------------------------------------------

    def ips(self, zone: str) -> list[JSON]:
        params = {"project": self.project, "per_page": 50}
        return list(self._pages(f"instance/v1/zones/{zone}/ips", "ips", params))

    def delete_ip(self, zone: str, ip_id: str) -> None:
        _gone_ok(lambda: self._call("DELETE", f"instance/v1/zones/{zone}/ips/{ip_id}"))

    def block_volumes(self, zone: str) -> list[JSON]:
        params = {"project_id": self.project, "page_size": 100}
        path = f"block/v1/zones/{zone}/volumes"
        return cast("list[JSON]", self._call("GET", path, params=params)["volumes"])

    def delete_block_volume(self, zone: str, volume_id: str) -> None:
        path = f"block/v1/zones/{zone}/volumes/{volume_id}"
        _gone_ok(lambda: self._call("DELETE", path))

    def instance_volumes(self, zone: str) -> list[JSON]:
        params = {"project": self.project, "per_page": 50}
        return list(self._pages(f"instance/v1/zones/{zone}/volumes", "volumes", params))

    def delete_instance_volume(self, zone: str, volume_id: str) -> None:
        path = f"instance/v1/zones/{zone}/volumes/{volume_id}"
        _gone_ok(lambda: self._call("DELETE", path))


def _gone_ok(call: Callable[[], object]) -> None:
    try:
        call()
    except ScalewayError as err:
        if err.status != 404:
            raise


# --- one machine, from creation to deletion ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Machine:
    """A running GPU machine."""

    server_id: str
    name: str
    offer: Offer
    ip: str
    created: float
    """`time.time()` when it was created."""
    deadline: float
    """`time.time()` past which its worker deletes it."""


def cloud_init(public_key: str) -> str:
    """Lets the Mac in as root with this run's key, whatever the image's own key setup does."""
    return (
        "#cloud-config\n"
        "disable_root: false\n"
        "runcmd:\n"
        "  - mkdir -p /root/.ssh\n"
        f"  - echo '{public_key}' >> /root/.ssh/authorized_keys\n"
        "  - chmod 600 /root/.ssh/authorized_keys\n"
    )


@contextmanager
def gpu_machine(
    settings: Settings,
    api: InstanceApi,
    *,
    public_key: str,
    max_hours: float,
    offers: Sequence[Offer] = OFFERS,
    purpose: str = "",
    clock: Callable[[], float] = time.time,
) -> Generator[Machine]:
    """Start a GPU machine (first offer that has stock), yield it, delete it whatever happens."""
    ledger = MachineLedger(settings.scaleway_dir / "spend.jsonl")
    worst = max(o.eur_per_hour for o in offers) * max_hours
    spent = ledger.total_eur()
    if spent + worst > settings.scw_budget_eur:
        msg = (
            f"a machine may cost up to €{worst:.2f} ({max_hours} h) and €{spent:.2f} is spent; "
            f"together that passes the €{settings.scw_budget_eur:.2f} Scaleway cap"
        )
        raise RuntimeError(msg)
    name = f"{TAG}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    tags = [TAG, f"AUTHORIZED_KEY={public_key.replace(' ', '_')}"]
    with _signals_raise():
        errors: list[str] = []
        for offer in offers:
            created = clock()
            try:
                server = api.create_server(offer, name, tags)
            except ScalewayError as err:
                errors.append(f"{offer.zone} {offer.commercial_type}: {err}")
                if _out_of_stock(err):
                    logger.warning("%s %s: out of stock", offer.zone, offer.commercial_type)
                    continue
                raise
            server_id = str(server["id"])
            logger.info("%s: created %s in %s", name, offer.commercial_type, offer.zone)
            try:
                api.set_user_data(offer.zone, server_id, "cloud-init", cloud_init(public_key))
                try:
                    api.action(offer.zone, server_id, "poweron")
                except ScalewayError as err:
                    if not _out_of_stock(err):
                        raise
                    errors.append(f"{offer.zone} {offer.commercial_type}: {err}")
                    logger.warning(
                        "%s %s: out of stock at power on", offer.zone, offer.commercial_type
                    )
                    continue
                ip = _wait_running(api, offer.zone, server_id)
                machine = Machine(server_id, name, offer, ip, created, created + max_hours * 3600)
                logger.info("%s: running at %s", name, ip)
                yield machine
                return
            finally:
                delete_machine(api, offer.zone, server_id)
                ledger.record(name, offer, clock() - created, purpose)
        msg = "no GPU machine could start:\n" + "\n".join(errors)
        raise NoMachineError(msg)


def _wait_running(api: InstanceApi, zone: str, server_id: str) -> str:
    end = time.monotonic() + BOOT_TIMEOUT_S
    while time.monotonic() < end:
        server = api.server(zone, server_id)
        if server is None:
            msg = f"server {server_id} vanished while starting"
            raise RuntimeError(msg)
        ips = cast("list[JSON]", server.get("public_ips") or [])
        v4 = [str(i["address"]) for i in ips if i.get("family") == "inet"]
        if server.get("state") == "running" and v4:
            return v4[0]
        time.sleep(POLL_S)
    msg = f"server {server_id} not running after {BOOT_TIMEOUT_S} s"
    raise TimeoutError(msg)


def delete_machine(api: InstanceApi, zone: str, server_id: str) -> None:
    """Delete a server with its volumes and IPs, and wait until it is gone. Never skipped:
    on failure it logs what is left and raises, so `miroir scw ps` shows it."""
    server = api.server(zone, server_id)
    if server is None:
        return
    volumes = [
        (str(v["id"]), str(v.get("volume_type", "")))
        for v in cast("dict[str, JSON]", server.get("volumes") or {}).values()
    ]
    ips = [str(i["id"]) for i in cast("list[JSON]", server.get("public_ips") or [])]
    end = time.monotonic() + DELETE_TIMEOUT_S
    while server is not None and time.monotonic() < end:
        state = server.get("state")
        try:
            if state == "running":
                api.action(zone, server_id, "terminate")
            elif state == "stopped":
                api.delete_server(zone, server_id)
        except ScalewayError as err:
            logger.warning("deleting %s (%s): %s; trying again", server_id, state, err)
        time.sleep(POLL_S)
        server = api.server(zone, server_id)
    if server is not None:
        msg = (
            f"server {server_id} in {zone} still there after {DELETE_TIMEOUT_S} s: "
            "run `miroir scw cleanup`"
        )
        raise RuntimeError(msg)
    for volume_id, kind in volumes:
        _retry(lambda v=volume_id, k=kind: _delete_volume(api, zone, v, k))
    for ip_id in ips:
        _retry(lambda i=ip_id: api.delete_ip(zone, i))
    logger.info(
        "deleted server %s with %d volume(s) and %d IP(s)", server_id, len(volumes), len(ips)
    )


def _delete_volume(api: InstanceApi, zone: str, volume_id: str, kind: str) -> None:
    if kind == "sbs_volume":
        api.delete_block_volume(zone, volume_id)
    else:
        api.delete_instance_volume(zone, volume_id)


def _retry(call: Callable[[], None], attempts: int = 12) -> None:
    """Volumes stay "in use" for a little while after their server goes."""
    for attempt in range(attempts):
        try:
            call()
            return
        except ScalewayError as err:
            if attempt == attempts - 1:
                raise
            logger.info("not yet: %s", err)
            time.sleep(POLL_S)


@contextmanager
def _signals_raise() -> Generator[None]:
    """SIGTERM and SIGHUP raise KeyboardInterrupt, as Ctrl-C does, so cleanup runs."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def interrupt(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    before = {s: signal.signal(s, interrupt) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for s, handler in before.items():
            signal.signal(s, handler)


# --- what is left behind ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Leftover:
    zone: str
    kind: str
    """"server", "volume" or "ip"."""
    id: str
    label: str


def leftovers(api: InstanceApi, zones: Sequence[str] = ZONES) -> list[Leftover]:
    """Servers tagged `miroir`, and volumes and IPs attached to no server, in these zones."""
    found: list[Leftover] = []
    for zone in zones:
        for server in api.servers(zone):
            if TAG in cast("list[str]", server.get("tags") or []):
                label = (
                    f"{server.get('name')} {server.get('commercial_type')} {server.get('state')}"
                )
                found.append(Leftover(zone, "server", str(server["id"]), label))
        for volume in api.block_volumes(zone):
            if not volume.get("references"):
                size = cast("int", volume.get("size", 0)) // 10**9
                label = f"block {volume.get('name')} {size} GB"
                found.append(Leftover(zone, "volume", str(volume["id"]), label))
        for volume in api.instance_volumes(zone):
            if not volume.get("server"):
                label = f"{volume.get('volume_type')} {volume.get('name')}"
                found.append(Leftover(zone, "instance-volume", str(volume["id"]), label))
        for ip in api.ips(zone):
            if not ip.get("server"):
                found.append(Leftover(zone, "ip", str(ip["id"]), str(ip.get("address"))))
    return found


def delete_leftovers(api: InstanceApi, found: Sequence[Leftover]) -> None:
    """Servers first (with their volumes and IPs), then what is still loose."""
    for item in found:
        if item.kind == "server":
            delete_machine(api, item.zone, item.id)
    for item in found:
        if item.kind == "volume":
            _retry(lambda i=item: api.delete_block_volume(i.zone, i.id))
        elif item.kind == "instance-volume":
            _retry(lambda i=item: api.delete_instance_volume(i.zone, i.id))
        elif item.kind == "ip":
            api.delete_ip(item.zone, item.id)


# --- money ------------------------------------------------------------------------------------


class MachineLedger:
    """One JSON line per machine: its life and what it cost, rounded up to the minute."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, name: str, offer: Offer, seconds: float, purpose: str) -> float:
        minutes = max(1, math.ceil(seconds / 60))
        cost = minutes * 60 * offer.eur_per_s
        line = {
            "at": datetime.now(UTC).isoformat(),
            "name": name,
            "zone": offer.zone,
            "type": offer.commercial_type,
            "minutes": minutes,
            "cost_eur": round(cost, 4),
            "purpose": purpose,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as out:
            out.write(json.dumps(line) + "\n")
        logger.info("%s: %d min on %s, €%.3f", name, minutes, offer.commercial_type, cost)
        return cost

    def total_eur(self) -> float:
        if not self._path.exists():
            return 0.0
        lines = self._path.read_text().splitlines()
        return sum(float(json.loads(line)["cost_eur"]) for line in lines if line.strip())
