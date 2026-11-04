"""Scaleway machines: whatever happens inside the block, the machine is deleted."""

# ruff: noqa: ARG002 (the fakes take the API's arguments and ignore the zone)

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import cast

import pytest
from pydantic import SecretStr

from miroir.config import Settings
from miroir.jobs import scaleway
from miroir.jobs.objects import MAX_PARTS, MIN_PART_BYTES, part_bytes
from miroir.jobs.scaleway import (
    JSON,
    OFFERS,
    InstanceApi,
    MachineLedger,
    NoMachineError,
    Offer,
    ScalewayError,
    gpu_machine,
    leftovers,
)
from miroir.jobs.scw_run import build_context, torch_index, worker_env
from tests.helpers import FIXTURE_EXPORTS

L4 = Offer("fr-par-2", "L4-1-24G", 0.75)
WAW = Offer("pl-waw-2", "L4-1-24G", 0.75)


@pytest.fixture
def scw(tmp_path: Path) -> Settings:
    keys = {
        "scw_access_key": SecretStr("SCWTESTACCESS"),
        "scw_secret_key": SecretStr("test-secret"),
        "scw_project_id": "project",
        "hf_token": None,
    }
    return Settings(data_dir=tmp_path / "data", exports_dir=FIXTURE_EXPORTS).model_copy(update=keys)


class FakeApi:
    """Servers in memory. `stock` says which zones can power on a machine."""

    def __init__(self, stock: set[str]) -> None:
        self.stock = stock
        self.servers: dict[str, JSON] = {}
        self.deleted_volumes: list[str] = []
        self.deleted_ips: list[str] = []
        self._n = 0

    def create_server(self, offer: Offer, name: str, tags: list[str]) -> JSON:
        self._n += 1
        server: JSON = {
            "id": f"s{self._n}",
            "name": name,
            "state": "stopped",
            "zone": offer.zone,
            "tags": tags,
            "volumes": {"0": {"id": f"v{self._n}", "volume_type": "sbs_volume"}},
            "public_ips": [{"id": f"ip{self._n}", "address": "10.0.0.1", "family": "inet"}],
        }
        self.servers[server["id"]] = server  # pyright: ignore[reportArgumentType]
        return server

    def set_user_data(self, zone: str, server_id: str, key: str, text: str) -> None:
        assert key == "cloud-init"
        assert "ssh-ed25519" in text

    def action(self, zone: str, server_id: str, action: str) -> None:
        server = self.servers[server_id]
        if action == "poweron":
            if zone not in self.stock:
                raise ScalewayError(412, "out_of_stock", "out of stock")
            server["state"] = "running"
        elif action == "terminate":
            del self.servers[server_id]

    def server(self, zone: str, server_id: str) -> JSON | None:
        return self.servers.get(server_id)

    def delete_server(self, zone: str, server_id: str) -> None:
        del self.servers[server_id]

    def delete_block_volume(self, zone: str, volume_id: str) -> None:
        self.deleted_volumes.append(volume_id)

    def delete_instance_volume(self, zone: str, volume_id: str) -> None:
        self.deleted_volumes.append(volume_id)

    def delete_ip(self, zone: str, ip_id: str) -> None:
        self.deleted_ips.append(ip_id)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scaleway, "POLL_S", 0.0)


def test_a_machine_is_deleted_with_its_volume_and_ip_when_the_block_ends(scw: Settings) -> None:
    api = FakeApi({"fr-par-2"})
    with gpu_machine(scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=1) as m:
        assert m.ip == "10.0.0.1"
        assert api.servers
    assert not api.servers
    assert api.deleted_volumes == ["v1"]
    assert api.deleted_ips == ["ip1"]


@pytest.mark.parametrize("error", [RuntimeError("job failed"), KeyboardInterrupt()])
def test_a_machine_is_deleted_on_error_and_on_ctrl_c(scw: Settings, error: BaseException) -> None:
    api = FakeApi({"fr-par-2"})
    with (
        pytest.raises(type(error)),
        gpu_machine(scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=1),
    ):
        raise error
    assert not api.servers


def test_an_offer_out_of_stock_falls_back_to_the_next_and_leaves_nothing(scw: Settings) -> None:
    api = FakeApi({"pl-waw-2"})
    offers = (L4, WAW)
    with gpu_machine(
        scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=1, offers=offers
    ) as m:
        assert m.offer == WAW
        assert len(api.servers) == 1  # the fr-par-2 one is gone already
    assert not api.servers


def test_no_stock_anywhere_says_why(scw: Settings) -> None:
    api = FakeApi(set())
    with (
        pytest.raises(NoMachineError, match="out of stock"),
        gpu_machine(
            scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=1, offers=(L4, WAW)
        ),
    ):
        pass
    assert not api.servers


def test_each_machine_is_written_to_the_ledger_and_the_cap_refuses_more(scw: Settings) -> None:
    api = FakeApi({"fr-par-2"})
    with gpu_machine(scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=1):
        pass
    ledger = MachineLedger(scw.scaleway_dir / "spend.jsonl")
    assert ledger.total_eur() == pytest.approx(OFFERS[0].eur_per_hour / 60, abs=1e-4)  # 1 min
    with (
        pytest.raises(RuntimeError, match="Scaleway cap"),
        gpu_machine(scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=7),
    ):
        pass


def test_a_machine_alive_in_another_run_counts_against_the_cap(scw: Settings) -> None:
    api = FakeApi({"fr-par-2"})
    ledger = MachineLedger(scw.scaleway_dir / "spend.jsonl")
    cap = scw.scw_budget_eur
    with ledger.hold(cap - 1.0, cap):  # another run's machine, alive now
        with (
            pytest.raises(RuntimeError, match="alive now"),
            gpu_machine(scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=7),
        ):
            pass
        assert not api.servers  # refused before any machine was made
    assert ledger.held_eur() == 0.0
    with gpu_machine(scw, cast("InstanceApi", api), public_key="ssh-ed25519 A", max_hours=1):
        pass


def test_leftovers_are_miroir_servers_and_loose_volumes_and_ips() -> None:
    class Listing:
        def servers(self, zone: str) -> list[JSON]:
            return [
                {"id": "a", "tags": ["miroir"], "name": "miroir-1"},
                {"id": "b", "tags": ["other"], "name": "someone else's"},
            ]

        def block_volumes(self, zone: str) -> list[JSON]:
            return [{"id": "v1", "references": []}, {"id": "v2", "references": [{"id": "x"}]}]

        def instance_volumes(self, zone: str) -> list[JSON]:
            return []

        def ips(self, zone: str) -> list[JSON]:
            return [{"id": "i1", "server": None}, {"id": "i2", "server": {"id": "b"}}]

    found = leftovers(cast("InstanceApi", Listing()), ("fr-par-2",))
    assert [(f.kind, f.id) for f in found] == [("server", "a"), ("volume", "v1"), ("ip", "i1")]


def test_torch_wheels_follow_the_driver() -> None:
    assert torch_index(13.0).endswith("/cu130")
    assert torch_index(12.9).endswith("/cu128")
    with pytest.raises(RuntimeError):
        torch_index(12.2)


def test_the_build_context_holds_the_dockerfile_pins_and_package() -> None:
    with tarfile.open(fileobj=io.BytesIO(build_context()), mode="r:gz") as tar:
        names = set(tar.getnames())
        requirements = tar.extractfile("requirements.txt")
        assert requirements is not None
        pins = requirements.read().decode().split()
    assert {"Dockerfile", "pyproject.toml", "src/miroir/cli/__init__.py"} <= names
    assert not any("__pycache__" in n for n in names)
    assert "torch==2.14.1" in pins
    assert "boto3==1.43.123" in pins


def test_the_worker_env_holds_the_keys_and_nothing_unset(scw: Settings) -> None:
    lines = worker_env(scw).splitlines()
    assert "SCW_SECRET_KEY=test-secret" in lines
    assert not any(line.startswith("HF_TOKEN") for line in lines)


def test_upload_parts_stay_within_scaleways_limit() -> None:
    assert part_bytes(1) == MIN_PART_BYTES
    assert part_bytes(5 * 1024**3) == MIN_PART_BYTES
    for size in (19_069_036_928, 36_440_397_440):
        part = part_bytes(size)
        assert part % (1024 * 1024) == 0
        assert -(-size // part) <= MAX_PARTS
