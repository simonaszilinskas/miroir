"""The demo machine: one Scaleway GPU machine that serves the website, on only for demos.

Unlike the job machines (`miroir.jobs.scaleway.gpu_machine`), this one is made once and kept:

- `create_demo` reserves a flexible IP (the address, and so the URL, never changes) and makes
  the machine, an L4 in fr-par-2 with one 100 GB block volume. `install` then builds the job
  image on it, copies the index bundle and the model weights from their buckets onto that
  volume, writes the secrets into root-only files and starts three containers that Docker
  starts again at every boot: the site (`miroir.remote.site`), Caddy (HTTPS with a Let's
  Encrypt certificate for `<ip>.sslip.io`) and the watcher (`watch_idle`).
- `up` powers it on and waits until the page answers; `down` powers it off. Off, Scaleway
  bills only the volume and the IP (`rest_eur_per_hour`). GPUs are often all taken: when no
  L4 is free, `up` changes the machine's type, while it is off, to the next of
  `DEMO_OFFERS` (an L40S, then machines without a GPU, where the site searches more slowly
  and reads no captions) and puts it back on an L4 at the next `up` when one is free.
- The watcher powers the machine off after `Settings.demo_idle_minutes` with no visitor, or
  after `Settings.demo_max_hours` on, so a closed Mac never leaves it running. It holds the
  project's Scaleway key in a root-only file: the key could not be narrowed to this machine
  (the project key may not create IAM keys).

Money: each time on, and the storage between demos, goes to the machines' ledger
(`MachineLedger`) when a command sees it end (`settle`); `up` is refused when the longest
session (`demo_max_hours`) would take the ledger past `Settings.scw_budget_eur`.

Secrets go to the machine over SSH, on standard input, into files only root reads. They are
never on a command line, in cloud-init, or in a log.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

import httpx

from miroir.config import DEMO_INDEX, Settings
from miroir.jobs.scaleway import (
    DEMO_TAG,
    IMAGE_LABEL,
    POLL_S,
    TAG,
    InstanceApi,
    MachineLedger,
    Offer,
    ScalewayError,
    cloud_init,
    key_tag,
    out_of_stock,
)
from miroir.jobs.scw_run import Shell, build_image

logger = logging.getLogger(__name__)

DEMO_NAME = "miroir-demo"
#: L4: the site also runs the outline finder and the caption reader (Qwen3-VL 4B), and the
#: pop-all bundle keeps 19 GB of vectors in memory (the L4 machine has 48 GB).
DEMO_OFFER = Offer("fr-par-2", "L4-1-24G", 0.79, IMAGE_LABEL, root_gb=100)
#: The types `up` tries, in order, on the same disk (prices from the public catalogue,
#: 2026-11-03, rounded up). Without a GPU the site needs about 55 GB of memory (float32
#: gallery vectors). Left out: RENDER-S (P100, 16 GB: the models need 19 GB, EmbeddingGemma 2
#: needs bfloat16, which Pascal lacks, and it costs €1.22 an hour), and H100 (€2.73 an hour).
DEMO_OFFERS = (
    DEMO_OFFER,
    Offer("fr-par-2", "L40S-1-48G", 1.47, IMAGE_LABEL, root_gb=100),
    Offer("fr-par-2", "POP2-HM-12C-96G", 0.62, IMAGE_LABEL, root_gb=100),
    Offer("fr-par-2", "POP2-HM-16C-128G", 0.83, IMAGE_LABEL, root_gb=100),
)
#: Commercial types with a GPU (the rest run the site on the CPU).
GPU_TYPES = ("L4-", "L40S-", "H100-")


def has_gpu(commercial_type: str) -> bool:
    return commercial_type.startswith(GPU_TYPES)


def offer_of(commercial_type: str) -> Offer:
    """The offer of a type `up` may pick; an unknown one is priced as the dearest."""
    for offer in DEMO_OFFERS:
        if offer.commercial_type == commercial_type:
            return offer
    return max(DEMO_OFFERS, key=lambda o: o.eur_per_hour)


#: Block Storage (5K IOPS) and a zonal flexible IPv4, in euros an hour, from the public
#: catalogue (2026-11-03): €0.00013 per GB an hour, and €0.005 an hour for the IP.
VOLUME_EUR_PER_GB_HOUR = 0.00013
IP_EUR_PER_HOUR = 0.005
HOURS_PER_MONTH = 730

#: On the machine. The site listens on the loopback only; Caddy takes ports 80 and 443.
ROOT = "/opt/miroir"
SITE_PORT = 8000
IMAGE_TAG = "miroir-jobs"
CADDY_IMAGE = "caddy:2.10.2"
#: Folders under ROOT, and where the containers see them.
MOUNTS = {"web-index": "/web-index", "weights": "/weights", "state": "/state"}
#: Root-only files under ROOT: the Scaleway keys (watcher and bucket copies), the site's
#: secrets, and its session key, made once on the machine and never sent anywhere.
SCW_ENV = "env"
SITE_ENV = "site.env"
SESSION_ENV = "session.env"
#: Under state/: the last visit (`miroir.web.visits`), the idle limit for this session,
#: whether the containers were made for a GPU, and the conversation page's spending per day
#: and logs (`miroir.remote.site`).
LAST_VISIT = "last-visit"
IDLE_FILE = "idle-minutes"
HARDWARE_FILE = "hardware"
CHAT_DAYS = "chat-days.json"
CHAT_LOGS = "chat-logs"
CONTAINERS = ("miroir-site", "miroir-watch", "caddy")

#: How long powering on may take, and the site to load its models after that.
BOOT_TIMEOUT_S = 15 * 60
SITE_TIMEOUT_S = 20 * 60
STOP_TIMEOUT_S = 10 * 60
#: When no GPU is free at power-on, try again this many times, this far apart.
STOCK_TRIES = 10
STOCK_WAIT_S = 30.0
#: The watcher looks this often.
WATCH_S = 60.0


def hostname(address: str) -> str:
    """A name that resolves to the address, for the HTTPS certificate (sslip.io)."""
    return f"{address.replace('.', '-')}.sslip.io"


def site_url(address: str) -> str:
    return f"https://{hostname(address)}"


def rest_eur_per_hour(offer: Offer = DEMO_OFFER, extra_gb: int = 0) -> float:
    """What the machine costs while off: its volumes (`extra_gb` beside the root one) and
    its IP."""
    return (offer.root_gb + extra_gb) * VOLUME_EUR_PER_GB_HOUR + IP_EUR_PER_HOUR


# --- what the Mac remembers between commands ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class DemoState:
    """Times the ledger has not covered yet (Unix seconds)."""

    session_start: float | None = None
    """When the machine was powered on, while that session is not in the ledger yet."""
    rest_paid_until: float | None = None
    """The volume and IP are in the ledger up to here."""
    session_type: str | None = None
    """The commercial type of that session (the price it is billed at)."""
    crops_volume: str | None = None
    """The Block Storage volume holding the whole crop vectors (`demo crops`), if any."""
    extra_gb: int = 0
    """Its size: billed with the root volume, on or off."""
    index: str | None = None
    """The bundle the site serves, when installed by `install` or `crops` (`up` keeps it)."""


def read_state(path: Path) -> DemoState:
    if not path.exists():
        return DemoState()
    raw = cast("dict[str, float | str | None]", json.loads(path.read_text(encoding="utf-8")))
    start, rest, kind, volume = (
        raw.get("session_start"),
        raw.get("rest_paid_until"),
        raw.get("session_type"),
        raw.get("crops_volume"),
    )
    return DemoState(
        None if start is None else float(start),
        None if rest is None else float(rest),
        None if kind is None else str(kind),
        None if volume is None else str(volume),
        int(raw.get("extra_gb") or 0),
        None if raw.get("index") is None else str(raw.get("index")),
    )


def write_state(path: Path, state: DemoState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state)) + "\n", encoding="utf-8")


# --- the machine as the API sees it ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DemoMachine:
    server_id: str
    zone: str
    state: str
    """Scaleway's: "running", "stopped", "starting", "stopping", ..."""
    address: str | None
    changed: float | None
    """Unix time of its last change of state (Scaleway's `modification_date`)."""
    commercial_type: str = DEMO_OFFER.commercial_type

    @property
    def url(self) -> str | None:
        return None if self.address is None else site_url(self.address)


def find_demo(api: InstanceApi, zone: str = DEMO_OFFER.zone) -> DemoMachine | None:
    for server in api.servers(zone):
        if DEMO_TAG in cast("list[str]", server.get("tags") or []):
            return _machine(zone, server)
    return None


def _machine(zone: str, server: dict[str, object]) -> DemoMachine:
    ips = cast("list[dict[str, object]]", server.get("public_ips") or [])
    v4 = [str(i["address"]) for i in ips if i.get("family") == "inet"]
    changed = server.get("modification_date")
    return DemoMachine(
        server_id=str(server["id"]),
        zone=zone,
        state=str(server.get("state")),
        address=v4[0] if v4 else None,
        changed=None if changed is None else datetime.fromisoformat(str(changed)).timestamp(),
        commercial_type=str(server.get("commercial_type", DEMO_OFFER.commercial_type)),
    )


def wait_state(api: InstanceApi, machine: DemoMachine, want: str, timeout_s: float) -> DemoMachine:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        server = api.server(machine.zone, machine.server_id)
        if server is None:
            msg = f"the demo machine {machine.server_id} is gone"
            raise RuntimeError(msg)
        found = _machine(machine.zone, server)
        if found.state == want:
            return found
        time.sleep(POLL_S)
    msg = f"the demo machine is not {want} after {timeout_s:.0f} s"
    raise TimeoutError(msg)


# --- money -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Demo:
    """What the commands share: the API, the ledger and the Mac's notes."""

    settings: Settings
    api: InstanceApi
    clock: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep

    @property
    def ledger(self) -> MachineLedger:
        return MachineLedger(self.settings.scaleway_dir / "spend.jsonl")

    @property
    def state_path(self) -> Path:
        return self.settings.demo_dir / "state.json"

    def settle(self, machine: DemoMachine | None) -> DemoState:
        """Write to the ledger what has ended since the last command: the storage up to now,
        and the last session when the machine is off (it may have powered itself off)."""
        state = read_state(self.state_path)
        now = self.clock()
        if machine is None:
            return state
        if state.rest_paid_until is not None and now > state.rest_paid_until:
            hours = (now - state.rest_paid_until) / 3600
            self.ledger.record_cost(
                DEMO_NAME,
                machine.zone,
                f"block {DEMO_OFFER.root_gb + state.extra_gb} GB + flexible IP",
                minutes=math.ceil(hours * 60),
                cost=hours * rest_eur_per_hour(extra_gb=state.extra_gb),
                purpose="demo storage",
            )
        state = replace(state, rest_paid_until=now)
        if state.session_start is None and machine.state != "stopped":
            # Powered on elsewhere (the console): count from its last change, or from now.
            state = replace(
                state, session_start=machine.changed or now, session_type=machine.commercial_type
            )
        if state.session_start is not None and machine.state == "stopped":
            end = machine.changed if machine.changed is not None else now
            end = min(max(end, state.session_start), now)
            offer = offer_of(state.session_type or machine.commercial_type)
            self.ledger.record(DEMO_NAME, offer, end - state.session_start, "demo")
            state = replace(state, session_start=None, session_type=None)
        write_state(self.state_path, state)
        return state

    def check_budget(self, hours: float, offer: Offer = DEMO_OFFER) -> None:
        spent = self.ledger.total_eur() + self.ledger.held_eur()  # machines alive in other runs
        worst = hours * offer.eur_per_hour
        cap = self.settings.scw_budget_eur
        if spent + worst > cap:
            msg = (
                f"a demo may cost up to €{worst:.2f} ({hours:g} h) and €{spent:.2f} is spent; "
                f"together that passes the €{cap:.2f} Scaleway cap (the scw_budget_eur setting)"
            )
            raise RuntimeError(msg)

    def session_cost(self, state: DemoState) -> float:
        """The open session's cost so far, rounded up to the minute as Scaleway bills."""
        if state.session_start is None:
            return 0.0
        minutes = max(1, math.ceil((self.clock() - state.session_start) / 60))
        return minutes * 60 * offer_of(state.session_type or "").eur_per_s

    # --- power -----------------------------------------------------------------------------

    def power_on(
        self, machine: DemoMachine, *, allow_cpu: bool = True, only: str | None = None
    ) -> DemoMachine:
        """Power on and wait until Scaleway says it runs.

        Tries each of `DEMO_OFFERS` in order (without a GPU only with `allow_cpu`), skipping
        those whose longest session would pass the cap, changing the machine's type while it
        is off; when none starts, tries again every `STOCK_WAIT_S`, `STOCK_TRIES` times.
        `only` tries that one type (to check how the site runs on it)."""
        self.settle(machine)
        if machine.state in {"running", "starting"}:
            return wait_state(self.api, machine, "running", BOOT_TIMEOUT_S)
        if machine.state != "stopped":
            msg = f"the demo machine is {machine.state}: wait a minute and try again"
            raise RuntimeError(msg)
        offers = [
            o
            for o in DEMO_OFFERS
            if (allow_cpu or has_gpu(o.commercial_type)) and only in {None, o.commercial_type}
        ]
        if not offers:
            msg = f"{only} is not one of {', '.join(o.commercial_type for o in DEMO_OFFERS)}"
            raise ValueError(msg)
        affordable = [o for o in offers if self._fits(o)]
        if not affordable:
            self.check_budget(self.settings.demo_max_hours, offers[0])  # raises, with the sums
        self._keep_key(machine)
        current = machine.commercial_type
        for attempt in range(STOCK_TRIES):
            for offer in affordable:
                if offer.commercial_type != current:
                    self.api.set_type(machine.zone, machine.server_id, offer.commercial_type)
                    current = offer.commercial_type
                try:
                    self.api.action(machine.zone, machine.server_id, "poweron")
                except ScalewayError as err:
                    if not out_of_stock(err):
                        raise
                    logger.warning("%s: none free", offer.commercial_type)
                    continue
                logger.info("powering on as %s (€%.2f an hour)", current, offer.eur_per_hour)
                started = replace(
                    read_state(self.state_path), session_start=self.clock(), session_type=current
                )
                write_state(self.state_path, started)
                return wait_state(self.api, machine, "running", BOOT_TIMEOUT_S)
            if attempt < STOCK_TRIES - 1:
                logger.warning("no machine free yet; trying again in %.0f s", STOCK_WAIT_S)
                self.sleep(STOCK_WAIT_S)
        names = ", ".join(o.commercial_type for o in affordable)
        msg = (
            f"none of {names} free in {machine.zone} for "
            f"{STOCK_TRIES * STOCK_WAIT_S / 60:.0f} min; try again later"
        )
        raise RuntimeError(msg)

    def _fits(self, offer: Offer) -> bool:
        try:
            self.check_budget(self.settings.demo_max_hours, offer)
        except RuntimeError:
            logger.info("%s left out: past the cap", offer.commercial_type)
            return False
        return True

    def _keep_key(self, machine: DemoMachine) -> None:
        """Make sure the boot writes the Mac's key: machines made before the tag lacked it."""
        server = self.api.server(machine.zone, machine.server_id) or {}
        tags = cast("list[str]", server.get("tags") or [])
        public = demo_key(self.settings).with_suffix(".pub").read_text().strip()
        if key_tag(public) not in tags:
            self.api.set_tags(machine.zone, machine.server_id, [*tags, key_tag(public)])

    def power_off(self, machine: DemoMachine) -> DemoMachine:
        """Power off, wait until it is off, and write the session to the ledger."""
        if machine.state == "running":
            self.api.action(machine.zone, machine.server_id, "poweroff")
        if machine.state != "stopped":
            machine = wait_state(self.api, machine, "stopped", STOP_TIMEOUT_S)
        self.settle(replace(machine, changed=self.clock()))
        return machine


def wait_site(
    url: str, timeout_s: float = SITE_TIMEOUT_S, http: httpx.Client | None = None
) -> float:
    """Until the page answers over HTTPS with a valid certificate. Returns the seconds waited.

    The page itself needs no session, so this never counts as a visit."""
    client = http or httpx.Client(timeout=10.0)
    start = time.monotonic()
    try:
        while time.monotonic() - start < timeout_s:
            try:
                if client.get(url).status_code == 200:
                    return time.monotonic() - start
            except httpx.HTTPError:
                pass
            time.sleep(POLL_S)
    finally:
        if http is None:
            client.close()
    msg = f"{url} did not answer after {timeout_s:.0f} s: see `miroir demo logs`"
    raise TimeoutError(msg)


# --- making the machine ------------------------------------------------------------------------


def create_demo(demo: Demo, public_key: str) -> DemoMachine:
    """Reserve the flexible IP, make the machine, power it on. Once per project."""
    api, zone = demo.api, DEMO_OFFER.zone
    if find_demo(api, zone) is not None:
        msg = "the demo machine exists already (`miroir demo status`)"
        raise RuntimeError(msg)
    demo.check_budget(demo.settings.demo_max_hours)
    tags = [TAG, DEMO_TAG, key_tag(public_key)]
    ip = api.create_ip(zone, tags)
    logger.info("flexible IP %s reserved", ip.get("address"))
    server = api.create_server(DEMO_OFFER, DEMO_NAME, tags, ip_id=str(ip["id"]))
    server_id = str(server["id"])
    api.set_user_data(zone, server_id, "cloud-init", cloud_init(public_key))
    now = demo.clock()
    first = DemoState(
        session_start=now, rest_paid_until=now, session_type=DEMO_OFFER.commercial_type
    )
    write_state(demo.state_path, first)
    api.action(zone, server_id, "poweron")
    machine = DemoMachine(server_id, zone, "starting", str(ip.get("address")), None)
    return wait_state(api, machine, "running", BOOT_TIMEOUT_S)


# --- what runs on the machine (`miroir demo install`) ------------------------------------------


def demo_key(settings: Settings) -> Path:
    """The demo machine's own SSH key, made on first use and kept (the machine is too)."""
    key = settings.demo_dir / "id_ed25519"
    if not key.exists():
        settings.demo_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        subprocess.run(  # noqa: S603
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", DEMO_NAME, "-f", str(key)],  # noqa: S607
            check=True,
        )
    return key


def demo_shell(settings: Settings, address: str) -> Shell:
    return Shell(address, demo_key(settings), settings.demo_dir / "known_hosts")


def served(demo: Demo, index: str) -> None:
    """Remember the bundle the site now serves."""
    write_state(demo.state_path, replace(read_state(demo.state_path), index=index))


def add_crops(demo: Demo, machine: DemoMachine, shell: Shell, index: str, gb: int) -> DemoState:
    """Give the demo machine the volume for `index`'s whole crop vectors (`crop_volume`),
    once: made and attached the first time, mounted again otherwise. The storage is settled
    first, so the ledger bills the new volume only from now."""
    from miroir.jobs.crop_volume import add_crop_volume, mount_point, mount_script  # noqa: PLC0415

    state = demo.settle(machine)
    mount = mount_point(f"{ROOT}/web-index", index)
    if state.crops_volume is None:
        volume = add_crop_volume(
            demo.api, shell, zone=machine.zone, server_id=machine.server_id, mount=mount,
            name=f"{DEMO_NAME}-crops", gb=gb,
        )  # fmt: skip
        state = replace(state, crops_volume=volume, extra_gb=gb)
        write_state(demo.state_path, state)
    else:
        shell.run(mount_script(state.crops_volume, mount), timeout=600)
    return state


def install(
    shell: Shell, settings: Settings, machine: DemoMachine, index: str = DEMO_INDEX
) -> None:
    """Build the image from this checkout, copy the bundle and the weights onto the disk,
    write the secrets and the Caddyfile, and (re)start the three containers.

    Run again after a code change: Docker reuses its cached layers and the copy skips files
    already there, so a second install takes a few minutes."""
    assert machine.address is not None
    build_image(shell, settings, page=True)  # also writes the Scaleway keys (`SCW_ENV`)
    start = time.monotonic()
    logger.info("copying %s and the model weights onto the disk", index)
    shell.run(fetch_command(index), timeout=60 * 60)
    logger.info("copied in %.0f s", time.monotonic() - start)
    shell.run(f"umask 077 && cat > {ROOT}/{SITE_ENV}", stdin=site_env(settings).encode())
    shell.run(session_env_command())
    shell.run(
        f"mkdir -p {ROOT}/caddy && cat > {ROOT}/caddy/Caddyfile",
        stdin=caddyfile(machine.address).encode(),
    )
    start_containers(shell, machine, settings, index, gpu=has_gpu(machine.commercial_type))


def reinstall(demo: Demo, machine: DemoMachine, shell: Shell, index: str) -> float:
    """`install` on the running machine, then wait for the site. Returns the seconds waited."""
    install(shell, demo.settings, machine, index)
    served(demo, index)
    assert machine.url is not None
    return wait_site(machine.url)


def create_site(demo: Demo, index: str) -> tuple[DemoMachine, float]:
    """Make the demo machine (`create_demo`) and install the site on it. Returns the machine
    and the seconds the site took to answer after the install."""
    public = demo_key(demo.settings).with_suffix(".pub").read_text().strip()
    machine = create_demo(demo, public)
    assert machine.address is not None
    logger.info("machine %s running at %s; installing", machine.server_id, machine.address)
    shell = demo_shell(demo.settings, machine.address)
    shell.wait()
    return machine, reinstall(demo, machine, shell, index)


@dataclass(frozen=True, slots=True)
class Up:
    """The demo machine powered on and its site answering (`bring_up`)."""

    machine: DemoMachine
    offer: Offer
    gpu: bool
    boot_s: float
    total_s: float
    idle_minutes: int


def bring_up(
    demo: Demo,
    machine: DemoMachine,
    *,
    idle_minutes: int | None,
    allow_cpu: bool,
    only: str | None,
    index: str | None,
) -> Up:
    """Power the machine on (`Demo.power_on`), fit the containers to its hardware, set its
    idle timer, and wait until the site answers."""
    start = time.monotonic()
    machine = demo.power_on(machine, allow_cpu=allow_cpu, only=only)
    if machine.address is None or machine.url is None:
        msg = f"the demo machine is {machine.state} after powering on"
        raise RuntimeError(msg)
    booted = time.monotonic() - start
    shell = demo_shell(demo.settings, machine.address)
    shell.wait()  # "running" comes a little before SSH answers
    bundle = index or read_state(demo.state_path).index or DEMO_INDEX
    gpu = fit_hardware(shell, machine, demo.settings, bundle)
    shell.run(idle_command(idle_minutes), timeout=60)
    wait_site(machine.url)
    return Up(
        machine,
        offer_of(machine.commercial_type),
        gpu,
        booted,
        time.monotonic() - start,
        idle_minutes or demo.settings.demo_idle_minutes,
    )


def minutes_since_visit(shell: Shell) -> float | None:
    """Minutes since the site's last visitor, or None when nobody came since it started."""
    seen = shell.run(f"stat -c %Y {ROOT}/state/{LAST_VISIT} 2>/dev/null || true", timeout=30)
    return (time.time() - float(seen)) / 60 if seen.strip() else None


def pull_chat_logs(settings: Settings, shell: Shell) -> tuple[list[Path], Path]:
    """Copy the conversation logs (JSON lines, no images) that are new or grew on the machine
    into `chat_logs_dir/hosted`. Returns the files written and that folder."""
    import shlex  # noqa: PLC0415

    from miroir.web.chatlog import pull  # noqa: PLC0415

    folder = f"{ROOT}/state/{CHAT_LOGS}"

    def names() -> list[str]:
        found = shell.run(f"cd {folder} 2>/dev/null && find . -name '*.jsonl' || true")
        return [line.removeprefix("./") for line in found.splitlines() if line.strip()]

    def read(name: str) -> bytes:
        return shell.run(f"cat {shlex.quote(f'{folder}/{name}')}").encode()

    target = settings.chat_logs_dir / "hosted"
    return pull(names, read, target), target


def fetch_bundle(settings: Settings, index: str) -> tuple[int, int]:
    """On the machine: copy the bundle and the weights from their buckets onto its disk.
    Returns the files copied of each."""
    from miroir.jobs.objects import (  # noqa: PLC0415
        Bucket,
        bucket_name,
        download_tree,
        link_siblings,
        s3_client,
    )

    s3 = s3_client(settings)
    root = Path(MOUNTS["web-index"])
    link_siblings(s3, bucket_name(settings, Bucket.INDEX), root, f"{index}/")
    got = download_tree(s3, bucket_name(settings, Bucket.INDEX), root, prefix=f"{index}/")
    weights = download_tree(s3, bucket_name(settings, Bucket.WEIGHTS), Path(MOUNTS["weights"]))
    return got, weights


def start_containers(
    shell: Shell, machine: DemoMachine, settings: Settings, index: str, *, gpu: bool
) -> None:
    """(Re)make the three containers for this hardware, and note which it was for."""
    for command in container_commands(machine, settings, index, gpu=gpu):
        shell.run(command, timeout=10 * 60)  # the first run pulls Caddy's image
    shell.run(f"echo {'gpu' if gpu else 'cpu'} > {ROOT}/state/{HARDWARE_FILE}")


def fit_hardware(shell: Shell, machine: DemoMachine, settings: Settings, index: str) -> bool:
    """After a boot: when the machine's type changed between GPU and CPU since the containers
    were made, make them again for it (the site's container asks Docker for the GPU).
    Returns whether the machine has a GPU."""
    gpu = has_gpu(machine.commercial_type)
    made_for = shell.run(f"cat {ROOT}/state/{HARDWARE_FILE} 2>/dev/null || echo gpu").strip()
    if made_for != ("gpu" if gpu else "cpu"):
        logger.info("containers made for %s, machine has %s: making them again", made_for,
                    "a GPU" if gpu else "no GPU")  # fmt: skip
        start_containers(shell, machine, settings, index, gpu=gpu)
    return gpu


def site_env(settings: Settings) -> str:
    """The site's settings and secrets as an env file (`Settings.demo_site_env`). Holds
    secrets: never log it."""
    return "".join(f"{k}={v}\n" for k, v in settings.demo_site_env().items())


def session_env_command() -> str:
    """Makes the session key on the machine, once: sessions survive restarts and deploys."""
    path = f"{ROOT}/{SESSION_ENV}"
    return (
        f"[ -s {path} ] || (umask 077 && "
        f"printf 'SESSION_SECRET=%s\\n' \"$(openssl rand -hex 32)\" > {path})"
    )


def caddyfile(address: str) -> str:
    """HTTPS for the machine's name, passed on to the site on the loopback. `flush_interval
    -1` sends the conversation page's stream as it comes."""
    return (
        f"{hostname(address)} {{\n"
        "\tencode gzip\n"
        f"\treverse_proxy 127.0.0.1:{SITE_PORT} {{\n"
        "\t\tflush_interval -1\n"
        "\t}\n"
        "}\n"
    )


def _volumes(*names: str, read_only: tuple[str, ...] = ()) -> str:
    return " ".join(f"-v {ROOT}/{n}:{MOUNTS[n]}{':ro' if n in read_only else ''}" for n in names)


def fetch_command(index: str = DEMO_INDEX) -> str:
    """Copies the index bundle and the model weights from their buckets onto the disk (only
    what is missing or changed)."""
    return (
        f"docker run --rm --env-file {ROOT}/{SCW_ENV} {_volumes('web-index', 'weights')} "
        f"--entrypoint miroir {IMAGE_TAG} demo fetch --index {index}"
    )


def container_commands(
    machine: DemoMachine, settings: Settings, index: str = DEMO_INDEX, *, gpu: bool = True
) -> list[str]:
    """The three containers, made again from scratch. `--restart unless-stopped` starts
    them at every boot. Without `gpu` the site's container asks for no GPU (Docker refuses
    to start it otherwise) and its models run on the CPU."""
    restart = "--restart unless-stopped"
    gpus = "--gpus all " if gpu else ""
    site = (
        f"docker run -d --name miroir-site {restart} {gpus}--network host --shm-size 8g "
        f"--env-file {ROOT}/{SITE_ENV} --env-file {ROOT}/{SESSION_ENV} "
        f"{_volumes('web-index', 'weights', 'state', read_only=('web-index',))} "
        f"--entrypoint miroir {IMAGE_TAG} demo serve "
        f"--index {MOUNTS['web-index']}/{index} --state {MOUNTS['state']} --port {SITE_PORT}"
    )
    watch = (
        f"docker run -d --name miroir-watch {restart} --env-file {ROOT}/{SCW_ENV} "
        f"{_volumes('state', read_only=('state',))} "
        f"--entrypoint miroir {IMAGE_TAG} demo watch "
        f"--server-id {machine.server_id} --zone {machine.zone} "
        f"--idle-minutes {settings.demo_idle_minutes} --max-hours {settings.demo_max_hours} "
        f"--state {MOUNTS['state']}"
    )
    caddy = (
        f"docker run -d --name caddy {restart} --network host "
        f"-v {ROOT}/caddy/Caddyfile:/etc/caddy/Caddyfile:ro "
        f"-v {ROOT}/caddy/data:/data -v {ROOT}/caddy/config:/config {CADDY_IMAGE}"
    )
    return [
        f"docker rm -f {' '.join(CONTAINERS)} >/dev/null 2>&1 || true",
        f"mkdir -p {ROOT}/state {ROOT}/caddy/data {ROOT}/caddy/config",
        site,
        watch,
        caddy,
    ]


def idle_command(minutes: int | None) -> str:
    """Sets this session's idle limit (the watcher reads it each minute); None restores the
    default by removing the file. The file lives in state/, so a reboot keeps it: `up`
    always writes or removes it."""
    path = f"{ROOT}/state/{IDLE_FILE}"
    return f"rm -f {path}" if minutes is None else f"echo {int(minutes)} > {path}"


# --- the watcher (runs on the machine, `miroir demo watch`) ------------------------------------


@dataclass(frozen=True, slots=True)
class WatchRule:
    idle_s: float
    max_s: float


def should_power_off(
    rule: WatchRule, *, now: float, booted: float, last_visit: float | None
) -> str | None:
    """Why the machine should power off now, or None. Idle time counts from the last visit,
    or from boot when nobody came."""
    if now - booted >= rule.max_s:
        return f"on for {rule.max_s / 3600:g} h"
    since = booted if last_visit is None else max(last_visit, booted)
    if now - since >= rule.idle_s:
        return f"no visitor for {rule.idle_s / 60:g} min"
    return None


def read_idle_minutes(path: Path, default: int) -> int:
    try:
        return max(1, int(path.read_text(encoding="utf-8").strip()))
    except (FileNotFoundError, ValueError):
        return default


def boot_time(uptime: Path = Path("/proc/uptime")) -> float:
    """When the machine booted (the container shares the host's uptime)."""
    return time.time() - float(uptime.read_text(encoding="utf-8").split()[0])


def watch_idle(
    settings: Settings,
    *,
    server_id: str,
    zone: str,
    state: Path,
    idle_minutes: int,
    max_hours: float,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    booted: float | None = None,
    api: InstanceApi | None = None,
) -> str:
    """Power this machine off when `should_power_off` says so. Returns the reason."""
    from miroir.web.visits import last_visit  # noqa: PLC0415

    started = boot_time() if booted is None else booted
    while True:
        idle = read_idle_minutes(state / IDLE_FILE, idle_minutes)
        rule = WatchRule(idle * 60.0, max_hours * 3600.0)
        reason = should_power_off(
            rule, now=clock(), booted=started, last_visit=last_visit(state / LAST_VISIT)
        )
        if reason is not None:
            logger.warning("powering off: %s", reason)
            client = api or InstanceApi(settings)
            try:
                client.action(zone, server_id, "poweroff")
            except ScalewayError:
                logger.exception("power off refused; trying again in a minute")
                sleep(WATCH_S)
                continue
            return reason
        sleep(WATCH_S)
