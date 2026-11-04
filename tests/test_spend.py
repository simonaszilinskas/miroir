from __future__ import annotations

import contextlib
import multiprocessing
import threading
import time
from contextlib import AbstractContextManager
from datetime import date
from pathlib import Path

import httpx
import pytest

from miroir.jobs.openrouter import OpenRouterError
from miroir.jobs.spend import (
    Bill,
    Budget,
    BudgetError,
    Probe,
    RunSpend,
    Service,
    SpendKind,
    SpendLedger,
    check_headroom,
    check_probe,
    check_run,
    load_probe,
    run_limit,
    save_probe,
    spent_usd,
)

BUDGET = Budget(cap_usd=50.0, reserve_usd=5.0, probe_cap_usd=2.0)
PROBE = Probe(
    option="A9",
    model="google/gemini-embedding-2",
    resolution=512,
    gallery="screening-publishable",
    images=100,
    cost_usd=0.012,
    created=date(2026, 10, 12),
)


def ledger(tmp_path: Path, *costs: tuple[SpendKind, float]) -> SpendLedger:
    led = SpendLedger(tmp_path / "spend.jsonl")
    for kind, cost in costs:
        led.record(kind, option="A9", run="r", model="m", cost_usd=cost, items=1)
    return led


def run_check(led: SpendLedger, key: float | None, probe: Probe | None, count: int) -> float:
    return check_run(
        BUDGET,
        led,
        key,
        probe,
        option="A9",
        model="google/gemini-embedding-2",
        resolution=512,
        count=count,
    )


def test_ledger_sums_by_kind(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.PROBE, 0.5), (SpendKind.RUN, 2.0))
    assert led.total() == pytest.approx(2.5)
    assert led.total(SpendKind.PROBE) == pytest.approx(0.5)


def test_run_needs_a_probe(tmp_path: Path) -> None:
    with pytest.raises(BudgetError, match="--probe"):
        run_check(ledger(tmp_path), None, None, 20_000)


def test_run_needs_a_probe_of_the_same_model_and_size(tmp_path: Path) -> None:
    stale = Probe("A9", "google/gemini-embedding-2", 1024, "g", 100, 0.01, date(2026, 10, 1))
    with pytest.raises(BudgetError, match="probe again"):
        run_check(ledger(tmp_path), None, stale, 20_000)


def test_run_projection_uses_cost_per_image(tmp_path: Path) -> None:
    assert run_check(ledger(tmp_path), None, PROBE, 20_000) == pytest.approx(2.4)


def test_run_that_would_pass_cap_minus_reserve_is_refused(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 43.0))
    with pytest.raises(BudgetError, match="usable budget"):
        run_check(led, None, PROBE, 20_000)  # 43 + 2.4 > 45


def test_key_usage_counts_when_higher_than_the_ledger(tmp_path: Path) -> None:
    with pytest.raises(BudgetError):
        run_check(ledger(tmp_path), 44.0, PROBE, 20_000)
    assert run_check(ledger(tmp_path), 1.0, PROBE, 20_000) == pytest.approx(2.4)


def test_probes_stop_at_their_cap(tmp_path: Path) -> None:
    assert check_probe(BUDGET, ledger(tmp_path, (SpendKind.PROBE, 1.5)), None) == pytest.approx(0.5)
    with pytest.raises(BudgetError, match="cap"):
        check_probe(BUDGET, ledger(tmp_path / "b", (SpendKind.PROBE, 2.0)), None)


def test_probe_round_trip(tmp_path: Path) -> None:
    save_probe(tmp_path / "A9.json", PROBE)
    assert load_probe(tmp_path / "A9.json") == PROBE
    assert load_probe(tmp_path / "none.json") is None


def test_total_spend_adds_modal_to_the_openrouter_figure(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 2.0))
    led.record(
        SpendKind.RUN, option="R0", run="r", model="m", cost_usd=1.0, items=1, service=Service.MODAL
    )
    assert led.total(service=Service.MODAL) == pytest.approx(1.0)
    assert spent_usd(led, None) == pytest.approx(3.0)
    assert spent_usd(led, 5.0) == pytest.approx(6.0)  # the key's count replaces OpenRouter's only


def test_old_ledger_lines_count_as_openrouter(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    path.write_text(
        '{"at":"2026-10-12T10:00:00Z","kind":"probe","option":"A9","run":"r","model":"m",'
        '"cost_usd":0.5,"items":100}\n'
    )
    assert SpendLedger(path).total(service=Service.OPENROUTER) == pytest.approx(0.5)


def test_a_run_limit_never_passes_what_the_budget_has_left(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 20.0))
    # $20 spent, $25 projected: 1.5 x 25 would end at $57.50, past the $45 usable budget.
    limit = run_limit(BUDGET, led, None, 25.0, factor=1.5)
    assert limit == pytest.approx(25.0)
    assert run_limit(BUDGET, led, None, 2.0, factor=1.5, margin_usd=0.25) == pytest.approx(3.25)
    assert run_limit(BUDGET, led, None, 2.0, factor=1.5, max_usd=2.5) == pytest.approx(2.5)


def hold(led: SpendLedger, worst: float, expected: float = 0.0) -> AbstractContextManager[Bill]:
    return led.hold(
        BUDGET,
        None,
        worst_usd=worst,
        expected_usd=expected,
        kind=SpendKind.RUN,
        option="A9",
        run="r",
        model="m",
        items=1,
    )


def test_a_call_in_flight_holds_its_worst_case_until_settled(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 40.0))
    with hold(led, 4.0) as bill:
        assert led.held_usd() == pytest.approx(4.0)
        with pytest.raises(BudgetError, match="in flight"), hold(led, 2.0):
            pass  # 40 + 4 held + 2 > 45: refused before anything is sent
        bill.cost_usd = 0.5
    assert led.held_usd() == 0.0
    assert led.total() == pytest.approx(40.5)
    with hold(led, 2.0) as bill:
        bill.cost_usd = 0.1


def _hold_in_another_process(path: Path, held: str, release: str) -> None:
    led = SpendLedger(path)
    with hold(led, 4.0) as bill:
        Path(held).touch()
        while not Path(release).exists():
            time.sleep(0.01)
        bill.cost_usd = 1.0


def test_holds_of_another_process_count(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 40.0))
    held, release = tmp_path / "held", tmp_path / "release"
    other = multiprocessing.get_context("spawn").Process(
        target=_hold_in_another_process, args=(led.path, str(held), str(release))
    )
    other.start()
    try:
        while not held.exists():
            time.sleep(0.01)
        with pytest.raises(BudgetError, match="in flight"), hold(led, 2.0):
            pass
    finally:
        release.touch()
        other.join()
    assert led.total() == pytest.approx(41.0)
    with hold(led, 2.0) as bill:
        bill.cost_usd = 0.0


def test_parallel_holds_never_pass_the_budget_together(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 40.0))
    gate = threading.Barrier(8)
    took: list[bool] = []

    def one() -> None:
        other = SpendLedger(led.path)  # each its own ledger, as each process would have
        gate.wait()
        try:
            with hold(other, 2.0) as bill:
                took.append(True)
                time.sleep(0.05)
                bill.cost_usd = 2.0
        except BudgetError:
            took.append(False)

    threads = [threading.Thread(target=one) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert took.count(True) == 2  # 40 + 2 + 2 <= 45; a third would pass it
    assert led.total() == pytest.approx(44.0)


def test_a_hold_of_a_stopped_process_no_longer_counts(tmp_path: Path) -> None:
    path = tmp_path / "spend.jsonl"
    led = SpendLedger(path)
    lines = b""
    with hold(led, 30.0) as bill:
        lines = path.read_bytes()
        bill.cost_usd = 0.0
    path.write_bytes(lines)  # as if the process died before settling
    assert SpendLedger(path).held_usd() == pytest.approx(30.0)
    assert SpendLedger(path, alive=lambda _pid: False).held_usd() == 0.0


def test_a_failed_call_is_settled_at_its_expected_cost_unless_refused(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    with contextlib.suppress(httpx.ReadTimeout), hold(led, 1.0, 0.25):
        raise httpx.ReadTimeout("no answer")  # it may have run and been billed
    with contextlib.suppress(OpenRouterError), hold(led, 1.0, 0.25):
        raise OpenRouterError("answered without a cost")  # billed, price unknown
    with contextlib.suppress(OpenRouterError), hold(led, 1.0, 0.25):
        raise OpenRouterError("HTTP 400", 400)  # refused: nothing billed
    settled = [line for line in led.lines() if not line.hold_usd]
    assert [(line.cost_usd, line.estimated) for line in settled] == [
        (0.25, True),
        (0.25, True),
        (0.0, False),
    ]
    assert led.held_usd() == 0.0


def test_a_run_stops_once_past_its_limit(tmp_path: Path) -> None:
    led = ledger(tmp_path)
    spend = RunSpend(
        led,
        BUDGET,
        None,
        limit_usd=0.5,
        per_item_usd=0.1,
        kind=SpendKind.RUN,
        option="A9",
        run="r",
        model="m",
    )
    for cost in (0.3, 0.3):
        with spend.call(3) as bill:
            bill.cost_usd = cost
    with pytest.raises(BudgetError, match="over its"), spend.call(3):
        pass
    assert spend.spent_usd == pytest.approx(0.6)


def test_headroom_refuses_a_run_whose_limit_could_pass_the_budget(tmp_path: Path) -> None:
    led = ledger(tmp_path, (SpendKind.RUN, 40.0))
    check_headroom(BUDGET, led, None, max_usd=5.0)
    with pytest.raises(BudgetError, match="usable budget"):
        check_headroom(BUDGET, led, None, max_usd=5.5)
    with pytest.raises(BudgetError, match=r"spent \$44\.00"):
        check_headroom(BUDGET, led, 44.0, max_usd=1.5)
