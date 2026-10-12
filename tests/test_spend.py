from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from miroir.jobs.spend import (
    Budget,
    BudgetError,
    Probe,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    load_probe,
    save_probe,
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
