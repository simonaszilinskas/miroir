"""The `miroir` command: every group and command loads and documents itself, and the small
pieces of output the commands print come from the library."""

from __future__ import annotations

from collections import Counter

import pytest
import typer
from typer.testing import CliRunner

from miroir.cli import app
from miroir.eval.build_sets import BuildReport, SetReport
from miroir.eval.parts import Kind, PartsReport
from miroir.eval.scoring import Interval
from miroir.geo.model import PlaceSource
from miroir.geo.places import NO_HINT, PlacesReport
from miroir.jobs.bulk import parse_trials

GROUPS = ("records", "eval", "testsets", "embed", "index", "vlm", "words", "web", "geo", "scw")


def _paths(command: object, path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    found = [path]
    for name, child in getattr(command, "commands", {}).items():
        found += _paths(child, (*path, name))
    return found


COMMANDS = _paths(typer.main.get_command(app))


def test_the_root_lists_every_group() -> None:
    names = {p[0] for p in COMMANDS if len(p) == 1}
    assert {*GROUPS, "demo"} == names


@pytest.mark.parametrize("path", COMMANDS, ids=lambda p: " ".join(p) or "miroir")
def test_every_command_prints_its_help(path: tuple[str, ...]) -> None:
    done = CliRunner().invoke(app, [*path, "--help"])
    assert done.exit_code == 0, done.output
    assert "Usage:" in done.output


def test_an_interval_reads_in_percent() -> None:
    assert Interval(0.8125, 0.74, 0.881).percent() == "81.2 [74.0, 88.1]"


def test_trials_parse_batch_and_streams() -> None:
    assert parse_trials("32:1,64:2") == [(32, 1), (64, 2)]
    assert parse_trials("") == []


def test_the_build_report_flags_a_set_off_its_spec() -> None:
    sets = (
        SetReport("T1", tune=3, sealed=3, dropped=0, spec=6, closed_ok=6),
        SetReport("T5", tune=2, sealed=2, dropped=1, spec=9, closed_ok=4),
    )
    lines = BuildReport(sets, full=10, screening=8, needed=5, screening_publishable=6).lines()
    assert lines[1].startswith("T1")
    assert "differs" not in lines[1]
    assert lines[2].endswith("differs from spec §7")
    assert lines[-1] == (
        "galleries: full 10, screening 8 (5 needed by queries, 3 distractors), "
        "screening publishable 6"
    )


def test_the_parts_report_has_one_row_per_kind() -> None:
    kind = next(iter(Kind)).value
    report = PartsReport(Counter({kind: 2}), Counter({kind: 1}), Counter(rule=3), Counter())
    lines = report.lines()
    assert len(lines) == 2 + len(Kind)
    assert lines[1].split() == [kind, "2", "1", "0"]
    assert lines[-1] == "why: {'rule': 3}"


def test_the_places_report_ends_with_each_collection_s_coverage() -> None:
    counts = {("mh", PlaceSource.MERIMEE.value): 3, ("mh", NO_HINT): 1}
    lines = PlacesReport((), counts).lines()
    assert lines[-1] == "mh: 3 of 4 entities placed"
    assert len(lines) == 3
