"""Round R4: frozen once, refused on a changed bundle, scored with the site's ranking."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from miroir.config import Settings
from miroir.eval import r4
from miroir.eval.sets import Query, write_set
from miroir.web.levels import Levels, RouteLevels
from miroir.web.measure import IDS_FILE
from miroir.web.routing import GUESS_FILE, ROUTING_FILE
from tests import test_measure as m


def _bundle(settings: Settings, index: str = "full-f2") -> Path:
    folder = settings.web_index_dir(index)
    folder.mkdir(parents=True)
    for name in (ROUTING_FILE, GUESS_FILE, IDS_FILE, "manifest.json"):
        (folder / name).write_text(name)
    return folder


def _freeze(settings: Settings, path: Path) -> r4.R4Config:
    return r4.freeze(
        settings,
        ["full-f2"],
        a7_option="A7-embeddinggemma-2-512",
        a3_option="A3-megaloc-322",
        path=path,
        today=date(2026, 11, 4),
    )


def test_r4_is_frozen_once_and_names_the_sealed_query_runs(
    settings: Settings, tmp_path: Path
) -> None:
    _bundle(settings)
    path = tmp_path / "r4.json"
    config = _freeze(settings, path)
    assert r4.read_config(path) == config
    (stack,) = config.stacks
    assert set(stack.files) == {ROUTING_FILE, GUESS_FILE, IDS_FILE, "manifest.json"}
    assert stack.a7_query_run == "A7-embeddinggemma-2-512-T2-T2b-T8c-sealed-queries"
    with pytest.raises(FileExistsError):
        _freeze(settings, path)


def test_r4_refuses_a_bundle_changed_since_the_freeze(settings: Settings, tmp_path: Path) -> None:
    folder = _bundle(settings)
    (stack,) = _freeze(settings, tmp_path / "r4.json").stacks
    r4.check_stack(settings, stack)
    (folder / ROUTING_FILE).write_text("a cut-off moved after the freeze")
    with pytest.raises(ValueError, match=r"routing\.json"):
        r4.check_stack(settings, stack)
    (folder / "levels.json").write_text("{}")  # a file added since counts too
    with pytest.raises(ValueError, match=r"levels\.json"):
        r4.check_stack(settings, stack)


def test_only_a_frozen_set_s_sealed_half_opens(settings: Settings, tmp_path: Path) -> None:
    _bundle(settings)
    config = _freeze(settings, tmp_path / "r4.json")
    write_set(settings, "T2", [], [Query("T2-c", "T2", "PA1", "commons-c", ("PA1",), ())])
    write_set(settings, "T8c", [], [Query("T8c-d", "T8c", "PA9", "commons-d", (), ())])
    write_set(settings, "T2b", [], [Query("T2b-c", "T2b", "J1", "commons-c", ("J1",), ())])
    assert r4.sealed_query_images(settings, config) == (
        ["commons-c", "commons-d"],
        "T2-T2b-T8c-sealed-queries",
    )
    with pytest.raises(ValueError, match="frozen"):
        r4._sealed(settings, config, "PARTS")


def test_sealed_halves_are_scored_with_the_site_s_ranking() -> None:
    bundle, _, a7, a3 = m._pair(building=False)
    found = m._asked(a7, a3, 30.0, (2, 4))
    levels = Levels(routes=(RouteLevels("object", -1.0, -1.0, None, None),))
    scores, detail = r4.score_sealed(bundle, {"T5": found}, levels=levels, gallery="web-x")
    (t5,) = scores
    assert (t5.test_set, t5.half, t5.gallery) == ("T5", "sealed", "web-x")
    hits = m.site_search(bundle, found, top=r4.TOP)
    assert [q.rank for q in t5.per_query] == [h.right_rank for h in hits]
    assert detail["T5"]["very_close"] == 1.0  # every first card passes a threshold of -1
    assert detail["T5"]["answer_in_index"] == 2
