"""The sealed halves stay out of reach outside building them and round R4 (CLAUDE.md rule 4).

`load_queries(..., allow_sealed=True)` is checked in test_conventions. These checks close the
other ways in: the in-memory draw `build_sets` returns (both halves), a `SplitSet`'s sealed
half, the sealed folder itself, and the T7 writer's saved answers (sealed T7 texts).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "miroir"
MODULES = sorted(SRC.rglob("*.py"))

#: Code that builds sets: it draws both halves at once, and may hold the sealed one.
BUILDERS = {"eval/build_sets.py", "eval/words.py", "eval/parts.py", "eval/sets.py"}
#: The one module that reads a sealed half, once R4 is frozen.
R4 = {"eval/r4.py"}
#: `.sealed` also names the count of sealed queries in build reports, which the CLI prints.
COUNTS = {"cli.py"}


def _name(module: Path) -> str:
    return str(module.relative_to(SRC))


def _tree(module: Path) -> ast.Module:
    return ast.parse(module.read_text(encoding="utf-8"))


def _called(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
    return ""


@pytest.mark.parametrize("module", MODULES, ids=_name)
def test_only_set_builders_draw_both_halves(module: Path) -> None:
    calls = [
        n.lineno
        for n in ast.walk(_tree(module))
        if isinstance(n, ast.Call) and _called(n) == "build_sets"
    ]
    if _name(module) not in BUILDERS:
        assert not calls, f"{_name(module)}:{calls}: build_sets holds the sealed halves"


@pytest.mark.parametrize("module", MODULES, ids=_name)
def test_only_set_builders_touch_a_sealed_half(module: Path) -> None:
    uses = [
        n.lineno
        for n in ast.walk(_tree(module))
        if isinstance(n, ast.Attribute) and n.attr == "sealed"
    ]
    if _name(module) not in BUILDERS | R4 | COUNTS:
        assert not uses, f"{_name(module)}:{uses}: reads a sealed half outside its builders"


@pytest.mark.parametrize("module", MODULES, ids=_name)
def test_only_config_names_the_sealed_folder(module: Path) -> None:
    named = [
        n.lineno
        for n in ast.walk(_tree(module))
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "sealed" in n.value
        and ("testsets-sealed" in n.value or n.value.endswith(".sealed.jsonl"))
    ]
    if _name(module) != "config.py":
        assert not named, f"{_name(module)}:{named}: the sealed folder's path is config's"


@pytest.mark.parametrize("module", MODULES, ids=_name)
def test_only_the_t7_build_reads_the_writer_s_answers(module: Path) -> None:
    # Those answers are the T7 texts of both halves; words.build_sets_from_answers splits them.
    reads = [
        n.lineno
        for n in ast.walk(_tree(module))
        if isinstance(n, ast.Call)
        and _called(n) == "vlm_answers_path"
        and any("WRITER" in ast.unparse(a) for a in n.args)
    ]
    if _name(module) != "eval/words.py":
        assert not reads, f"{_name(module)}:{reads}: reads the sealed T7 texts"


def test_the_checks_see_what_they_guard() -> None:
    # Each rule matches the code it allows, so a rename cannot make it vacuous.
    builders = _tree(SRC / "eval" / "words.py")
    assert any(_called(n) == "build_sets" for n in ast.walk(builders))
    assert any(_called(n) == "vlm_answers_path" for n in ast.walk(builders))
    assert any(
        isinstance(n, ast.Attribute) and n.attr == "sealed"
        for n in ast.walk(_tree(SRC / "eval" / "build_sets.py"))
    )
