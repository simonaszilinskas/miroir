"""Checks for rules in CODING_STANDARDS.md that ruff and pyright cannot see."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SRC = ROOT / "src" / "miroir"
MODULES = sorted(p for p in SRC.rglob("*.py") if p.name != "config.py")
TESTS = sorted((ROOT / "tests").rglob("*.py"))


def _id(path: Path) -> str:
    return str(path.relative_to(SRC if SRC in path.parents else ROOT))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# Building a data path by hand, or hard-coding a machine path.
PATH_SMELLS = re.compile(r"expanduser|Path\.home|MIROIR_|/Users/|Downloads|miroir-data|\"~")
# Tests may name settings variables (they set them); they may not reach the machine's own paths.
TEST_PATH_SMELLS = re.compile(r"expanduser|Path\.home|/Users/|Downloads|miroir-data")
#: Tests of the defaults themselves.
TEST_PATH_ALLOWED = {"test_config.py", "test_pop_full_export.py", "test_conventions.py"}


def _smells(path: Path, pattern: re.Pattern[str]) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [
        f"{path.name}:{n}: {line.strip()}"
        for n, line in enumerate(lines, 1)
        if pattern.search(line)
    ]


@pytest.mark.parametrize("module", MODULES, ids=_id)
def test_only_config_builds_data_paths(module: Path) -> None:
    hits = _smells(module, PATH_SMELLS)
    assert not hits, "get paths from miroir.config instead:\n" + "\n".join(hits)


@pytest.mark.parametrize("module", TESTS, ids=_id)
def test_tests_use_no_machine_path(module: Path) -> None:
    hits = [] if module.name in TEST_PATH_ALLOWED else _smells(module, TEST_PATH_SMELLS)
    assert not hits, "use tmp_path or the settings fixture:\n" + "\n".join(hits)


#: Dataclasses that may change while one function runs. They never leave their module.
MUTABLE_ALLOWED = {("commons/build.py", "_Tally")}  # counts while one Commons set is gathered


def _dataclass_flags(node: ast.ClassDef) -> set[str] | None:
    """The keywords set True on the class's @dataclass decorator; None when it has none."""
    for deco in node.decorator_list:
        call = deco if isinstance(deco, ast.Call) else None
        name = call.func if call is not None else deco
        if isinstance(name, ast.Name) and name.id == "dataclass":
            keywords = call.keywords if call is not None else []
            return {
                k.arg
                for k in keywords
                if k.arg and isinstance(k.value, ast.Constant) and k.value.value is True
            }
    return None


@pytest.mark.parametrize("module", MODULES, ids=_id)
def test_dataclasses_are_frozen(module: Path) -> None:
    bare = [
        node.name
        for node in ast.walk(_tree(module))
        if isinstance(node, ast.ClassDef)
        and (flags := _dataclass_flags(node)) is not None
        and not {"frozen", "slots"} <= flags
        and (_id(module), node.name) not in MUTABLE_ALLOWED
    ]
    assert not bare, f"{_id(module)}: use @dataclass(frozen=True, slots=True) for {bare}"


# Modules that may open the sealed half, and how many times. Only R4 code reads it
# (CODING_STANDARDS 19), in eval/r4.py; eval/sets.py's write_set writes it. Passing on a
# parameter of the same name is plumbing.
SEALED_ALLOWED = {"eval/sets.py": 1, "eval/r4.py": 1}


def _sealed_functions() -> set[str]:
    """Functions that take an `allow_sealed` parameter."""
    return {
        node.name
        for module in MODULES
        for node in ast.walk(_tree(module))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(a.arg == "allow_sealed" for a in (*node.args.args, *node.args.kwonlyargs))
    } | {"testset_path"}


SEALED_FUNCTIONS = _sealed_functions()


def _sealed_uses(tree: ast.Module) -> list[int]:
    """Lines of calls that may open the sealed half: `allow_sealed=` anything but False or a
    same-named parameter, or `**kwargs` into a function taking `allow_sealed`."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        for k in node.keywords:
            forwarded = isinstance(k.value, ast.Name) and k.value.id == "allow_sealed"
            closed = isinstance(k.value, ast.Constant) and k.value.value is False
            if k.arg == "allow_sealed" and not (forwarded or closed):
                lines.append(node.lineno)
            if k.arg is None and name in SEALED_FUNCTIONS:
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("module", [*MODULES, SRC / "config.py"], ids=_id)
def test_only_r4_code_opens_the_sealed_half(module: Path) -> None:
    uses = _sealed_uses(_tree(module))
    allowed = SEALED_ALLOWED.get(_id(module), 0)
    assert len(uses) <= allowed, f"{_id(module)}:{uses}: only R4 code may open the sealed half"


@pytest.mark.parametrize(
    ("code", "opens"),
    [
        ("load_queries(s, 'T1', 'sealed', allow_sealed=True)", True),
        ("load_queries(s, 'T1', 'sealed', allow_sealed=flag)", True),
        ("load_queries(s, 'T1', 'sealed', **options)", True),
        ("load_queries(s, 'T1', 'tune', allow_sealed=False)", False),
        ("load_queries(s, 'T1', half, allow_sealed=allow_sealed)", False),
        ("print(**options)", False),
    ],
)
def test_the_sealed_half_check_sees_every_way_in(code: str, *, opens: bool) -> None:
    assert bool(_sealed_uses(ast.parse(code))) == opens
