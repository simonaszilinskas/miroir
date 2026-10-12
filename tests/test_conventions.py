"""Checks for rules in CODING_STANDARDS.md that ruff and pyright cannot see."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "miroir"
MODULES = sorted(p for p in SRC.rglob("*.py") if p.name != "config.py")

# Building a data path by hand, or hard-coding a machine path.
PATH_SMELLS = re.compile(r"expanduser|Path\.home|MIROIR_|/Users/|Downloads|miroir-data|\"~")


@pytest.mark.parametrize("module", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_only_config_builds_data_paths(module: Path) -> None:
    hits = [
        f"{module.name}:{n}: {line.strip()}"
        for n, line in enumerate(module.read_text(encoding="utf-8").splitlines(), start=1)
        if PATH_SMELLS.search(line)
    ]
    assert not hits, "get paths from miroir.config instead:\n" + "\n".join(hits)


@pytest.mark.parametrize("module", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_dataclasses_are_frozen(module: Path) -> None:
    text = module.read_text(encoding="utf-8")
    bare = re.findall(r"^@dataclass(?!\(frozen=True).*$", text, flags=re.MULTILINE)
    allowed = {"build.py"}  # BuildReport accumulates counts while a build runs
    if module.name not in allowed:
        assert not bare, f"{module.name}: use @dataclass(frozen=True, slots=True)"


# Modules that may pass allow_sealed=True. Only R4 code reads the sealed half (CODING_STANDARDS
# 19); eval/sets.py is here because write_set writes it.
SEALED_ALLOWED = {"eval/sets.py"}


@pytest.mark.parametrize("module", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_only_r4_code_opens_the_sealed_half(module: Path) -> None:
    text = module.read_text(encoding="utf-8")
    uses = len(re.findall(r"[(,]\s*allow_sealed\s*=\s*True", text))  # calls, not docs
    allowed = 1 if str(module.relative_to(SRC)) in SEALED_ALLOWED else 0
    assert uses <= allowed, f"{module.name}: only R4 code may pass allow_sealed=True"
