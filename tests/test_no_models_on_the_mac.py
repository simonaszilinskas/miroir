"""No ML model library is ever imported on the Mac (CLAUDE.md rule 1).

Ruff bans torch and friends everywhere except `miroir.remote`, the code that runs inside
Modal containers. These tests prove the rest of the package never pulls that code in: they
import the CLI and every other module in a fresh interpreter that refuses model imports.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src"
PACKAGE = SRC / "miroir"
REMOTE = PACKAGE / "remote"
BANNED = ("torch", "transformers", "timm", "open_clip", "sentence_transformers", "mlx")


def local_modules() -> list[str]:
    """Every module except the remote-only package."""
    names: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if REMOTE in path.parents:
            continue
        parts = path.relative_to(SRC).with_suffix("").parts
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


IMPORT_ALL = f"""
import importlib, importlib.abc, sys

BANNED = {BANNED!r}

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BANNED or name.startswith("miroir.remote"):
            raise ImportError(f"imported on the Mac: {{name}}")
        return None

sys.meta_path.insert(0, Refuse())
for name in {local_modules()!r}:
    importlib.import_module(name)
from miroir.cli import app
loaded = [m for m in sys.modules if m.split(".")[0] in BANNED or m.startswith("miroir.remote")]
assert not loaded, loaded
"""


def test_cli_and_every_local_module_import_without_model_libraries() -> None:
    done = subprocess.run(
        [sys.executable, "-c", IMPORT_ALL], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_the_modal_app_is_among_the_modules_checked() -> None:
    assert "miroir.jobs.modal_app" in local_modules()
    assert "miroir.cli" in local_modules()


def test_remote_code_is_imported_only_inside_functions() -> None:
    """Outside `miroir.remote`, an import of it must sit in a function body (run in Modal)."""
    top_level: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        if REMOTE in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("miroir.remote"):
                top_level.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Import) and any(
                a.name.startswith("miroir.remote") for a in node.names
            ):
                top_level.append(f"{path.name}:{node.lineno}")
    assert not top_level, top_level
