"""No ML model library is ever imported on the Mac (CLAUDE.md rule 1).

Ruff bans torch and friends everywhere except `miroir.remote`, the code that runs only on a
Scaleway GPU machine. These tests prove the rest of the package never pulls that code in: they
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


def test_the_gpu_machine_driver_is_among_the_modules_checked() -> None:
    assert "miroir.jobs.scw_run" in local_modules()
    assert "miroir.cli" in local_modules()


def _imports_remote(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        return (node.module or "").startswith("miroir.remote")
    if isinstance(node, ast.Import):
        return any(a.name.startswith("miroir.remote") for a in node.names)
    return False


def _hidden_command(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether a function is a CLI command registered with `hidden=True`: one the machine runs."""
    for decorator in fn.decorator_list:
        if isinstance(decorator, ast.Call) and any(
            k.arg == "hidden" and isinstance(k.value, ast.Constant) and k.value.value is True
            for k in decorator.keywords
        ):
            return True
    return False


def remote_imports_off_the_machine(path: Path) -> list[str]:
    """Imports of `miroir.remote` anywhere but in the body of a hidden CLI command."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed: set[int] = set()
    if path.parent.name == "cli":  # the `miroir` command package
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) and _hidden_command(fn):
                allowed.update(id(n) for n in ast.walk(fn))
    return [
        f"{path.name}:{node.lineno}"  # pyright: ignore[reportAttributeAccessIssue]
        for node in ast.walk(tree)
        if _imports_remote(node) and id(node) not in allowed
    ]


def test_remote_code_is_imported_only_by_the_machines_own_commands() -> None:
    """Outside `miroir.remote`, it may be imported only in the body of a hidden CLI command
    (`miroir scw serve`, `miroir demo serve`...), which only the GPU machine runs."""
    found: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        if REMOTE not in path.parents:
            found.extend(remote_imports_off_the_machine(path))
    assert not found, found


def test_an_import_in_a_mac_side_function_is_caught(tmp_path: Path) -> None:
    module = tmp_path / "cli" / "scw.py"
    module.parent.mkdir()
    module.write_text(
        "import typer\n"
        "app = typer.Typer()\n"
        "@app.command('serve', hidden=True)\n"
        "def serve():\n"
        "    from miroir.remote.worker import serve\n"
        "@app.command('embed')\n"
        "def embed():\n"
        "    from miroir.remote.models import encoder\n",
        encoding="utf-8",
    )
    assert remote_imports_off_the_machine(module) == ["scw.py:8"]
