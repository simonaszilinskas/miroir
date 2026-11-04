"""Each command imports only base packages and the extras that cover it (CODING_STANDARDS 4).

Each check imports modules in a fresh interpreter that refuses the packages of the extras left
out, as an install without them would.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "miroir"
FETCH = ("httpx", "PIL")
API = ("fastapi", "uvicorn", "starlette")
SCALEWAY = ("boto3", "botocore")

CHECK = """
import importlib, importlib.abc, sys

class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {refused!r}:
            raise ImportError(f"{{name}} is not installed")
        return None

sys.meta_path.insert(0, Refuse())
for name in {modules!r}:
    importlib.import_module(name)
"""


def _modules(*folders: str) -> list[str]:
    return [
        f"miroir.{folder}.{path.stem}".removesuffix(".__init__")
        for folder in folders
        for path in sorted((PACKAGE / folder).glob("*.py"))
    ]


def _imports(modules: list[str], refused: tuple[str, ...]) -> str:
    code = CHECK.format(refused=refused, modules=modules)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    return done.stderr


#: Base install: records, test-set building, the CLI itself.
BASE = [
    "miroir.cli",
    "miroir.config",
    *_modules("records"),
    "miroir.eval.build_sets",
    "miroir.eval.sets",
    "miroir.index.pca",
]
#: Base and `fetch`: Commons, geography, image fetches, OpenRouter, evaluation.
FETCH_ONLY = [*_modules("commons", "geo", "images", "index", "eval"), "miroir.jobs.openrouter"]


@pytest.mark.parametrize(
    ("modules", "refused"),
    [(BASE, (*FETCH, *API, *SCALEWAY)), (FETCH_ONLY, (*API, *SCALEWAY))],
    ids=["base", "base+fetch"],
)
def test_commands_need_only_their_extras(modules: list[str], refused: tuple[str, ...]) -> None:
    assert _imports(modules, refused) == ""
