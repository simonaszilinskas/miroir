"""The built page: is it there, and built from today's sources?

The page lives in `web-ui/` (React with `@codegouvfr/react-dsfr`, built by Vite with `make ui`)
and its build lands in `src/miroir/web/ui/`, outside git, so the package carries it and the
demo machine needs no Node. Vite's build writes `source.sha256`, a digest of the sources it
was built from, with the same rule as `source_digest` (`web-ui/vite.config.ts`). The demo
image is built from this checkout (`miroir.jobs.scw_run.build_context`), which refuses a page
that is missing or older than its sources.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: Files at the top of `web-ui/` that change the build.
TOP_FILES = ("index.html", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts")
#: Folders whose every file changes the build, but `public/dsfr` (copied from the DSFR
#: package by the build itself).
FOLDERS = ("src", "public")
SKIPPED = ("public/dsfr/",)
DIGEST_FILE = "source.sha256"


class PageNotBuiltError(RuntimeError):
    """The page is missing, or was built from other sources."""


def source_files(source: Path) -> list[str]:
    """The sources the build depends on, as sorted paths relative to `source`."""
    names = [name for name in TOP_FILES if (source / name).is_file()]
    for folder in FOLDERS:
        for path in (source / folder).rglob("*"):
            name = path.relative_to(source).as_posix()
            if path.is_file() and not name.startswith(SKIPPED) and path.name != ".DS_Store":
                names.append(name)
    return sorted(names)


def source_digest(source: Path) -> str:
    """SHA-256 over each source file's path and bytes, in path order."""
    digest = hashlib.sha256()
    for name in source_files(source):
        digest.update(name.encode() + b"\0")
        digest.update((source / name).read_bytes() + b"\0")
    return digest.hexdigest()


def check_built(source: Path, build: Path) -> None:
    """Raise `PageNotBuiltError` unless `build` holds the page built from `source` as it is."""
    if not (build / "index.html").is_file():
        msg = f"the page is not built in {build}: run `make ui`"
        raise PageNotBuiltError(msg)
    stamp = build / DIGEST_FILE
    built = stamp.read_text().strip() if stamp.is_file() else ""
    if built != source_digest(source):
        msg = f"the page in {build} is older than {source}: run `make ui`"
        raise PageNotBuiltError(msg)
