"""When a visitor last used the site: one file whose change time is the answer.

The demo machine powers itself off after an hour with no visitor (`miroir.jobs.demo`). The
site marks each request that carries a valid session (`VisitMark.touch`, called by
`miroir.web.app`), so robots that probe every address on the internet, and the health checks
of `miroir demo up`, never keep the machine on. The watcher, in another container, reads the
same file (`last_visit`).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

#: The file is touched at most this often: a search burst writes once.
TOUCH_EVERY_S = 30.0


class VisitMark:
    """Touches `path` when a visitor uses the site, at most once every `every_s` seconds."""

    def __init__(
        self,
        path: Path,
        every_s: float = TOUCH_EVERY_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = path
        self._every_s = every_s
        self._clock = clock
        self._last: float | None = None

    def touch(self) -> None:
        now = self._clock()
        if self._last is not None and now - self._last < self._every_s:
            return
        self._last = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()


def last_visit(path: Path) -> float | None:
    """Unix time of the last visit, or None when nobody has come since the file was made."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None
