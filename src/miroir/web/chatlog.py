"""What each conversation of the conversation page did (step H5), one JSON line per event.

One file per conversation: `<directory>/<day>/<conversation id>.jsonl`. A line holds the
visitor's words, each model reply (its words, tool calls, cost, time), each tool call (its
arguments and `why`, the ids it listed, its time), and the end of each turn. Never an image
(the visitor's photo, a framed part, a thumbnail), never a key, never a position: a `near`
filter's coordinates are written as "(position)".

On the Mac the files go to `Settings.chat_logs_dir`; on Modal to the index volume
(`config.MODAL_CHAT_LOGS`), committed at the end of each turn, and `miroir web chat-logs`
copies them back.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from miroir.jobs.openrouter import JSON

logger = logging.getLogger(__name__)

#: Conversation ids and days that may name a file.
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatLog(Protocol):
    def write(self, day: str, conversation: str, line: Mapping[str, JSON]) -> None: ...
    async def flush(self) -> None:
        """Make what was written durable (a Modal volume commit)."""
        ...


class NoLog:
    def write(self, day: str, conversation: str, line: Mapping[str, JSON]) -> None:
        del day, conversation, line

    async def flush(self) -> None:
        return None


class FileChatLog:
    """JSON lines in a folder; `commit`, when given, runs on each flush."""

    def __init__(
        self, directory: Path, commit: Callable[[], Awaitable[None]] | None = None
    ) -> None:
        self.directory = directory
        self._commit = commit
        self._dirty = False

    def path(self, day: str, conversation: str) -> Path:
        if not (_SAFE.match(day) and _SAFE.match(conversation)):
            msg = f"not a safe log name: {day!r}/{conversation!r}"
            raise ValueError(msg)
        return self.directory / day / f"{conversation}.jsonl"

    def write(self, day: str, conversation: str, line: Mapping[str, JSON]) -> None:
        stamped = {"t": datetime.now(UTC).isoformat(timespec="milliseconds"), **line}
        try:
            path = self.path(day, conversation)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        except (OSError, ValueError) as err:  # a log must never break a conversation
            logger.warning("chat log not written: %s", err)
            return
        self._dirty = True

    async def flush(self) -> None:
        if self._commit is None or not self._dirty:
            return
        self._dirty = False
        try:
            await self._commit()
        except Exception:
            self._dirty = True
            logger.warning("chat logs not committed", exc_info=True)


def redact(args: Mapping[str, object]) -> dict[str, JSON]:
    """A tool call's arguments as logged: a `near` point becomes "(position)"."""
    out: dict[str, JSON] = json.loads(json.dumps(dict(args), ensure_ascii=False, default=str))
    filters = out.get("filters")
    if isinstance(filters, dict) and isinstance(filters.get("near"), dict):
        filters["near"] = "(position)"
    return out


def pull(
    names: Callable[[], list[str]], read: Callable[[str], bytes], destination: Path
) -> list[Path]:
    """Copy log files from a store (the Modal volume) into `destination`, keeping their
    `<day>/<conversation>.jsonl` layout. `names` lists them, relative to the logs folder.
    Files already copied whole are skipped. Returns the files written."""
    written: list[Path] = []
    for name in sorted(names()):
        parts = Path(name).parts
        if len(parts) != 2 or not parts[1].endswith(".jsonl"):
            continue
        day, file_name = parts
        if not (_SAFE.match(day) and _SAFE.match(file_name.removesuffix(".jsonl"))):
            continue
        data = read(name)
        target = destination / day / file_name
        if target.exists() and target.read_bytes() == data:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(target)
    return written
