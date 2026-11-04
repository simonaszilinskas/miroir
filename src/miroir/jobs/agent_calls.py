"""Model calls of the offline agentic search (step H5 on test sets): saved, replayed, paid
within the budget.

`CachedModel` is a `miroir.web.chat_model.ChatModel` around the site's own model, so the
evaluation sends what the site sends. Every answer is saved as it arrives, keyed by its
request (`CallCache`), so a rerun replays it at no cost and the conversation takes the same
path; a replayed answer keeps its cost, so the conversation's own cap acts the same too.
Every paid call is held against the project budget and settled with its real cost
(`miroir.jobs.spend.RunSpend`), and the run stops past its limit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter

from miroir.jobs.openrouter import JSON, ChatAnswer, ToolCall
from miroir.jobs.spend import BudgetError, RunSpend
from miroir.web.chat_model import ChatModel, ModelError

logger = logging.getLogger(__name__)

#: The most one call may cost, held against the budget while it runs: a long conversation's
#: history (60,000 tokens in) and the 3,000 tokens out the site allows.
WORST_CALL_USD = 0.05


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model answer, as saved. `cost_usd` is what it cost when it was paid for."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    key: str
    model: str
    query_id: str
    text: str
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost_usd: float
    finish_reason: str
    at: datetime
    thinking: str = ""

    def answer(self) -> ChatAnswer:
        return ChatAnswer(
            text=self.text,
            cost_usd=self.cost_usd,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            finish_reason=self.finish_reason,
            tool_calls=self.tool_calls,
            cached_tokens=self.cached_tokens,
            thinking=self.thinking,
        )


_RECORD = TypeAdapter(CallRecord)


class CallCache:
    """Answers by request key, in a JSON-lines file; the last line wins for a key."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._by_key: dict[str, CallRecord] = {}
        if path.exists():
            with path.open("rb") as lines:
                for line in lines:
                    if line.strip():
                        record = _RECORD.validate_json(line)
                        self._by_key[record.key] = record

    def get(self, key: str) -> CallRecord | None:
        with self._lock:
            return self._by_key.get(key)

    def put(self, record: CallRecord) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as out:
                out.write(_RECORD.dump_json(record) + b"\n")
                out.flush()
            self._by_key[record.key] = record


def request_key(model: str, body: Mapping[str, JSON]) -> str:
    """The same model and request (images as their data URLs) give the same key."""
    canonical = json.dumps({"model": model, "body": body}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class CachedModel:
    """One query's model calls: replayed from `cache`, else asked of `inner` and paid
    through `spend`. `calls` keeps every answer it gave, in order."""

    def __init__(
        self, inner: ChatModel, cache: CallCache, spend: RunSpend, *, query_id: str
    ) -> None:
        self._inner = inner
        self._cache = cache
        self._spend = spend
        self.query_id = query_id
        self.calls: list[CallRecord] = []
        self.replayed = 0
        self.stopped: BudgetError | None = None
        """Set when the run's limit or the budget refused a call: the run must stop."""

    @property
    def model(self) -> str:
        return self._inner.model

    async def complete(self, body: Mapping[str, JSON]) -> ChatAnswer:
        key = request_key(self.model, body)
        saved = self._cache.get(key)
        if saved is not None:
            self.calls.append(saved)
            self.replayed += 1
            return saved.answer()
        try:
            with self._spend.call(1, worst_usd=WORST_CALL_USD) as bill:
                try:
                    answer = await self._inner.complete(body)
                except ModelError as err:
                    if not err.maybe_billed:
                        bill.cost_usd = 0.0
                    raise
                bill.cost_usd = answer.cost_usd
        except BudgetError as err:
            self.stopped = err
            # The conversation reads it as a model that did not answer, and ends its turn.
            raise ModelError(str(err)) from err
        record = CallRecord(
            key=key,
            model=self.model,
            query_id=self.query_id,
            text=answer.text,
            tool_calls=answer.tool_calls,
            prompt_tokens=answer.prompt_tokens,
            completion_tokens=answer.completion_tokens,
            cached_tokens=answer.cached_tokens,
            cost_usd=answer.cost_usd,
            finish_reason=answer.finish_reason,
            at=datetime.now(UTC),
            thinking=answer.thinking,
        )
        self._cache.put(record)
        self.calls.append(record)
        return answer


@dataclass(frozen=True, slots=True)
class AgentKit:
    """What an offline run of the agent needs from outside: the site's model, the saved
    answers, the run's spend, and the images (`miroir.jobs.factory.open_agent_kit`)."""

    model: ChatModel
    cache: CallCache
    spend: RunSpend
    photo: Callable[[str], bytes]
    """A query photo's JPEG by image id; the caller checks it may go out first."""
    fetch: Callable[[str], Awaitable[bytes]]
    """A thumbnail's bytes by URL; the conversation's `ThumbSource` checks its id first."""
