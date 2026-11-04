"""The search method of the agentic search (step H5), as a state machine with no I/O.

The first hosted test drifted away from close first results on guessed words, so the method
is enforced here and not only in the brief:

1. Look before you move: when the first look found something, no search runs before the
   model has looked at its top thumbnails (`view_results`).
2. Say why: every search and look carries a French `why` (checked in `miroir.web.chat_tools`).
3. Pivot from evidence: `more_like` searches from a record already seen.
4. Keep a best-so-far and stop: the model names its closest ids (`best`); after
   `MAX_SEARCHES` searches without a sure match, or `MAX_STALE` searches in a row that add
   nothing to the best-so-far, only the finishing tools run (`FINISHING`).
5. Honest end: `show_results` says `found` (exact, close, none).

`Method` holds the state of one conversation; `Turn`, the counters of one visitor message.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

from miroir.web.chat_prompts import (
    BEST_COUNT,
    BEST_LINE,
    BEST_NONE,
    LOCK_SEARCHES,
    LOCK_STALE,
    NOT_OFFERED,
    NOT_RUN_CAP,
    NOT_RUN_LOCKED,
    NOT_RUN_LOOK,
    NOT_RUN_VIEWS,
    NOT_RUN_WAITING,
)
from miroir.web.chat_tools import BEST, SEARCHES, ToolName, ToolOutcome, best_of

#: The step caps of one visitor message. Raised 2026-11-03 (from 8 searches, 3 without a
#: sure match, 2 stale, 3 looks) so a thorough "Approfondi" run, cropped and grey searches
#: included, can go on; the spending caps are the real limit.
#: Searches per visitor message, transformed ones included; then results or a question.
MAX_TOOL_CALLS = 20
#: Searches per visitor message without a sure match (a first-look card rated "Très
#: proche"); then only a question or the results.
MAX_SEARCHES = 10
#: Searches in a row that bring nothing new into the best-so-far; then the same stop.
MAX_STALE = 4
#: First-look results the model must look at before it searches.
LOOK_FIRST = 4
#: Looks at thumbnails per visitor message, apart from the searches (the smoke test with
#: Mistral spent its 8 calls on searches and could not check its picks by eye).
MAX_VIEWS = 6
#: The first look's results that start the best-so-far.
FIRST_BEST = 3

COUNTED = frozenset(t.value for t in SEARCHES)
"""The searches, by name: each counts towards the turn's searches."""
FINISHING = frozenset(
    {ToolName.ASK_VISITOR, ToolName.ASK_REGION, ToolName.SHOW_RESULTS, ToolName.READ_RECORD}
)
"""What still runs once the turn is locked or out of searches: no search, no look."""


class Turn:
    """What one visitor message has used so far (the method's counters; they change)."""

    __slots__ = ("lock", "searches", "stale", "unjudged", "used", "views")

    def __init__(self) -> None:
        self.searches = 0
        """Searches that ran."""
        self.used = 0
        """Searches asked for that were not refused, failed ones included."""
        self.views = 0
        """Looks at thumbnails."""
        self.stale = 0
        """Searches in a row that added nothing to the best-so-far."""
        self.unjudged: tuple[tuple[str, ...], tuple[str, ...]] | None = None
        """The last search's listed ids and the best-so-far before it, until judged."""
        self.lock: str | None = None
        """Why only a question or the results may come now."""


class Method:
    """The method's state over one conversation."""

    def __init__(self) -> None:
        self.best: list[str] = []
        """The closest ids so far, best first, as the model last named them."""
        self.sure: set[str] = set()
        """First-look entities the site rated "Très proche"."""
        self.must_view: tuple[str, ...] = ()
        """First-look entities to look at before any search (empty once looked)."""
        self.turn = Turn()

    def new_turn(self) -> None:
        self.turn = Turn()

    def first_look(self, ids: Sequence[str], sure: Iterable[str]) -> None:
        """The first look's results, best first, and those rated "Très proche"."""
        self.sure = set(sure)
        self.must_view = tuple(ids[:LOOK_FIRST])
        self.best = list(ids[:FIRST_BEST])

    @property
    def out_of_calls(self) -> bool:
        return self.turn.used >= MAX_TOOL_CALLS

    @property
    def finishing(self) -> bool:
        """Only the finishing tools run now."""
        return self.out_of_calls or self.turn.lock is not None

    def name_best(self, named: Sequence[str], seen: Callable[[str], bool]) -> None:
        """The model's closest ids: only ones it has seen, at most `BEST`."""
        self.best = [e for e in dict.fromkeys(named) if seen(e)][:BEST]

    def refusal(  # noqa: PLR0911 (one rule per return)
        self,
        name: str,
        args: Mapping[str, object],
        *,
        offered: set[str],
        waiting: bool,
        seen: Callable[[str], bool],
    ) -> str | None:
        """Why this call may not run now, or None. A search judges the one before it first
        (`judge_last_search`), with the best-so-far this call names."""
        if waiting:
            return NOT_RUN_WAITING
        if name not in offered:
            return NOT_OFFERED.format(name=name)
        named = best_of(args)
        if named is not None:
            self.name_best(named, seen)
        search = name in COUNTED
        if search:
            self.judge_last_search()
            self.update_lock()
        looking = search or name == ToolName.VIEW_RESULTS.value
        if looking and self.turn.lock is not None:
            return NOT_RUN_LOCKED.format(reason=self.turn.lock, best=", ".join(self.best) or "—")
        if search and self.must_view:
            return NOT_RUN_LOOK.format(ids=", ".join(self.must_view))
        if search and self.out_of_calls:
            return NOT_RUN_CAP
        if name == ToolName.VIEW_RESULTS.value and self.turn.views >= MAX_VIEWS:
            return NOT_RUN_VIEWS.format(views=MAX_VIEWS)
        return None

    def ran(self, name: str, outcome: ToolOutcome, best_before: Sequence[str]) -> None:
        """Count a call that was not refused, once it has run."""
        search = name in COUNTED
        ran = not outcome.error
        if name == ToolName.VIEW_RESULTS.value:
            self.turn.views += 1
        if outcome.counted and search:
            self.turn.used += 1
        if ran and search:
            self.turn.searches += 1
            self.turn.unjudged = (outcome.hit_ids, tuple(best_before))
        looked = ran and name == ToolName.VIEW_RESULTS.value
        if looked and set(outcome.hit_ids) & set(self.must_view):
            self.must_view = ()
        self.update_lock()

    def judge_last_search(self) -> None:
        """Did the turn's last search bring something new into the best-so-far? Judged when
        the next search comes, so the model can look at its results first."""
        last = self.turn.unjudged
        if last is None:
            return
        listed, before = last
        self.turn.unjudged = None
        if set(self.best) & (set(listed) - set(before)):
            self.turn.stale = 0
        else:
            self.turn.stale += 1

    def update_lock(self) -> None:
        if self.turn.lock is not None:
            return
        if self.turn.stale >= MAX_STALE:
            self.turn.lock = LOCK_STALE.format(stale=MAX_STALE)
        elif not self.sure and self.turn.searches >= MAX_SEARCHES:
            self.turn.lock = LOCK_SEARCHES.format(searches=MAX_SEARCHES)

    def best_line(self, title_of: Callable[[str], str]) -> str:
        if not self.best:
            best = BEST_NONE
        else:
            best = "; ".join(f"{e} ({title_of(e)[:50]})" for e in self.best)
        line = BEST_LINE.format(best=best)
        if not self.sure:
            line += BEST_COUNT.format(searches=self.turn.searches, cap=MAX_SEARCHES)
        return line
