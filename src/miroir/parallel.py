"""Run a function over many items on a few threads, holding only a few results at a time."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor


def bounded_map[T, R](
    fn: Callable[[T], R], items: Iterable[T], *, workers: int, window: int | None = None
) -> Iterator[R]:
    """Like `map`, in input order, with at most `window` items started but not yet yielded.

    `ThreadPoolExecutor.map` reads the whole input first and keeps every result until it is
    consumed; this reads the input lazily, so memory stays small on long runs.
    """
    if workers <= 1:
        yield from map(fn, items)
        return
    limit = window or 2 * workers
    pending: deque[Future[R]] = deque()
    with ThreadPoolExecutor(workers) as pool:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= limit:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
