"""Run a function over many items on a few threads or processes, holding only a few results
at a time."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)

logger = logging.getLogger(__name__)


def bounded_map[T, R](
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    workers: int,
    window: int | None = None,
    salvage: Callable[[R], None] | None = None,
) -> Generator[R]:
    """Like `map`, in input order, with at most `window` items started but not yet yielded.

    `ThreadPoolExecutor.map` reads the whole input first and keeps every result until it is
    consumed; this reads the input lazily, so memory stays small on long runs.

    If the map stops early (`fn` raised, or the consumer stopped or raised), items not started
    are cancelled, and the ones already running are waited for: each result they bring is
    passed to `salvage`, so paid work in flight is saved rather than dropped.
    """
    if workers <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(workers) as pool:
        yield from _ordered(pool, fn, items, window or 2 * workers, salvage)


def process_map[T, R](
    fn: Callable[[T], R], items: Iterable[T], *, workers: int, window: int | None = None
) -> Generator[R]:
    """`bounded_map` on processes: for work that holds Python's lock (decoding and cutting
    images in Python). `fn` must be a module-level function; items and results are pickled."""
    with ProcessPoolExecutor(workers) as pool:
        yield from _ordered(pool, fn, items, window or 2 * workers, None)


def _ordered[T, R](
    pool: Executor,
    fn: Callable[[T], R],
    items: Iterable[T],
    limit: int,
    salvage: Callable[[R], None] | None,
) -> Generator[R]:
    pending: deque[Future[R]] = deque()
    finished = False
    try:
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= limit:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
        finished = True
    finally:
        if not finished:
            _drain(pending, salvage)


def _drain[R](pending: Iterable[Future[R]], salvage: Callable[[R], None] | None) -> None:
    """Cancel what has not started; wait for the rest and hand each result to `salvage`."""
    running = [f for f in pending if not f.cancel()]
    saved = 0
    for future in running:
        error = future.exception()
        if error is not None:
            logger.warning("a call stopped with the map failed too: %s", error)
        elif salvage is not None:
            salvage(future.result())
            saved += 1
    if running:
        logger.info("map stopped: %d call(s) in flight finished, %d saved", len(running), saved)


def unordered_map[T, R](
    fn: Callable[[T], R], items: Iterable[T], *, workers: int, window: int | None = None
) -> Generator[R]:
    """Like `bounded_map`, but yields each result as soon as it is ready, so one slow item
    does not hold back the others. Items not started are cancelled if it stops early."""
    limit = window or 2 * workers
    source = iter(items)
    pending: set[Future[R]] = set()
    finished = False
    with ThreadPoolExecutor(workers) as pool:
        try:
            for item in source:
                pending.add(pool.submit(fn, item))
                if len(pending) < limit:
                    continue
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
            finished = True
        finally:
            if not finished:
                _drain(pending, None)
