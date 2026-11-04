"""The agentic search (step H5) when things go wrong: a search that fails, a model that is slow
or down, a stream that drops, calls in flight against the spending caps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from miroir.jobs.openrouter import JSON
from tests.test_web_chat import (
    EXACT,
    WHY,
    Rig,
    calls,
    card_ids,
    of,
    shown,
    tool_messages,
)


def assert_valid_history(bodies: Sequence[Mapping[str, JSON]]) -> None:
    """Every tool call the model made has its reply, right after it, in every request."""
    for body in bodies:
        messages = body["messages"]
        assert isinstance(messages, list)
        for i, message in enumerate(messages):
            assert isinstance(message, dict)
            asked = message.get("tool_calls")
            if not isinstance(asked, list):
                continue
            ids = [str(c["id"]) for c in asked if isinstance(c, dict)]
            replies = messages[i + 1 : i + 1 + len(ids)]
            got = [str(r["tool_call_id"]) for r in replies if isinstance(r, dict)]
            assert got == ids, f"message {i}: calls {ids}, replies {got}"


# --- 1. a tool that raises ---------------------------------------------------------------------


def test_a_search_that_fails_gets_an_error_reply_and_the_turn_goes_on() -> None:
    """A transport error from the search backend once left tool calls without replies: every
    later model call was refused, and the conversation was dead."""
    script = [
        calls(
            [
                ("text_search", {"query": "église", "why": WHY}),
                ("text_search", {"query": "vase céramique", "why": WHY}),
            ]
        ),
        shown("OBJ1"),
    ]
    with Rig(script) as rig:
        rig.backend.failures.append(RuntimeError("connection reset"))
        cid = rig.start(photo=False, words="église")
        events = rig.send(cid, "goal", EXACT)
    assert not of(events, "error")
    failed, second = tool_messages(rig.model.bodies[1])
    assert failed.startswith("error: text_search failed (RuntimeError)")
    assert second.startswith("text_search 'vase céramique'")
    assert card_ids(events) == ["OBJ1"]
    assert_valid_history(rig.model.bodies)
    assert any(s.get("error") for s in of(events, "step"))


def test_a_search_that_failed_may_be_tried_again() -> None:
    """A failed search is not a repeat: its signature is noted only once it ran."""
    search = ("text_search", {"query": "église", "why": WHY})
    script = [calls([search]), calls([search]), shown("PA1")]
    with Rig(script) as rig:
        rig.backend.failures.append(RuntimeError("connection reset"))
        cid = rig.start(photo=False, words="église")
        rig.send(cid, "goal", EXACT)
    failed, again = tool_messages(rig.model.bodies[2])
    assert failed.startswith("error: text_search failed (RuntimeError)")
    assert again.startswith("text_search 'église'")
