"""The agentic search's messages (step H5): how they are built, and how images named in them
become data URLs only when a request leaves (`render`)."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from miroir.jobs.openrouter import JSON, ToolCall
from miroir.web.chat_prompts import QUERY_SHOWN, THUMB_SHOWN, THUMBNAILS
from miroir.web.chat_tools import ToolName, ToolOutcome, arguments_of
from miroir.web.toolsearch import Hit

#: Thumbnails kept in what the model is sent again each call; older ones become a note.
KEEP_THUMBS = 12
#: Transformed photos kept in what the model is sent again each call (the latest ones).
KEEP_QUERY_IMAGES = 3
#: Long side of the visitor's photo and framed region as the model sees them.
PHOTO_SIDE = 768

Message = dict[str, JSON]
Event = dict[str, JSON]


@dataclass(frozen=True, slots=True)
class CardView:
    """A result card, as the page draws it (the site's card, plus the model's reason)."""

    entity_id: str
    title: str
    place: str
    collection: str
    collection_label: str
    pop_url: str
    image_url: str | None
    image_fallback_url: str | None
    matched: str | None
    photo_count: int
    level: str | None = None
    level_label: str | None = None
    reason: str = ""


def render(
    messages: Sequence[Message],
    *,
    photo: JSON | None,
    regions: Sequence[JSON],
    thumbs: Mapping[str, JSON],
    query_images: Mapping[str, JSON],
    check: Callable[[str], None],
) -> list[JSON]:
    """The messages with the images they name, each already encoded once as a data URL
    (`data_url`): no image work here. Only the last `KEEP_THUMBS` thumbnails go again;
    `check` sees each thumbnail's id once more before it leaves."""
    keep = {(i, j) for i, j, _ in _sent_thumbs(messages)}
    keep_queries = set(list(query_images)[-KEEP_QUERY_IMAGES:])
    out: list[JSON] = []
    for i, message in enumerate(messages):
        parts = parts_of(message)
        if not parts:
            out.append(dict(message))
            continue
        rendered: list[JSON] = []
        for j, part in enumerate(parts):
            if not (isinstance(part, dict) and part.get("type") == "image_ref"):
                rendered.append(part)
                continue
            ref = str(part.get("ref"))
            if ref == "thumb":
                image_id = str(part.get("image_id"))
                if (i, j) not in keep or image_id not in thumbs:
                    rendered.append(text_part(THUMB_SHOWN))
                    continue
                check(image_id)
                rendered.append(thumbs[image_id])
            elif ref == "query":
                key = str(part.get("image_id"))
                if key in query_images and key in keep_queries:
                    rendered.append(query_images[key])
                else:
                    rendered.append(text_part(QUERY_SHOWN))
            elif ref == "region" and (n := _index(part.get("image_id"))) < len(regions):
                rendered.append(regions[n])
            elif ref == "photo" and photo is not None:
                rendered.append(photo)
        out.append({**message, "content": rendered})
    return out


def _index(value: object) -> int:
    try:
        return int(str(value))
    except ValueError:
        return 1 << 30


def sent_thumbs(messages: Sequence[Message]) -> set[str]:
    """The ids of the thumbnails `render` sends again: the conversation keeps only those."""
    return {image_id for _, _, image_id in _sent_thumbs(messages)}


def _sent_thumbs(messages: Sequence[Message]) -> list[tuple[int, int, str]]:
    """(message, part, image id) of the last `KEEP_THUMBS` thumbnails named."""
    named = [
        (i, j, str(p.get("image_id")))
        for i, m in enumerate(messages)
        for j, p in enumerate(parts_of(m))
        if isinstance(p, dict) and p.get("type") == "image_ref" and p.get("ref") == "thumb"
    ]
    return named[-KEEP_THUMBS:] if KEEP_THUMBS > 0 else []


def shown_images(
    images: Sequence[tuple[str, str]], query_images: Mapping[str, object]
) -> list[Message]:
    """After a reply's tools: the thumbnails it asked for, then the transformed photos it
    searched, each as one message the model reads next."""
    searched = [(label, key) for label, key in images if key in query_images]
    thumbs = [(label, key) for label, key in images if key not in query_images]
    out: list[Message] = []
    if thumbs:
        parts: list[JSON] = [text_part(THUMBNAILS)]
        for label, image_id in thumbs:
            parts += [text_part(label), image_ref("thumb", image_id)]
        out.append(user_message(*parts))
    if searched:
        shown: list[JSON] = []
        for label, key in searched:
            shown += [text_part(label), image_ref("query", key)]
        out.append(user_message(*shown))
    return out


def text_part(text: str) -> JSON:
    return {"type": "text", "text": text}


def image_ref(ref: str, image_id: object) -> JSON:
    """An image named, not held: it becomes a data URL only when sent."""
    return {"type": "image_ref", "ref": ref, "image_id": str(image_id)}


def data_url(jpeg: bytes) -> JSON:
    url = f"data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def user_message(*parts: JSON) -> Message:
    return {"role": "user", "content": list(parts)}


def parts_of(message: Mapping[str, JSON]) -> list[JSON]:
    content = message.get("content")
    return content if isinstance(content, list) else []


def assistant_message(text: str, calls: Sequence[ToolCall], ids: Sequence[str]) -> Message:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call, call_id in zip(calls, ids, strict=True)
        ],
    }


def tool_message(call_id: str, name: str, text: str) -> Message:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": text}


def call_id_of(call: ToolCall, turn: int, n: int) -> str:
    # Mistral wants nine letters or digits; the provider's own ids already are.
    return call.id or f"c{turn:03d}n{n:02d}xy"[:9]


def step_event(
    step: int,
    tool: str,
    label: str,
    summary: str,
    thumbs: Sequence[JSON],
    *,
    error: bool = False,
) -> Event:
    """A finished step. `tool` is the tool's name, or `FIRST_LOOK` (the page picks its icon)."""
    return {
        "type": "step",
        "id": step,
        "tool": tool,
        "state": "done",
        "label": label,
        "summary": summary,
        "thumbs": list(thumbs),
        "error": error,
    }


#: The `tool` of the first look's step.
FIRST_LOOK = "first_look"


def card_of_hit(hit: Hit) -> CardView:
    return CardView(
        hit.entity_id,
        hit.title,
        hit.place,
        hit.collection,
        hit.collection_label,
        hit.pop_url,
        hit.image_url,
        hit.image_fallback_url,
        hit.matched,
        hit.photo_count,
    )


def card_json(card: CardView, reason: str) -> JSON:
    return {
        "entity_id": card.entity_id,
        "title": card.title,
        "place": card.place,
        "collection": card.collection,
        "collection_label": card.collection_label,
        "pop_url": card.pop_url,
        "image_url": card.image_url,
        "image_fallback_url": card.image_fallback_url,
        "matched": card.matched,
        "photo_count": card.photo_count,
        "level": card.level,
        "level_label": card.level_label,
        "reason": reason,
    }


def summarised(calls: Sequence[ToolCall]) -> bool:
    """The reply shows results with a summary: the visitor reads that, not the same thing
    twice (Mistral tends to write both)."""
    return any(
        call.name == ToolName.SHOW_RESULTS.value
        and str(arguments_of(call.arguments).get("summary") or "").strip()
        for call in calls
    )


def with_note(outcome: ToolOutcome, note: str) -> ToolOutcome:
    return replace(outcome, text=f"{outcome.text}\n{note}")
