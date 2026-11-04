"""The simulated visitor of the offline agentic search (step H5 on test sets): what they know
about their own photo, and how they answer the agent's questions.

A real visitor knows where they stand, what kind of thing they photographed, in plain words,
and, in a museum, its name. They do not know the record's title, its author, its date, or any
name POP gives it. `visitor_facts` is the one function that decides what the simulated visitor
is told; it reads only the fields in `VISITOR_FIELDS`, and the kind of thing only through a
small fixed list of plain words (`KIND_WORDS`), so no record wording reaches it.

`Visitor` answers one question of the agent (`ask_visitor`, as the page shows it: the text,
the suggested answers, whether it offers to share a position or to frame the work): a model
call with no photo and only the facts. It shares the position when the question offers it and
the photo carries one. It frames nothing: offline, a framed part could not be searched
(`miroir.eval.offline`), so `ask_region` is declined.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Protocol

from miroir.eval.bm25 import tokens
from miroir.jobs.openrouter import JSON
from miroir.records.bases import Base
from miroir.web.chat_model import ChatModel, ModelError

#: Every fact the simulated visitor may be given. A test keeps it to these.
VISITOR_FIELDS = ("commune", "departement", "position", "kind", "museum")

#: Plain words for the kind of thing, picked from a record's domain and denomination words.
#: First match wins; the words matched never reach the visitor, only the plain phrase.
KIND_WORDS: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"peinture", "tableau", "huile"}), "un tableau"),
    (frozenset({"vitrail"}), "un vitrail"),
    (frozenset({"retable"}), "un retable, le grand décor derrière l'autel"),
    (frozenset({"sculpture", "statue", "statuette", "buste", "relief", "bas"}), "une sculpture"),
    (frozenset({"monnaie", "numismatique", "medaille", "jeton"}), "une pièce ou une médaille"),
    (frozenset({"dessin"}), "un dessin"),
    (frozenset({"estampe", "gravure", "imprime"}), "une gravure"),
    (frozenset({"photographie"}), "une photographie"),
    (frozenset({"ceramique", "vase", "poterie", "assiette", "verrerie"}), "un vase ou une poterie"),
    (frozenset({"mobilier", "meuble", "ameublement"}), "un meuble"),
    (frozenset({"textile", "tapisserie", "costume"}), "un tissu ou un vêtement"),
    (frozenset({"orfevrerie", "calice", "reliquaire"}), "un objet en métal précieux"),
    (
        frozenset({"archeologie", "romain", "prehistoire", "paleolithique"}),
        "un objet ancien de fouilles",
    ),
)


@dataclass(frozen=True, slots=True)
class VisitorFacts:
    """What the simulated visitor knows. Empty strings and None: they do not know."""

    commune: str = ""
    departement: str = ""
    position: tuple[float, float] | None = None
    """Where the photo was taken (latitude, longitude), when the query photo carries it."""
    kind: str = ""
    museum: str = ""

    def text(self) -> str:
        """The facts as the visitor's prompt gives them, in French."""
        lines: list[str] = []
        if self.kind:
            lines.append(f"Ce que vous avez photographié : {self.kind}.")
        if self.museum:
            lines.append(f"Vous êtes dans ce musée : {self.museum}.")
        if self.commune:
            lines.append(f"Vous êtes à {self.commune}.")
        if self.departement:
            lines.append(f"Département : {self.departement}.")
        if self.position is not None:
            lat, lon = self.position
            lines.append(f"Votre téléphone donne votre position : {lat:.5f}, {lon:.5f}.")
        return "\n".join(lines) or "Vous ne savez rien de plus que ce que montre la photo."

    def given(self) -> dict[str, str]:
        """The facts given, for the trace."""
        out: dict[str, str] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value:
                out[f.name] = str(value)
        return out


def plain_kind(base: Base, words: Sequence[str]) -> str:
    """The kind of thing in a visitor's words, from the base and a record's domain words."""
    if base is Base.MERIMEE:
        return "un bâtiment (ou une partie d'un bâtiment)"
    if base is Base.MEMOIRE:
        return "un bâtiment ou un lieu"
    folded = set(tokens(" ".join(words)))
    for keys, phrase in KIND_WORDS:
        if folded & keys:
            return phrase
    return "un objet"


class VisitorRecord(Protocol):
    """The fields of a record the facts may read (`Record` and `CatalogRecord` both have them)."""

    @property
    def commune(self) -> str | None: ...
    @property
    def departement(self) -> str | None: ...
    @property
    def domain(self) -> tuple[str, ...]: ...
    @property
    def denomination(self) -> tuple[str, ...]: ...


def visitor_facts(
    records: Sequence[VisitorRecord],
    base: Base,
    *,
    position: tuple[float, float] | None,
    museum: str | None,
) -> VisitorFacts:
    """What a visitor standing before the answer knows, from the answer's records.

    Reads `commune`, `departement`, and `domain` and `denomination` only through
    `plain_kind`. Never `title`, `description`, `authors`, `period` or `photo_date`.
    """
    communes = [r.commune for r in records if r.commune]
    depts = [r.departement for r in records if r.departement]
    words = [w for r in records for w in (*r.domain, *r.denomination)]
    return VisitorFacts(
        commune=communes[0] if communes else "",
        departement=depts[0] if depts else "",
        position=position,
        kind=plain_kind(base, words),
        museum=(museum or "") if base is Base.JOCONDE else "",
    )


VISITOR = """Tu joues un visiteur qui a pris une photo d'un monument ou d'une œuvre et qui \
répond à la question d'un assistant. Tu ne sais que ce qui t'est donné ci-dessous. Réponds \
comme un visiteur, en mots simples, en une phrase courte, jamais avec des termes de base de \
données. Si tu ne sais pas, réponds exactement "je ne sais pas". N'invente aucun nom."""

VISITOR_TASK = """Ce que tu sais :
{facts}

Question : {question}
Réponses proposées : {choices}

Réponds par un objet JSON seulement : {{"answer": "ta réponse"}}. Tu peux reprendre une \
réponse proposée si elle est juste. Si la question porte sur une partie de la photo, réponds \
"toute la photo"."""

DONT_KNOW = "je ne sais pas"
#: What the visitor writes first, with the photo: they want to know what it shows.
GOAL = "Je cherche ce monument ou cette œuvre précise : qu’est-ce que c’est ?"


@dataclass(frozen=True, slots=True)
class Heard:
    """The visitor's answer to one question."""

    text: str
    share_position: bool


class Visitor:
    """Answers the agent's questions from `facts`, through `model` (no photo goes with it)."""

    def __init__(self, model: ChatModel, facts: VisitorFacts) -> None:
        self.model = model
        self.facts = facts

    async def answer(self, question: Mapping[str, JSON]) -> Heard:
        """An answer to a `question` event of the conversation."""
        text = str(question.get("text") or "")
        raw = question.get("choices")
        choices = [str(c) for c in raw] if isinstance(raw, list) else []
        body: dict[str, JSON] = {
            "messages": [
                {"role": "system", "content": VISITOR},
                {
                    "role": "user",
                    "content": VISITOR_TASK.format(
                        facts=self.facts.text(), question=text, choices=" / ".join(choices) or "-"
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            said = await self.model.complete(body)
        except ModelError:
            said = None
        answer = DONT_KNOW if said is None else parse_answer(said.text)
        share = question.get("position") is True and self.facts.position is not None
        return Heard(answer, share)


def parse_answer(text: str) -> str:
    """The answer in a JSON reply; "je ne sais pas" when there is none."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return DONT_KNOW
    try:
        raw: object = json.loads(match.group(0))
    except json.JSONDecodeError:
        return DONT_KNOW
    answer = raw.get("answer") if isinstance(raw, dict) else None  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return " ".join(str(answer).split())[:300] if answer else DONT_KNOW  # pyright: ignore[reportUnknownArgumentType]
