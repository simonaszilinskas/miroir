"""Step H4, scenario S4: what a simulated visitor knows about their own photo.

A real visitor knows where they stand, what kind of thing they photographed, in plain words,
and, in a museum, its name. They do not know the record's title, its author, its date, or any
name POP gives it. `visitor_facts` is the one function that decides what the simulated visitor
is told; it reads only the fields in `VISITOR_FIELDS`, and the kind of thing only through a
small fixed list of plain words (`KIND_WORDS`), so no record wording reaches it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

from miroir.eval.bm25 import tokens
from miroir.eval.h4_tools import Base
from miroir.records.model import Record

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


def visitor_facts(
    records: Sequence[Record],
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
