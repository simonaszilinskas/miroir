"""Where an entity comes from in POP, as a searcher names it, and place names folded to compare.

Shared by step H4 (`miroir.eval.h4_tools`) and the conversation page (`miroir.web.toolsearch`).
"""

from __future__ import annotations

import re
from enum import StrEnum

from miroir.eval.bm25 import fold
from miroir.records.model import Collection


class Base(StrEnum):
    """Where an entity comes from in POP, as the model filters on it."""

    MERIMEE = "merimee"
    """A building (Mérimée id PA…); its photos are Mémoire records or Commons photos."""
    PALISSY = "palissy"
    """An object kept in a building (Palissy id PM…), photographed in Mémoire."""
    JOCONDE = "joconde"
    """A museum object."""
    MEMOIRE = "memoire"
    """A Mémoire photo tied to no building or object."""
    ROSE_VALLAND = "rose-valland"
    """A work looted during the war (MNR), or a Jeu de Paume room photo."""


def base_of(collection: Collection, entity_id: str) -> Base:
    if collection is Collection.JOCONDE:
        return Base.JOCONDE
    if collection is Collection.ROSE_VALLAND:
        return Base.ROSE_VALLAND
    if entity_id.startswith("PA"):
        return Base.MERIMEE
    if entity_id.startswith("PM"):
        return Base.PALISSY
    return Base.MEMOIRE


#: POP writes a commune's article after it: "Ferté-Bernard (La)", "Isle-Adam (L')".
_ARTICLE_AFTER = re.compile(r"^(.*?)\s*\((le|la|les|l')\)\s*$")


def place_key(text: str) -> str:
    """A commune or département name folded for comparison: "Saint-Étienne" = "saint etienne",
    and "Ferté-Bernard (La)", as POP writes it, = "La Ferté-Bernard"."""
    folded = fold(text).replace("’", "'").strip()
    moved = _ARTICLE_AFTER.match(folded)
    if moved is not None:
        folded = f"{moved.group(2)} {moved.group(1)}"
    return " ".join(re.findall(r"[a-z0-9]+", folded))
