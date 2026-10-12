"""Filter values (F3) read from record fields: region, domain and period, tidied for a menu.

Records spell the same value several ways ("Ile-de-France", "Île-de-France"; "Peinture",
"peinture") and still use pre-2016 regions. Period strings mix centuries, quarters, decades
and years; the menu offers broad periods instead.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

#: Pre-2016 regions and other spellings, mapped to today's regions.
_REGIONS: dict[str, str] = {
    "ile-de-france": "Île-de-France",
    "provence-alpes-cote-d'azur": "Provence-Alpes-Côte d'Azur",
    "provence-alpes-cote d'azur": "Provence-Alpes-Côte d'Azur",
    "pays-de-la-loire": "Pays de la Loire",
    "centre": "Centre-Val de Loire",
    "franche-comte": "Bourgogne-Franche-Comté",
    "bourgogne": "Bourgogne-Franche-Comté",
    "limousin": "Nouvelle-Aquitaine",
    "aquitaine": "Nouvelle-Aquitaine",
    "poitou-charentes": "Nouvelle-Aquitaine",
    "alsace": "Grand Est",
    "lorraine": "Grand Est",
    "champagne-ardenne": "Grand Est",
    "auvergne": "Auvergne-Rhône-Alpes",
    "rhone-alpes": "Auvergne-Rhône-Alpes",
    "picardie": "Hauts-de-France",
    "nord-pas-de-calais": "Hauts-de-France",
    "languedoc-roussillon": "Occitanie",
    "midi-pyrenees": "Occitanie",
    "basse-normandie": "Normandie",
    "haute-normandie": "Normandie",
}

PREHISTORY = "Préhistoire"
ANTIQUITY = "Antiquité"
MIDDLE_AGES = "Moyen Âge"
#: Periods in menu order. Centuries 1-5 count as Antiquity, 6-15 as the Middle Ages.
PERIODS: tuple[str, ...] = (
    PREHISTORY,
    ANTIQUITY,
    MIDDLE_AGES,
    *(f"{c}e siècle" for c in range(16, 22)),
)

_PREHISTORY_WORDS = ("paleolithique", "neolithique", "mesolithique", "age du bronze", "age du fer")
_ANTIQUITY_WORDS = ("romain", "antiquite", "antique", "grec", "egypt", "av. j", "avant j")
_CENTURY = re.compile(r"\b(\d{1,2})\s*(?:e|er|eme|ème)\s+siecle")
_ROMAN_CENTURY = re.compile(r"\b([ivxl]+)\s*(?:e|er|eme)\s+siecle")
_YEAR = re.compile(r"(?<![\d-])(\d{3,4})(?!\d)")
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50}


def fold(text: str) -> str:
    """Lower case without accents."""
    decomposed = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def region(value: str | None) -> str | None:
    """Today's region for a record's region, or the value as written when unknown."""
    if not value or not value.strip():
        return None
    return _REGIONS.get(fold(value), value.strip())


def domain(value: str) -> str:
    """One spelling per domain: first letter upper case, the rest as written in lower case."""
    text = value.strip().lower()
    return text[:1].upper() + text[1:]


def periods(values: Iterable[str]) -> tuple[str, ...]:
    """Broad periods (`PERIODS`) that a record's period strings mention, in menu order."""
    found: set[str] = set()
    for value in values:
        found.update(_period(fold(value)))
    return tuple(p for p in PERIODS if p in found)


def _period(text: str) -> set[str]:
    out: set[str] = set()
    if any(w in text for w in _PREHISTORY_WORDS):
        out.add(PREHISTORY)
    if any(w in text for w in _ANTIQUITY_WORDS):
        out.add(ANTIQUITY)
    centuries = [int(m) for m in _CENTURY.findall(text)]
    centuries += [_roman(m) for m in _ROMAN_CENTURY.findall(text)]
    if not centuries and "siecle" not in text:
        centuries = [(int(y) - 1) // 100 + 1 for y in _YEAR.findall(text) if 100 <= int(y) <= 2100]
    for century in centuries:
        label = _century_label(century)
        if label is not None:
            out.add(label)
    return out


def _century_label(century: int) -> str | None:
    if century <= 0 or century > 21:
        return None
    if century <= 5:
        return ANTIQUITY
    if century <= 15:
        return MIDDLE_AGES
    return f"{century}e siècle"


def _roman(numeral: str) -> int:
    total = 0
    for i, letter in enumerate(numeral):
        value = _ROMAN[letter]
        following = _ROMAN[numeral[i + 1]] if i + 1 < len(numeral) else 0
        total += -value if value < following else value
    return total
