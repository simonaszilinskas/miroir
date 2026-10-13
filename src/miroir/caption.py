"""Archive captions: what a small open vision model reads on a print's mount, and the place
it names.

Mémoire prints sit on a card whose typed or written caption names the commune, its
département in brackets, then the building ("BEAUNE (Côte d'Or) Eglise Notre-Dame"). The
model (`miroir.remote.caption`, on Modal) copies that line; `Gazetteer.find` looks the
commune up among the communes the index knows, and gives it a position: the median of the
positions held for that commune's buildings and objects. The site then uses that position
as it uses the visitor's own (step H2's soft prior), and the line as words.

Pure Python and NumPy: the Mac, the tests and the web container import this module.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

#: The open model that reads captions, and the ones compared with it (results.md, T1 archive).
CAPTION_MODELS: dict[str, str] = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
}
SITE_CAPTION_MODEL = "qwen3-vl-2b"
#: The photo goes to the model at most this long on its long side, in pixels (T1 was measured
#: on POP's prints, about 670 px wide; a larger photo costs time and was not measured).
CAPTION_SIDE = 768
#: The answer is one line: a place and a building.
CAPTION_TOKENS = 64
NO_CAPTION = "AUCUNE"
CAPTION_QUESTION = (
    "Cette photo d'archive est collée sur un carton qui porte une légende tapée ou écrite "
    "à la main. Recopie mot pour mot la ligne de la légende qui nomme le lieu : la commune, "
    "le département entre parenthèses, puis l'édifice et la partie montrée. Une seule ligne, "
    f"sans rien ajouter. S'il n'y a aucune légende, réponds {NO_CAPTION}."
)
#: Longest caption kept, in characters (as the site's text box).
MAX_CAPTION = 300
#: A commune named without its département is taken only from this many letters.
MIN_BARE_NAME = 4
#: Longest commune name looked up, in words.
MAX_NAME_WORDS = 8


def clean_caption(answer: str) -> str:
    """The model's answer as the page shows it: one line, no quotes, empty for no caption."""
    line = " ".join(answer.replace("```", " ").split()).strip(" \"'«»")
    if not line or line.upper().startswith(NO_CAPTION):
        return ""
    return line[:MAX_CAPTION]


def normalise(text: str) -> str:
    """Lower case, no accents, words split on anything but letters and digits; "St" and
    "Ste" written out."""
    plain = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in plain if not unicodedata.combining(c)).lower()
    words: list[str] = re.findall(r"[a-z0-9]+", plain)
    spelt = {"st": "saint", "ste": "sainte"}
    return " ".join(spelt.get(w, w) for w in words)


def name_forms(commune: str) -> set[str]:
    """Ways a caption may write a commune: "Arques (Les)" is also "Les Arques"."""
    forms = {normalise(commune)}
    found = re.fullmatch(r"\s*(.+?)\s*\((le|la|les|l')\)\s*", commune, flags=re.IGNORECASE)
    if found:
        name, article = found.groups()
        forms.add(normalise(name))
        forms.add(normalise(f"{article} {name}"))
    return {f for f in forms if f}


@dataclass(frozen=True, slots=True)
class CaptionPlace:
    """A commune the caption names, as the index writes it, and where it lies."""

    commune: str
    departement: str
    latitude: float
    longitude: float
    with_departement: bool
    """The caption also names the département."""

    @property
    def label(self) -> str:
        return f"{self.commune} ({self.departement})" if self.departement else self.commune


@dataclass(frozen=True, slots=True)
class _Commune:
    commune: str
    departement: str
    latitude: float
    longitude: float


class Gazetteer:
    """The communes the index knows, each with one position."""

    def __init__(self, communes: Iterable[_Commune]) -> None:
        self._by_name: dict[str, list[_Commune]] = defaultdict(list)
        for c in communes:
            for form in name_forms(c.commune):
                self._by_name[form].append(c)
        self.size = len({(c.commune, c.departement) for v in self._by_name.values() for c in v})

    @classmethod
    def build(
        cls,
        places: Iterable[tuple[str | None, str | None, float, float]],
    ) -> Gazetteer:
        """From (commune, département, latitude, longitude) of each placed entity: each
        commune lies at the median of its entities' positions."""
        points: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        for commune, departement, lat, lon in places:
            if commune and np.isfinite(lat) and np.isfinite(lon):
                points[(commune, departement or "")].append((lat, lon))
        return cls(
            _Commune(
                c, d, float(np.median([p[0] for p in ps])), float(np.median([p[1] for p in ps]))
            )
            for (c, d), ps in points.items()
        )

    def find(self, caption: str) -> CaptionPlace | None:
        """The commune the caption names, or None.

        Every run of words that spells a known commune is a candidate. One whose département
        the caption also names wins; else a name written in capitals, as the cards write the
        commune, of at least `MIN_BARE_NAME` letters. Ties go to the longer name, then the
        earlier one.
        """
        words = normalise(caption).split()
        if not words:
            return None
        text = f" {' '.join(words)} "
        capitals = _capital_words(caption)
        best: tuple[tuple[int, int, int], CaptionPlace] | None = None
        for start in range(len(words)):
            for size in range(1, min(MAX_NAME_WORDS, len(words) - start) + 1):
                name = " ".join(words[start : start + size])
                for c in self._by_name.get(name, ()):
                    with_dept = bool(c.departement) and f" {normalise(c.departement)} " in text
                    in_capitals = all(w in capitals for w in words[start : start + size])
                    if not with_dept and not (in_capitals and len(name) >= MIN_BARE_NAME):
                        continue
                    key = (int(with_dept), len(name), -start)
                    if best is None or key > best[0]:
                        place = CaptionPlace(
                            c.commune, c.departement, c.latitude, c.longitude, with_dept
                        )
                        best = (key, place)
        return None if best is None else best[1]


def _capital_words(caption: str) -> set[str]:
    """Normalised words the caption writes in capitals (two letters or more)."""
    return {normalise(w) for w in re.findall(r"[^\W\d_]{2,}", caption) if w.isupper()}
