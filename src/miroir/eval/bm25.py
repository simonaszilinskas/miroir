"""BM25 over French record text (experiments.md B4, H1). Plain Python and NumPy, no model.

Words are folded (lower case, accents dropped), French stop words removed, and a plural
"s" or "x" cut, so "Églises" matches "église". Scores follow Robertson's BM25 with the usual
k1 = 1.2 and b = 0.75.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from miroir.records.model import Record

K1 = 1.2
B = 0.75

_STOP_WORDS_TEXT = """
a au aux avec ce ces cet cette d dans de des du elle en et eux il ils je la le les leur
leurs lui ma mais me meme mes moi mon ne nos nous on ou par pas pour qu que qui sa se ses
son sur ta te tes toi ton tu un une vos votre vous y l s n c j m t est sont ete etre plus
tres sans sous entre vers chez comme dont ni si aussi ainsi alors donc vue voir probable
probablement inconnu inconnue
"""
# "notre" stays searchable (Notre-Dame), and so does "or" (gold).
STOP_WORDS = frozenset(_STOP_WORDS_TEXT.split())

_WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lower case without accents: "Église" -> "eglise"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c)).replace("œ", "oe")


def tokens(text: str) -> list[str]:
    """Searchable words of `text`: folded, stop words out, plural "s" or "x" cut."""
    out: list[str] = []
    for word in _WORD.findall(fold(text)):
        if word in STOP_WORDS or (len(word) < 2 and not word.isdigit()):
            continue
        plural = len(word) > 3 and word[-1] in "sx" and not word[-2].isdigit()
        out.append(word[:-1] if plural else word)
    return out


def record_text(record: Record) -> str:
    """Every field of a record worth searching, joined."""
    return " ".join(
        [
            record.title,
            record.description,
            *record.denomination,
            *record.domain,
            *record.authors,
            *record.period,
            record.commune or "",
            record.departement or "",
            record.region or "",
        ]
    )


class Bm25:
    """An index over a list of documents; `scores` gives every document's score for a query."""

    def __init__(self, documents: Sequence[str], *, k1: float = K1, b: float = B) -> None:
        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths = np.zeros(len(documents), dtype=np.float32)
        for row, text in enumerate(documents):
            counts = Counter(tokens(text))
            lengths[row] = sum(counts.values())
            for term, count in counts.items():
                rows, tfs = postings.setdefault(term, ([], []))
                rows.append(row)
                tfs.append(count)
        average = float(lengths.mean()) if len(documents) and lengths.mean() > 0 else 1.0
        norm = k1 * (1 - b + b * lengths / average)
        self.size = len(documents)
        self._weights: dict[str, tuple[NDArray[np.int64], NDArray[np.float32]]] = {}
        for term, (rows, tfs) in postings.items():
            row_array = np.asarray(rows, dtype=np.int64)
            tf = np.asarray(tfs, dtype=np.float32)
            idf = math.log(1 + (self.size - len(rows) + 0.5) / (len(rows) + 0.5))
            weight = idf * tf * (k1 + 1) / (tf + norm[row_array])
            self._weights[term] = (row_array, weight.astype(np.float32))

    def scores(self, query: str) -> NDArray[np.float32]:
        """Score of every document; a word said twice in the query counts twice."""
        out = np.zeros(self.size, dtype=np.float32)
        for term, count in Counter(tokens(query)).items():
            found = self._weights.get(term)
            if found is not None:
                rows, weights = found
                out[rows] += count * weights
        return out
