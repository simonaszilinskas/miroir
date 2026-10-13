"""T7 (search by words): the writer's brief and answers, and how word search is ranked."""

from __future__ import annotations

import numpy as np

from miroir.eval.sets import Query
from miroir.eval.words import Draw, Length, Register, parse_written
from miroir.eval.wordsearch import Signals, hit_at, rank_of, ranks, weighted
from miroir.records.model import Collection
from miroir.web import ranking
from miroir.web.engine import word_signals
from miroir.web.routing import TextWeights
from tests import web_helpers


def _draw(*, english: bool, length: Length) -> Draw:
    query = Query("T7-a", "T7", "E1", "a", ("E1",), ("a",))
    return Draw(query, "tune", Collection.JOCONDE, length, Register.PLAIN, english)


def test_the_brief_sets_length_and_asks_english_only_when_drawn() -> None:
    short = _draw(english=False, length=Length.WORDS).prompt
    assert "3 à 6 mots" in short
    assert '"en"' not in short
    assert '"en"' in _draw(english=True, length=Length.LONG).prompt


def test_written_answers_keep_french_and_optional_english() -> None:
    written = parse_written('```json\n{"fr": " « vase  bleu » ", "en": "blue vase"}\n```')
    assert written is not None
    assert (written.fr, written.en) == ("vase bleu", "blue vase")
    only_fr = parse_written('{"fr": "église romane"}')
    assert only_fr is not None
    assert only_fr.en == ""
    assert parse_written('{"en": "church"}') is None
    assert parse_written("pas de JSON") is None


def test_rank_of_breaks_ties_as_the_site_does() -> None:
    scores = np.asarray([0.5, 0.9, 0.5, -np.inf, 0.1], dtype=np.float32)
    order = list(ranking.top_entities(scores, 5))
    for row in (0, 1, 2, 4):
        assert rank_of(scores, row) == order.index(row) + 1
    assert rank_of(scores, 3) == 5


def test_weighted_mix_is_the_sites_fuse() -> None:
    a7 = np.asarray([0.1, 0.3, 0.2], dtype=np.float32)
    bm25 = np.asarray([0.0, 0.0, 4.0], dtype=np.float32)
    assert np.allclose(weighted(1.0)(a7, bm25), ranking.min_max(a7))
    assert np.allclose(weighted(0.0)(a7, bm25), ranking.min_max(bm25))
    assert int(np.argmax(weighted(0.5)(a7, bm25))) == 2  # second by A7, first by BM25
    signals = Signals(("q",), a7[None, :], bm25[None, :], np.asarray([2], dtype=np.int64))
    assert list(ranks(signals, weighted(0.0))) == [1]
    assert hit_at(np.asarray([1, 11], dtype=np.int64), 10) == 0.5


def test_word_signals_hide_masked_images_from_both_signals() -> None:
    catalog = web_helpers.catalog()
    a7 = np.linspace(0.1, 0.9, len(catalog.images)).astype(np.float32)
    bm25 = np.ones(len(catalog.records), dtype=np.float32)
    mask = np.ones(len(catalog.images), dtype=bool)
    hidden_entity = catalog.image_entity[0]
    mask[catalog.image_entity == hidden_entity] = False
    a7_signal, bm25_signal = word_signals(catalog, a7, bm25, mask, TextWeights(0.7, 0.3))
    assert (a7_signal.weight, bm25_signal.weight) == (0.7, 0.3)
    assert a7_signal.per_entity[hidden_entity] == -np.inf
    assert bm25_signal.per_entity[hidden_entity] == -np.inf
    only_a7 = word_signals(catalog, a7, bm25, mask, TextWeights(1.0, 0.0))
    assert len(only_a7) == 1


def test_the_site_searches_words_with_the_tuned_weights() -> None:
    weights = TextWeights()
    assert (weights.a7, weights.bm25, weights.refine) == (0.9, 0.1, 0.5)
