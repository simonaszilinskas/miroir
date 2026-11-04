from __future__ import annotations

import numpy as np

from miroir.eval.bm25 import DESCRIPTION_CHARS, Bm25, fold, record_text, tokens
from tests import web_helpers as h


def test_record_text_keeps_as_much_description_as_the_site_does() -> None:
    # The site's catalog holds DESCRIPTION_CHARS of a description: the evaluation's BM25
    # (H1, H4) must index the same text, or it finds words the site never sees.
    record = h.record("R1", "E1", "joconde", description="a" * DESCRIPTION_CHARS + " lièvre")
    assert "lièvre" not in record_text(record)
    assert "a" * DESCRIPTION_CHARS in record_text(record)


def test_tokens_fold_accents_drop_stop_words_and_plurals() -> None:
    assert fold("Église Œuvre") == "eglise oeuvre"
    assert tokens("Les églises de Notre-Dame, 1890") == ["eglise", "notre", "dame", "1890"]
    assert tokens("vitraux") == ["vitrau"] == tokens("vitrau")


def test_the_document_holding_the_rare_word_ranks_first() -> None:
    index = Bm25(["église de Beaune", "église de Dijon", "château de Dijon", ""])
    scores = index.scores("église Beaune")
    assert int(np.argmax(scores)) == 0
    assert scores[3] == 0
    assert index.scores("").sum() == 0


def test_a_word_said_twice_counts_twice() -> None:
    index = Bm25(["beaune", "dijon"])
    once = index.scores("beaune dijon")
    twice = index.scores("beaune beaune dijon")
    assert once[0] == once[1]
    assert twice[0] > twice[1]
