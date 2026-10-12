from __future__ import annotations

import numpy as np

from miroir.eval.bm25 import Bm25, fold, tokens


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
