import math

import pytest

from l3s_agent.retrieval.bm25 import BM25Index, tokenize


def test_tokenizer_is_deterministic_without_stemming_or_stop_words() -> None:
    assert tokenize("Wind-speed, PV's R² = 0.91") == (
        "wind",
        "speed",
        "pv",
        "s",
        "r2",
        "0",
        "91",
    )
    assert "the" in tokenize("The winds and wind")
    assert "winds" in tokenize("The winds and wind")
    assert "wind" in tokenize("The winds and wind")


def test_bm25_exact_single_term_behavior() -> None:
    index = BM25Index.build(["solar solar cloud", "wind cloud"])
    scores = index.scores("solar")
    expected_idf = math.log(1 + (2 - 1 + 0.5) / (1 + 0.5))
    expected = expected_idf * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * (3 / 2.5)))
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0.0


def test_bm25_query_terms_are_not_double_counted() -> None:
    index = BM25Index.build(["solar generation", "wind generation"])
    assert index.scores("solar solar") == index.scores("solar")


def test_bm25_round_trip_preserves_scores() -> None:
    original = BM25Index.build(["cloud affects solar output", "wind speed affects power"])
    restored = BM25Index.from_dict(original.to_dict())
    assert restored.scores("weather solar output") == original.scores("weather solar output")
