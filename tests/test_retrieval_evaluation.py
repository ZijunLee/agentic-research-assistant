import json

import pytest

from conftest import make_retrieval_evidence
from l3s_agent.retrieval.evaluation import (
    GoldQuery,
    evaluate_modes,
    evaluate_query,
    load_gold,
    summarize,
)
from l3s_agent.retrieval.models import RetrievalMode, RetrievalResult


class FixedEngine:
    def __init__(self, evidence):
        self.evidence = evidence

    def search(self, query, k, mode):
        return tuple(
            RetrievalResult(
                final_rank=rank,
                evidence=item,
                retrieval_mode=mode,
                bm25_rank=rank,
                bm25_score=float(10 - rank),
            )
            for rank, item in enumerate(self.evidence[:k], start=1)
        )


def test_page_metrics_handle_duplicate_chunks_and_multiple_gold_pages() -> None:
    engine = FixedEngine(
        [
            make_retrieval_evidence("a", "one", paper_id="paper-a", page=2),
            make_retrieval_evidence("b", "two", paper_id="paper-a", page=2),
            make_retrieval_evidence("c", "three", paper_id="paper-x", page=9),
            make_retrieval_evidence("d", "four", paper_id="paper-b", page=4),
        ]
    )
    gold = GoldQuery(
        "q1",
        "scientific query",
        "solar",
        frozenset({("paper-a", 2), ("paper-b", 4)}),
    )
    metrics = evaluate_query(engine, gold, RetrievalMode.BM25)
    assert metrics.hit_at_3 == 1.0
    assert metrics.hit_at_5 == 1.0
    assert metrics.page_recall_at_3 == 0.5
    assert metrics.page_recall_at_5 == 1.0
    assert metrics.first_relevant_rank == 1
    assert metrics.reciprocal_rank == 1.0


def test_mrr_uses_first_relevant_chunk_rank() -> None:
    engine = FixedEngine(
        [
            make_retrieval_evidence("a", "one", paper_id="paper-x", page=1),
            make_retrieval_evidence("b", "two", paper_id="paper-y", page=2),
            make_retrieval_evidence("c", "three", paper_id="paper-gold", page=3),
        ]
    )
    gold = GoldQuery("q", "query", "wind", frozenset({("paper-gold", 3)}))
    metrics = evaluate_query(engine, gold, RetrievalMode.BM25)
    assert metrics.first_relevant_rank == 3
    assert metrics.reciprocal_rank == pytest.approx(1 / 3)


def test_summary_is_macro_average() -> None:
    engine = FixedEngine([make_retrieval_evidence("a", "one", page=1)])
    hit = evaluate_query(
        engine, GoldQuery("hit", "q", "c", frozenset({("paper-1", 1)})), RetrievalMode.BM25
    )
    miss = evaluate_query(
        engine, GoldQuery("miss", "q", "c", frozenset({("paper-2", 2)})), RetrievalMode.BM25
    )
    assert summarize([hit, miss]) == {
        "hit_at_3": 0.5,
        "hit_at_5": 0.5,
        "page_recall_at_3": 0.5,
        "page_recall_at_5": 0.5,
        "mrr": 0.5,
    }


def test_gold_loader_requires_one_based_pages(tmp_path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "queries": [
                    {
                        "id": "q",
                        "query": "query",
                        "category": "solar",
                        "gold_pages": [{"paper_id": "paper", "page": 0}],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="1-based"):
        load_gold(path)


def test_gold_loader_rejects_a_different_evidence_artifact(tmp_path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source_evidence_sha256": "original",
                "queries": [
                    {
                        "id": "q",
                        "query": "query",
                        "category": "solar",
                        "gold_pages": [{"paper_id": "paper", "page": 1}],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="does not match"):
        load_gold(path, expected_evidence_sha256="replacement")


def test_evaluation_output_uses_plain_mode_values() -> None:
    engine = FixedEngine([make_retrieval_evidence("a", "one", page=1)])
    gold = [GoldQuery("q", "query", "solar", frozenset({("paper-1", 1)}))]
    output = evaluate_modes(engine, gold, modes=(RetrievalMode.BM25,))
    assert output["bm25"]["queries"][0]["mode"] == "bm25"
