"""Small page-level retrieval evaluator for the manually verified gold set."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .engine import RetrievalEngine
from .models import RetrievalMode


PageKey = tuple[str, int]


@dataclass(frozen=True)
class GoldQuery:
    query_id: str
    query: str
    category: str
    gold_pages: frozenset[PageKey]

    def __post_init__(self) -> None:
        if not self.query_id.strip() or not self.query.strip() or not self.category.strip():
            raise ValueError("gold query identifiers and text cannot be blank")
        if not self.gold_pages or any(page < 1 for _, page in self.gold_pages):
            raise ValueError("gold relevance requires at least one 1-based page")


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    mode: RetrievalMode
    hit_at_3: float
    hit_at_5: float
    page_recall_at_3: float
    page_recall_at_5: float
    reciprocal_rank: float
    first_relevant_rank: int | None


def load_gold(
    path: Path, *, expected_evidence_sha256: str | None = None
) -> tuple[GoldQuery, ...]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.0":
        raise ValueError("unsupported retrieval gold schema")
    source_sha256 = value.get("source_evidence_sha256")
    if expected_evidence_sha256 is not None and source_sha256 != expected_evidence_sha256:
        raise ValueError("retrieval gold set does not match the indexed evidence artifact")
    queries: list[GoldQuery] = []
    for item in value.get("queries", []):
        pages = frozenset(
            (str(page["paper_id"]), int(page["page"])) for page in item["gold_pages"]
        )
        queries.append(
            GoldQuery(
                query_id=str(item["id"]),
                query=str(item["query"]),
                category=str(item["category"]),
                gold_pages=pages,
            )
        )
    if not queries or len({item.query_id for item in queries}) != len(queries):
        raise ValueError("retrieval gold set must contain unique queries")
    return tuple(queries)


def evaluate_query(
    engine: RetrievalEngine, gold: GoldQuery, mode: RetrievalMode
) -> QueryMetrics:
    results = engine.search(gold.query, 5, mode)
    page_keys = [(item.evidence.paper_id, item.evidence.page) for item in results]

    def relevant_at(k: int) -> set[PageKey]:
        return set(page_keys[:k]) & set(gold.gold_pages)

    first_rank = next(
        (rank for rank, page in enumerate(page_keys, start=1) if page in gold.gold_pages),
        None,
    )
    return QueryMetrics(
        query_id=gold.query_id,
        mode=mode,
        hit_at_3=float(bool(relevant_at(3))),
        hit_at_5=float(bool(relevant_at(5))),
        page_recall_at_3=len(relevant_at(3)) / len(gold.gold_pages),
        page_recall_at_5=len(relevant_at(5)) / len(gold.gold_pages),
        reciprocal_rank=0.0 if first_rank is None else 1.0 / first_rank,
        first_relevant_rank=first_rank,
    )


def summarize(metrics: Sequence[QueryMetrics]) -> dict[str, float]:
    if not metrics:
        raise ValueError("cannot summarize an empty retrieval evaluation")
    return {
        "hit_at_3": sum(item.hit_at_3 for item in metrics) / len(metrics),
        "hit_at_5": sum(item.hit_at_5 for item in metrics) / len(metrics),
        "page_recall_at_3": sum(item.page_recall_at_3 for item in metrics) / len(metrics),
        "page_recall_at_5": sum(item.page_recall_at_5 for item in metrics) / len(metrics),
        "mrr": sum(item.reciprocal_rank for item in metrics) / len(metrics),
    }


def evaluate_modes(
    engine: RetrievalEngine,
    gold_queries: Sequence[GoldQuery],
    modes: Sequence[RetrievalMode] = tuple(RetrievalMode),
) -> Mapping[str, object]:
    output: dict[str, object] = {}
    for mode in modes:
        per_query = [evaluate_query(engine, item, mode) for item in gold_queries]
        output[mode.value] = {
            "summary": summarize(per_query),
            "queries": [
                {
                    "query_id": item.query_id,
                    "mode": item.mode.value,
                    "hit_at_3": item.hit_at_3,
                    "hit_at_5": item.hit_at_5,
                    "page_recall_at_3": item.page_recall_at_3,
                    "page_recall_at_5": item.page_recall_at_5,
                    "reciprocal_rank": item.reciprocal_rank,
                    "first_relevant_rank": item.first_relevant_rank,
                }
                for item in per_query
            ],
        }
    return output
