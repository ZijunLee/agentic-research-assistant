"""Typed Phase 4 retrieval contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..models import Evidence


class RetrievalMode(str, Enum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class RetrievalResult:
    final_rank: int
    evidence: Evidence
    retrieval_mode: RetrievalMode
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    rrf_score: float | None = None

    def __post_init__(self) -> None:
        if self.final_rank < 1:
            raise ValueError("retrieval ranks are 1-based")
        for rank in (self.bm25_rank, self.dense_rank):
            if rank is not None and rank < 1:
                raise ValueError("component retrieval ranks are 1-based")
        if self.retrieval_mode is RetrievalMode.BM25 and self.bm25_rank is None:
            raise ValueError("BM25 results require BM25 diagnostics")
        if self.retrieval_mode is RetrievalMode.DENSE and self.dense_rank is None:
            raise ValueError("dense results require dense diagnostics")
        if self.retrieval_mode is RetrievalMode.HYBRID and self.rrf_score is None:
            raise ValueError("hybrid results require an RRF score")


class RetrievalError(RuntimeError):
    """Clear retrieval/index failure without silently degrading behavior."""
