"""Query BM25, dense, and rank-fused Phase 4 indexes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..interfaces import validate_retrieval_scope
from ..models import Evidence
from .embeddings import normalize_rows
from .index import RetrievalIndex
from .models import RetrievalError, RetrievalMode, RetrievalResult


def _ordered(scores: Sequence[float], evidence: Sequence[Evidence]) -> list[tuple[int, float]]:
    return sorted(
        enumerate(scores),
        key=lambda item: (-float(item[1]), evidence[item[0]].evidence_id),
    )


class RetrievalEngine:
    def __init__(self, index: RetrievalIndex) -> None:
        self.index = index

    def search(
        self, query: str, k: int, mode: RetrievalMode | str
    ) -> tuple[RetrievalResult, ...]:
        if not query.strip():
            raise ValueError("retrieval query cannot be blank")
        if k <= 0:
            raise ValueError("retrieval k must be positive")
        try:
            selected_mode = mode if isinstance(mode, RetrievalMode) else RetrievalMode(mode)
        except ValueError as exc:
            raise ValueError(f"unsupported retrieval mode: {mode}") from exc
        evidence = self.index.evidence
        limit = min(k, len(evidence))
        candidate_depth = min(int(self.index.manifest["fusion"]["candidate_depth"]), len(evidence))

        bm25_scores = self.index.bm25.scores(query)
        bm25_ordered = [item for item in _ordered(bm25_scores, evidence) if item[1] > 0]
        bm25_candidates = bm25_ordered[:candidate_depth]
        bm25_diagnostics = {
            row: (rank, float(score))
            for rank, (row, score) in enumerate(bm25_candidates, start=1)
        }
        if selected_mode is RetrievalMode.BM25:
            return tuple(
                RetrievalResult(
                    final_rank=rank,
                    evidence=evidence[row],
                    retrieval_mode=selected_mode,
                    bm25_rank=rank,
                    bm25_score=float(score),
                )
                for rank, (row, score) in enumerate(bm25_ordered[:limit], start=1)
            )

        provider = self.index.embedding_provider
        if provider is None:
            raise RetrievalError("dense and hybrid retrieval require the indexed embedding provider")
        query_vector = normalize_rows(provider.encode_query(query))
        if query_vector.shape != (1, self.index.dense_embeddings.shape[1]):
            raise RetrievalError("query embedding dimension does not match the dense index")
        dense_scores = np.asarray(self.index.dense_embeddings @ query_vector[0], dtype=np.float32)
        dense_ordered = _ordered(dense_scores.tolist(), evidence)
        dense_candidates = dense_ordered[:candidate_depth]
        dense_diagnostics = {
            row: (rank, float(score))
            for rank, (row, score) in enumerate(dense_candidates, start=1)
        }
        if selected_mode is RetrievalMode.DENSE:
            return tuple(
                RetrievalResult(
                    final_rank=rank,
                    evidence=evidence[row],
                    retrieval_mode=selected_mode,
                    dense_rank=rank,
                    dense_score=float(score),
                )
                for rank, (row, score) in enumerate(dense_ordered[:limit], start=1)
            )

        rrf_k = int(self.index.manifest["fusion"]["rrf_k"])
        rows = set(bm25_diagnostics) | set(dense_diagnostics)
        fused: list[tuple[int, float, int]] = []
        for row in rows:
            bm25_rank = bm25_diagnostics.get(row, (None, None))[0]
            dense_rank = dense_diagnostics.get(row, (None, None))[0]
            score = sum(
                1.0 / (rrf_k + rank)
                for rank in (bm25_rank, dense_rank)
                if rank is not None
            )
            best_rank = min(rank for rank in (bm25_rank, dense_rank) if rank is not None)
            fused.append((row, score, best_rank))
        fused.sort(key=lambda item: (-item[1], item[2], evidence[item[0]].evidence_id))
        results: list[RetrievalResult] = []
        for final_rank, (row, rrf_score, _) in enumerate(fused[:limit], start=1):
            bm25_rank, bm25_score = bm25_diagnostics.get(row, (None, None))
            dense_rank, dense_score = dense_diagnostics.get(row, (None, None))
            results.append(
                RetrievalResult(
                    final_rank=final_rank,
                    evidence=evidence[row],
                    retrieval_mode=selected_mode,
                    bm25_rank=bm25_rank,
                    bm25_score=bm25_score,
                    dense_rank=dense_rank,
                    dense_score=dense_score,
                    rrf_score=rrf_score,
                )
            )
        return tuple(results)


class BaseEvidenceRetrievalTool:
    """Phase 1-compatible projection of rich base-corpus retrieval results."""

    def __init__(self, engine: RetrievalEngine, *, mode: RetrievalMode = RetrievalMode.HYBRID):
        self.engine = engine
        self.mode = mode

    def retrieve(
        self,
        *,
        query: str,
        k: int,
        session_id: str | None = None,
        include_session_evidence: bool = False,
    ) -> Sequence[Evidence]:
        validate_retrieval_scope(
            session_id=session_id,
            include_session_evidence=include_session_evidence,
        )
        if include_session_evidence:
            raise RetrievalError("session evidence retrieval is not implemented in Phase 4")
        return tuple(result.evidence for result in self.engine.search(query, k, self.mode))
