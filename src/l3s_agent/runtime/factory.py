"""Production Phase 5B wiring over the existing bounded runtime and Phase 4 index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..events import SafeEventSink
from ..interfaces import EvidenceRetrievalTool, LLMProvider
from ..providers import OpenAIResponsesProvider
from ..retrieval.embeddings import SentenceTransformersEmbeddingProvider
from ..retrieval.engine import BaseEvidenceRetrievalTool, RetrievalEngine
from ..retrieval.index import RetrievalIndex
from ..retrieval.models import RetrievalError, RetrievalMode
from .orchestrator import ResearchOrchestrator
from .registry import ToolRegistry
from .verifier import EvidenceVerifier


def assemble_runtime(
    *,
    config: AppConfig,
    provider: LLMProvider,
    retrieval: EvidenceRetrievalTool,
    event_sink: SafeEventSink | None = None,
) -> ResearchOrchestrator:
    """Assemble Phase 5A with retrieval as the only available production tool."""

    return ResearchOrchestrator(
        research_provider=provider,
        verifier=EvidenceVerifier(provider),
        tools=ToolRegistry(retrieval=retrieval),
        budgets=config.budgets,
        event_sink=event_sink,
    )


def _validate_frozen_retrieval_config(config: AppConfig, index: RetrievalIndex) -> None:
    bm25 = index.manifest["bm25"]
    fusion = index.manifest["fusion"]
    expected = {
        "k1": config.retrieval.bm25_k1,
        "b": config.retrieval.bm25_b,
        "rrf_k": config.retrieval.rrf_k,
        "candidate_depth": config.retrieval.candidate_depth,
    }
    actual = {
        "k1": float(bm25["k1"]),
        "b": float(bm25["b"]),
        "rrf_k": int(fusion["rrf_k"]),
        "candidate_depth": int(fusion["candidate_depth"]),
    }
    if actual != expected:
        raise RetrievalError("production retrieval index does not match frozen configuration")


def build_production_runtime(
    *,
    config: AppConfig,
    evidence_path: Path = Path("data/cache/base_index/evidence.jsonl"),
    index_dir: Path = Path("data/cache/retrieval/base"),
    client: Any | None = None,
    event_sink: SafeEventSink | None = None,
) -> ResearchOrchestrator:
    """Load the checksummed local index and construct the real Phase 5B runtime."""

    if config.embedding.model is None or config.embedding.revision is None:
        raise RetrievalError("production retrieval requires the frozen embedding revision")
    embedding = SentenceTransformersEmbeddingProvider(
        model_id=config.embedding.model,
        model_revision=config.embedding.revision,
        local_files_only=True,
        trust_remote_code=False,
        device="mps",
    )
    index = RetrievalIndex.load(
        evidence_path=evidence_path,
        index_dir=index_dir,
        embedding_provider=embedding,
    )
    _validate_frozen_retrieval_config(config, index)
    retrieval = BaseEvidenceRetrievalTool(
        RetrievalEngine(index), mode=RetrievalMode.HYBRID
    )
    provider = OpenAIResponsesProvider(config.llm, client=client, event_sink=event_sink)
    return assemble_runtime(
        config=config, provider=provider, retrieval=retrieval, event_sink=event_sink
    )
