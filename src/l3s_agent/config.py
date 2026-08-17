"""Typed, dependency-free configuration loading for the MVP."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class LLMConfig:
    provider: str | None
    text_model: str | None
    verifier_model: str | None
    multimodal_model: str | None


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str | None
    local_only: bool


@dataclass(frozen=True)
class RetrievalConfig:
    lexical_backend: str
    dense_enabled: bool
    fusion: str
    rrf_k: int

    def __post_init__(self) -> None:
        if self.fusion.lower() != "rrf":
            raise ValueError("Phase 1 retrieval fusion must be RRF")
        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be positive")


@dataclass(frozen=True)
class ChunkingConfig:
    target_min_tokens: int
    target_max_tokens: int
    overlap_tokens: int
    cross_page_boundaries: bool

    def __post_init__(self) -> None:
        if self.target_min_tokens <= 0 or self.target_max_tokens < self.target_min_tokens:
            raise ValueError("invalid chunk target range")
        if not 0 <= self.overlap_tokens < self.target_min_tokens:
            raise ValueError("chunk overlap must be smaller than the minimum target")
        if self.cross_page_boundaries:
            raise ValueError("chunks must not cross PDF page boundaries")


@dataclass(frozen=True)
class BudgetConfig:
    max_verifier_calls: int
    max_search_rounds: int
    max_tool_calls: int

    def __post_init__(self) -> None:
        if self.max_verifier_calls != 2:
            raise ValueError("the frozen verification budget is exactly two calls")
        if self.max_search_rounds < 0 or self.max_tool_calls <= 0:
            raise ValueError("invalid execution budget")


@dataclass(frozen=True)
class PathConfig:
    base_corpus_manifest: Path
    base_index_dir: Path
    session_evidence_dir: Path
    trace_dir: Path
    result_dir: Path

    def __post_init__(self) -> None:
        if self.base_index_dir == self.session_evidence_dir:
            raise ValueError("base and session evidence paths must be distinct")


@dataclass(frozen=True)
class IngestionConfig:
    page_image_format: str
    page_image_dpi: int

    def __post_init__(self) -> None:
        if self.page_image_format.lower() != "png":
            raise ValueError("Phase 3 page images must use PNG")
        if self.page_image_dpi <= 0:
            raise ValueError("page image DPI must be positive")


@dataclass(frozen=True)
class LiteratureRankingConfig:
    query_relevance: int
    domain_relevance: int
    accessibility: int
    metadata_completeness: int
    recency: int

    def __post_init__(self) -> None:
        if sum(
            (
                self.query_relevance,
                self.domain_relevance,
                self.accessibility,
                self.metadata_completeness,
                self.recency,
            )
        ) != 100:
            raise ValueError("literature ranking weights must sum to 100")


@dataclass(frozen=True)
class LiteratureConfig:
    topic: str
    modalities: tuple[str, ...]
    queries: tuple[str, ...]
    expansion_queries: tuple[str, ...]
    openalex_api_url: str
    per_query: int
    candidate_min: int
    candidate_max: int
    expansion_increment: int
    expansion_candidate_max: int
    expansion_max_pages: int
    selection_min: int
    selection_target: int
    selection_max: int
    request_timeout_seconds: float
    max_http_attempts: int
    recency_reference_year: int
    allowed_work_types: tuple[str, ...]
    pdf_dir: Path
    openalex_cache_dir: Path
    max_pdf_bytes: int
    min_pdf_bytes: int
    ranking: LiteratureRankingConfig

    def __post_init__(self) -> None:
        if not self.topic.strip() or not self.queries or not self.expansion_queries:
            raise ValueError("literature topic and at least one query are required")
        if not 1 <= self.per_query <= 100:
            raise ValueError("OpenAlex per_query must be between 1 and 100")
        if not 1 <= self.candidate_min <= self.candidate_max:
            raise ValueError("invalid literature candidate bounds")
        if self.expansion_increment <= 0:
            raise ValueError("literature expansion increment must be positive")
        if self.expansion_candidate_max < self.candidate_max:
            raise ValueError("expansion candidate maximum cannot be below initial maximum")
        if self.expansion_max_pages <= 0:
            raise ValueError("literature expansion page limit must be positive")
        if not 1 <= self.selection_min <= self.selection_target <= self.selection_max:
            raise ValueError("invalid literature selection bounds")
        if self.selection_max > self.candidate_max:
            raise ValueError("selection maximum cannot exceed candidate maximum")
        if self.request_timeout_seconds <= 0 or self.max_http_attempts <= 0:
            raise ValueError("invalid OpenAlex HTTP settings")
        if self.min_pdf_bytes <= 0 or self.max_pdf_bytes < self.min_pdf_bytes:
            raise ValueError("invalid PDF size bounds")


@dataclass(frozen=True)
class MLDatasetConfig:
    approved: bool
    adapter: str | None
    path: Path | None

    def __post_init__(self) -> None:
        if not self.approved and (self.adapter is not None or self.path is not None):
            raise ValueError("an unapproved ML dataset cannot define an adapter or path")
        if self.approved and (self.adapter is None or self.path is None):
            raise ValueError("an approved ML dataset requires an adapter and path")


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    chunking: ChunkingConfig
    budgets: BudgetConfig
    paths: PathConfig
    ingestion: IngestionConfig
    literature: LiteratureConfig
    ml_dataset: MLDatasetConfig


def _optional(value: object) -> str | None:
    text = str(value).strip()
    return text or None


def _env_value(env: Mapping[str, str], name: str, fallback: object) -> object:
    return env.get(name, fallback)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def load_config(
    path: str | Path = "config/default.toml",
    *,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load checked-in TOML defaults with explicit environment overrides."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    env = os.environ if environ is None else environ

    llm = raw["llm"]
    embedding = raw["embedding"]
    retrieval = raw["retrieval"]
    chunking = raw["chunking"]
    budgets = raw["budgets"]
    paths = raw["paths"]
    ingestion = raw["ingestion"]
    literature = raw["literature"]
    literature_ranking = literature["ranking"]
    ml_dataset = raw["ml_dataset"]

    approved = _as_bool(_env_value(env, "L3S_ML_DATASET_APPROVED", ml_dataset["approved"]))
    adapter = _optional(_env_value(env, "L3S_ML_DATASET_ADAPTER", ml_dataset["adapter"]))
    dataset_path = _optional(_env_value(env, "L3S_ML_DATASET_PATH", ml_dataset["path"]))

    return AppConfig(
        llm=LLMConfig(
            provider=_optional(_env_value(env, "L3S_LLM_PROVIDER", llm["provider"])),
            text_model=_optional(_env_value(env, "L3S_LLM_TEXT_MODEL", llm["text_model"])),
            verifier_model=_optional(_env_value(env, "L3S_LLM_VERIFIER_MODEL", llm["verifier_model"])),
            multimodal_model=_optional(
                _env_value(env, "L3S_LLM_MULTIMODAL_MODEL", llm["multimodal_model"])
            ),
        ),
        embedding=EmbeddingConfig(
            model=_optional(_env_value(env, "L3S_EMBEDDING_MODEL", embedding["model"])),
            local_only=_as_bool(
                _env_value(env, "L3S_EMBEDDING_LOCAL_ONLY", embedding["local_only"])
            ),
        ),
        retrieval=RetrievalConfig(
            lexical_backend=str(retrieval["lexical_backend"]),
            dense_enabled=_as_bool(retrieval["dense_enabled"]),
            fusion=str(retrieval["fusion"]),
            rrf_k=int(retrieval["rrf_k"]),
        ),
        chunking=ChunkingConfig(
            target_min_tokens=int(chunking["target_min_tokens"]),
            target_max_tokens=int(chunking["target_max_tokens"]),
            overlap_tokens=int(chunking["overlap_tokens"]),
            cross_page_boundaries=_as_bool(chunking["cross_page_boundaries"]),
        ),
        budgets=BudgetConfig(
            max_verifier_calls=int(budgets["max_verifier_calls"]),
            max_search_rounds=int(budgets["max_search_rounds"]),
            max_tool_calls=int(budgets["max_tool_calls"]),
        ),
        paths=PathConfig(
            base_corpus_manifest=Path(paths["base_corpus_manifest"]),
            base_index_dir=Path(paths["base_index_dir"]),
            session_evidence_dir=Path(paths["session_evidence_dir"]),
            trace_dir=Path(paths["trace_dir"]),
            result_dir=Path(paths["result_dir"]),
        ),
        ingestion=IngestionConfig(
            page_image_format=str(ingestion["page_image_format"]),
            page_image_dpi=int(ingestion["page_image_dpi"]),
        ),
        literature=LiteratureConfig(
            topic=str(literature["topic"]),
            modalities=tuple(str(value) for value in literature["modalities"]),
            queries=tuple(str(value) for value in literature["queries"] if str(value).strip()),
            expansion_queries=tuple(
                str(value) for value in literature["expansion_queries"] if str(value).strip()
            ),
            openalex_api_url=str(literature["openalex_api_url"]),
            per_query=int(literature["per_query"]),
            candidate_min=int(literature["candidate_min"]),
            candidate_max=int(literature["candidate_max"]),
            expansion_increment=int(literature["expansion_increment"]),
            expansion_candidate_max=int(literature["expansion_candidate_max"]),
            expansion_max_pages=int(literature["expansion_max_pages"]),
            selection_min=int(literature["selection_min"]),
            selection_target=int(literature["selection_target"]),
            selection_max=int(literature["selection_max"]),
            request_timeout_seconds=float(literature["request_timeout_seconds"]),
            max_http_attempts=int(literature["max_http_attempts"]),
            recency_reference_year=int(literature["recency_reference_year"]),
            allowed_work_types=tuple(str(value) for value in literature["allowed_work_types"]),
            pdf_dir=Path(literature["pdf_dir"]),
            openalex_cache_dir=Path(literature["openalex_cache_dir"]),
            max_pdf_bytes=int(literature["max_pdf_bytes"]),
            min_pdf_bytes=int(literature["min_pdf_bytes"]),
            ranking=LiteratureRankingConfig(
                query_relevance=int(literature_ranking["query_relevance"]),
                domain_relevance=int(literature_ranking["domain_relevance"]),
                accessibility=int(literature_ranking["accessibility"]),
                metadata_completeness=int(literature_ranking["metadata_completeness"]),
                recency=int(literature_ranking["recency"]),
            ),
        ),
        ml_dataset=MLDatasetConfig(
            approved=approved,
            adapter=adapter,
            path=Path(dataset_path) if dataset_path is not None else None,
        ),
    )
