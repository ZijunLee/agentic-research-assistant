from collections.abc import Callable
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from l3s_agent.literature.discovery import normalize_title
from l3s_agent.literature.models import (
    AuthorRecord,
    LiteratureCandidate,
    OpenAccessInfo,
    QueryMatch,
)
from l3s_agent.models import CorpusScope, Evidence, EvidenceModality
from l3s_agent.retrieval.embeddings import normalize_rows


class FakeEmbeddingProvider:
    provider_name = "fake"

    def __init__(
        self,
        vectors: dict[str, tuple[float, ...]],
        *,
        model_id: str = "fake-model",
        model_revision: str = "fake-revision-1",
        max_sequence_length: int = 1000,
    ) -> None:
        self.vectors = vectors
        self.model_id = model_id
        self.model_revision = model_revision
        self.max_sequence_length = max_sequence_length
        self.dimension = len(next(iter(vectors.values())))

    def token_count(self, text: str) -> int:
        return len(text.split()) + 2

    def encode_documents(self, texts):
        return normalize_rows(np.asarray([self.vectors[text] for text in texts], dtype=np.float32))

    def encode_query(self, query: str):
        return normalize_rows(np.asarray([self.vectors[query]], dtype=np.float32))


def make_retrieval_evidence(
    evidence_id: str,
    content: str,
    *,
    paper_id: str = "paper-1",
    page: int = 1,
    title: str = "Scientific paper",
    section: str | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        paper_id=paper_id,
        title=title,
        page=page,
        section=section,
        modality=EvidenceModality.TEXT,
        source_id=paper_id.removeprefix("paper_"),
        corpus_scope=CorpusScope.BASE,
        content=content,
    )


@pytest.fixture
def retrieval_artifact_factory(tmp_path: Path):
    def create(evidence: list[Evidence]) -> Path:
        root = tmp_path / f"source-{len(list(tmp_path.iterdir()))}"
        root.mkdir()
        evidence_path = root / "evidence.jsonl"
        rows = []
        for item in evidence:
            rows.append(
                {
                    "evidence_id": item.evidence_id,
                    "paper_id": item.paper_id,
                    "title": item.title,
                    "page": item.page,
                    "section": item.section,
                    "modality": item.modality.value,
                    "source_id": item.source_id,
                    "corpus_scope": item.corpus_scope.value,
                    "content": item.content,
                }
            )
        evidence_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        evidence_hash = sha256(evidence_path.read_bytes()).hexdigest()
        (root / "ingestion_manifest.json").write_text(
            json.dumps(
                {
                    "summary": {"complete": True},
                    "artifacts": {"evidence_jsonl_sha256": evidence_hash},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return evidence_path

    return create


@pytest.fixture
def candidate_factory() -> Callable[..., LiteratureCandidate]:
    def make_candidate(**overrides: object) -> LiteratureCandidate:
        title = str(overrides.pop("title", "Weather Effects on Solar Power Generation"))
        values = {
            "openalex_id": "W1",
            "openalex_url": "https://openalex.org/W1",
            "title": title,
            "normalized_title": normalize_title(title),
            "authors": (AuthorRecord("Ada Example", "A1"),),
            "year": 2022,
            "doi": "10.1000/example",
            "work_type": "article",
            "language": "en",
            "source_url": "https://publisher.example/work",
            "open_access": OpenAccessInfo(
                is_oa=True,
                status="green",
                landing_page_url="https://repository.example/work",
                pdf_urls=("https://repository.example/work.pdf",),
            ),
            "cited_by_count": 10,
            "abstract": "Solar generation changes with meteorological conditions and cloud cover.",
            "topics": ("Renewable Energy", "Weather and Climate"),
            "keywords": ("solar irradiance",),
            "query_matches": (QueryMatch("weather solar power generation", 1, 10.0),),
            "is_retracted": False,
        }
        values.update(overrides)
        return LiteratureCandidate(**values)  # type: ignore[arg-type]

    return make_candidate
