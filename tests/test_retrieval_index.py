import json
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeEmbeddingProvider, make_retrieval_evidence
from l3s_agent.retrieval.index import RetrievalIndex, build_retrieval_index
from l3s_agent.retrieval.models import RetrievalError


def build_fixture(retrieval_artifact_factory, tmp_path: Path):
    evidence = [
        make_retrieval_evidence("ev-a", "solar irradiance power", page=2),
        make_retrieval_evidence("ev-b", "wind forecast weather", paper_id="paper-2", page=4),
    ]
    source = retrieval_artifact_factory(evidence)
    provider = FakeEmbeddingProvider(
        {
            "solar irradiance power": (1.0, 0.0),
            "wind forecast weather": (0.0, 1.0),
            "solar query": (1.0, 0.0),
        }
    )
    output = tmp_path / "retrieval-index"
    return evidence, source, provider, output


def test_index_build_preserves_row_alignment_and_provenance(
    retrieval_artifact_factory, tmp_path: Path
) -> None:
    evidence, source, provider, output = build_fixture(retrieval_artifact_factory, tmp_path)
    index = build_retrieval_index(
        evidence_path=source, output_dir=output, embedding_provider=provider
    )
    assert [item.evidence_id for item in index.evidence] == ["ev-a", "ev-b"]
    assert np.argmax(index.dense_embeddings[:, 0]) == 0
    assert index.evidence[0] == evidence[0]
    assert index.manifest["embedding"] == {
        "provider": "fake",
        "model_id": "fake-model",
        "model_revision": "fake-revision-1",
        "max_sequence_length": 1000,
        "dimension": 2,
        "dtype": "float32",
        "normalized": True,
        "document_method": "encode_document",
        "query_method": "encode_query",
        "scored_field": "content",
        "trust_remote_code": False,
    }
    assert "solar irradiance power" not in (output / "evidence_ids.json").read_text()


def test_index_refuses_overwrite(retrieval_artifact_factory, tmp_path: Path) -> None:
    _, source, provider, output = build_fixture(retrieval_artifact_factory, tmp_path)
    build_retrieval_index(evidence_path=source, output_dir=output, embedding_provider=provider)
    with pytest.raises(RetrievalError, match="overwrite"):
        build_retrieval_index(evidence_path=source, output_dir=output, embedding_provider=provider)


def test_index_rejects_corruption(retrieval_artifact_factory, tmp_path: Path) -> None:
    _, source, provider, output = build_fixture(retrieval_artifact_factory, tmp_path)
    build_retrieval_index(evidence_path=source, output_dir=output, embedding_provider=provider)
    (output / "dense_embeddings.npy").write_bytes(b"corrupted")
    with pytest.raises(RetrievalError, match="missing or corrupted"):
        RetrievalIndex.load(
            evidence_path=source, index_dir=output, embedding_provider=provider
        )


def test_index_rejects_source_checksum_change(retrieval_artifact_factory, tmp_path: Path) -> None:
    _, source, provider, output = build_fixture(retrieval_artifact_factory, tmp_path)
    build_retrieval_index(evidence_path=source, output_dir=output, embedding_provider=provider)
    source.write_text(source.read_text() + "\n")
    with pytest.raises(RetrievalError, match="Phase 3 evidence checksum"):
        RetrievalIndex.load(
            evidence_path=source, index_dir=output, embedding_provider=provider
        )


def test_index_rejects_provider_revision_mismatch(
    retrieval_artifact_factory, tmp_path: Path
) -> None:
    _, source, provider, output = build_fixture(retrieval_artifact_factory, tmp_path)
    build_retrieval_index(evidence_path=source, output_dir=output, embedding_provider=provider)
    other = FakeEmbeddingProvider(provider.vectors, model_revision="different")
    with pytest.raises(RetrievalError, match="model_revision"):
        RetrievalIndex.load(evidence_path=source, index_dir=output, embedding_provider=other)


def test_index_refuses_silent_document_truncation(
    retrieval_artifact_factory, tmp_path: Path
) -> None:
    text = "one two three four"
    source = retrieval_artifact_factory([make_retrieval_evidence("ev-a", text)])
    provider = FakeEmbeddingProvider({text: (1.0, 0.0)}, max_sequence_length=5)
    with pytest.raises(RetrievalError, match="refusing silent truncation"):
        build_retrieval_index(
            evidence_path=source,
            output_dir=tmp_path / "index",
            embedding_provider=provider,
        )


def test_index_requires_explicit_model_revision(retrieval_artifact_factory, tmp_path: Path) -> None:
    text = "solar output"
    source = retrieval_artifact_factory([make_retrieval_evidence("ev-a", text)])
    provider = FakeEmbeddingProvider({text: (1.0, 0.0)}, model_revision="")
    with pytest.raises(RetrievalError, match="immutable model revision"):
        build_retrieval_index(
            evidence_path=source,
            output_dir=tmp_path / "index",
            embedding_provider=provider,
        )
