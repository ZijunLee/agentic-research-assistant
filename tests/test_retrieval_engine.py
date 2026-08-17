import pytest

from conftest import FakeEmbeddingProvider, make_retrieval_evidence
from l3s_agent.interfaces import EvidenceRetrievalTool
from l3s_agent.retrieval.engine import BaseEvidenceRetrievalTool, RetrievalEngine
from l3s_agent.retrieval.index import build_retrieval_index
from l3s_agent.retrieval.models import RetrievalError, RetrievalMode


def make_engine(retrieval_artifact_factory, tmp_path):
    contents = [
        "solar irradiance generation",
        "wind numerical weather prediction",
        "cloud variability photovoltaic output",
    ]
    evidence = [
        make_retrieval_evidence("ev-a", contents[0], paper_id="paper-a", page=1),
        make_retrieval_evidence("ev-b", contents[1], paper_id="paper-b", page=2),
        make_retrieval_evidence("ev-c", contents[2], paper_id="paper-c", page=3),
    ]
    query = "solar query"
    provider = FakeEmbeddingProvider(
        {
            contents[0]: (1.0, 0.0),
            contents[1]: (0.0, 1.0),
            contents[2]: (0.8, 0.2),
            query: (1.0, 0.0),
        }
    )
    source = retrieval_artifact_factory(evidence)
    index = build_retrieval_index(
        evidence_path=source,
        output_dir=tmp_path / "index",
        embedding_provider=provider,
    )
    return RetrievalEngine(index), query


def test_dense_search_uses_normalized_dot_product(retrieval_artifact_factory, tmp_path) -> None:
    engine, query = make_engine(retrieval_artifact_factory, tmp_path)
    results = engine.search(query, 3, RetrievalMode.DENSE)
    assert [item.evidence.evidence_id for item in results] == ["ev-a", "ev-c", "ev-b"]
    assert results[0].dense_score == pytest.approx(1.0)
    assert all(item.bm25_score is None and item.rrf_score is None for item in results)


def test_hybrid_rrf_uses_component_ranks(retrieval_artifact_factory, tmp_path) -> None:
    engine, query = make_engine(retrieval_artifact_factory, tmp_path)
    results = engine.search(query, 3, RetrievalMode.HYBRID)
    first = results[0]
    assert first.evidence.evidence_id == "ev-a"
    assert first.bm25_rank == 1
    assert first.dense_rank == 1
    assert first.rrf_score == pytest.approx(2 / 61)
    assert results[1].rrf_score == pytest.approx(1 / 62)


def test_title_and_section_do_not_influence_ranking(retrieval_artifact_factory, tmp_path) -> None:
    content = "identical solar output content"
    evidence = [
        make_retrieval_evidence(
            "ev-b", content, title="Solar Query Exact Match", section="Solar Query"
        ),
        make_retrieval_evidence("ev-a", content, title="Unrelated", section=None),
    ]
    query = "solar query"
    provider = FakeEmbeddingProvider({content: (1.0, 0.0), query: (1.0, 0.0)})
    source = retrieval_artifact_factory(evidence)
    index = build_retrieval_index(
        evidence_path=source,
        output_dir=tmp_path / "index",
        embedding_provider=provider,
    )
    engine = RetrievalEngine(index)
    assert [item.evidence.evidence_id for item in engine.search(query, 2, "bm25")] == [
        "ev-a",
        "ev-b",
    ]
    assert [item.evidence.evidence_id for item in engine.search(query, 2, "dense")] == [
        "ev-a",
        "ev-b",
    ]


def test_protocol_projection_and_session_rejection(retrieval_artifact_factory, tmp_path) -> None:
    engine, query = make_engine(retrieval_artifact_factory, tmp_path)
    tool = BaseEvidenceRetrievalTool(engine, mode=RetrievalMode.HYBRID)
    assert isinstance(tool, EvidenceRetrievalTool)
    evidence = tool.retrieve(query=query, k=2)
    assert [item.evidence_id for item in evidence] == ["ev-a", "ev-c"]
    with pytest.raises(ValueError, match="concrete session_id"):
        tool.retrieve(query=query, k=2, include_session_evidence=True)
    with pytest.raises(RetrievalError, match="not implemented"):
        tool.retrieve(
            query=query,
            k=2,
            session_id="session-1",
            include_session_evidence=True,
        )


def test_query_validation(retrieval_artifact_factory, tmp_path) -> None:
    engine, _ = make_engine(retrieval_artifact_factory, tmp_path)
    with pytest.raises(ValueError, match="blank"):
        engine.search(" ", 3, "bm25")
    with pytest.raises(ValueError, match="positive"):
        engine.search("solar", 0, "bm25")
