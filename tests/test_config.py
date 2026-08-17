from pathlib import Path

import pytest

from l3s_agent.config import MLDatasetConfig, PathConfig, load_config


CONFIG_PATH = Path(__file__).parents[1] / "config" / "default.toml"


def test_defaults_keep_llms_unset_and_freeze_core_decisions() -> None:
    config = load_config(CONFIG_PATH, environ={})

    assert config.llm.provider is None
    assert config.llm.text_model is None
    assert config.llm.verifier_model is None
    assert config.llm.multimodal_model is None
    assert config.retrieval.fusion == "rrf"
    assert config.budgets.max_verifier_calls == 2
    assert config.budgets.max_tool_calls == 6
    assert config.budgets.max_agent_decisions == 10
    assert config.budgets.max_literature_searches == 1
    assert config.budgets.max_page_inspections == 2
    assert config.budgets.max_python_calls == 1
    assert config.budgets.max_follow_up_tool_calls == 3
    assert config.budgets.default_retrieval_k == 5
    assert config.budgets.max_base_evidence == 40
    assert config.budgets.max_session_evidence == 20
    assert config.chunking.cross_page_boundaries is False
    assert config.embedding.model == "Alibaba-NLP/gte-modernbert-base"
    assert config.embedding.revision == "e7f32e3c00f91d699e8c43b53106206bcc72bb22"
    assert config.embedding.local_only is True
    assert config.embedding.trust_remote_code is False
    assert config.retrieval.rrf_k == 60
    assert config.retrieval.candidate_depth == 50
    assert config.retrieval.bm25_k1 == 1.5
    assert config.retrieval.bm25_b == 0.75
    assert config.paths.base_corpus_manifest == Path("data/manifests/base_corpus.json")
    assert config.ingestion.page_image_format == "png"
    assert config.ingestion.page_image_dpi == 144
    assert config.ml_dataset == MLDatasetConfig(approved=False, adapter=None, path=None)
    assert config.literature.topic == "Weather and climate impacts on renewable energy"
    assert config.literature.modalities == ("solar", "wind")
    assert config.literature.selection_target == 10
    assert config.literature.selection_min == 8
    assert config.literature.selection_max == 12
    assert config.literature.candidate_max == 50
    assert config.literature.expansion_increment == 10
    assert config.literature.expansion_candidate_max == 90
    assert len(config.literature.expansion_queries) == 3
    assert config.literature.ranking.query_relevance == 35
    assert config.literature.ranking.domain_relevance == 35
    assert config.literature.ranking.accessibility == 20


def test_environment_overrides_provider_models_without_hard_coding() -> None:
    config = load_config(
        CONFIG_PATH,
        environ={
            "L3S_LLM_PROVIDER": "test-provider",
            "L3S_LLM_TEXT_MODEL": "text-model",
            "L3S_LLM_VERIFIER_MODEL": "verification-model",
            "L3S_LLM_MULTIMODAL_MODEL": "vision-model",
            "L3S_EMBEDDING_MODEL": "local-embedding-model",
        },
    )

    assert config.llm.provider == "test-provider"
    assert config.llm.verifier_model == "verification-model"
    assert config.embedding.model == "local-embedding-model"


def test_base_and_session_paths_must_be_distinct() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        PathConfig(
            base_corpus_manifest=Path("manifest.json"),
            base_index_dir=Path("same"),
            session_evidence_dir=Path("same"),
            trace_dir=Path("traces"),
            result_dir=Path("results"),
        )


def test_unapproved_ml_dataset_rejects_implementation_details() -> None:
    with pytest.raises(ValueError, match="unapproved"):
        MLDatasetConfig(approved=False, adapter="csv", path=Path("dataset.csv"))
