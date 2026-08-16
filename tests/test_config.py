from pathlib import Path

import pytest

from l3s_agent.config import MLDatasetConfig, PathConfig, load_config


CONFIG_PATH = Path(__file__).parents[1] / "config" / "default.toml"


def test_defaults_keep_models_unset_and_freeze_core_decisions() -> None:
    config = load_config(CONFIG_PATH, environ={})

    assert config.llm.provider is None
    assert config.llm.text_model is None
    assert config.llm.verifier_model is None
    assert config.llm.multimodal_model is None
    assert config.embedding.model is None
    assert config.embedding.local_only is True
    assert config.retrieval.fusion == "rrf"
    assert config.budgets.max_verifier_calls == 2
    assert config.chunking.cross_page_boundaries is False
    assert config.ml_dataset == MLDatasetConfig(approved=False, adapter=None, path=None)


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

