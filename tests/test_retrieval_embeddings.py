import sys
import types

import numpy as np
import pytest

from l3s_agent.retrieval.embeddings import (
    SentenceTransformersEmbeddingProvider,
    normalize_rows,
)
from l3s_agent.retrieval.models import RetrievalError


class FakeSentenceTransformer:
    init_kwargs = None

    def __init__(self, model_id, **kwargs):
        self.model_id = model_id
        self.max_seq_length = 8192
        self.tokenizer = lambda text, **_: {"input_ids": text.split()}
        self.document_calls = []
        self.query_calls = []
        type(self).init_kwargs = kwargs

    def get_sentence_embedding_dimension(self):
        return 2

    def encode_document(self, texts, **kwargs):
        self.document_calls.append((texts, kwargs))
        return np.asarray([[3.0, 4.0] for _ in texts], dtype=np.float32)

    def encode_query(self, query, **kwargs):
        self.query_calls.append((query, kwargs))
        return np.asarray([3.0, 4.0], dtype=np.float32)


def test_production_adapter_pins_revision_and_forbids_remote_code(monkeypatch) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    provider = SentenceTransformersEmbeddingProvider(
        model_id="Alibaba-NLP/gte-modernbert-base",
        model_revision="verified-commit",
        local_files_only=True,
        trust_remote_code=False,
    )
    assert FakeSentenceTransformer.init_kwargs == {
        "revision": "verified-commit",
        "trust_remote_code": False,
        "local_files_only": True,
        "device": "cpu",
    }
    documents = provider.encode_documents(["document text"])
    query = provider.encode_query("query text")
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(query, axis=1), 1.0)
    assert provider._model.document_calls[0][1]["normalize_embeddings"] is True
    assert provider._model.query_calls[0][1]["normalize_embeddings"] is True

    with pytest.raises(ValueError, match="forbids remote"):
        SentenceTransformersEmbeddingProvider(
            model_id="model",
            model_revision="revision",
            local_files_only=True,
            trust_remote_code=True,
        )


def test_normalization_rejects_zero_or_nonfinite_vectors() -> None:
    with pytest.raises(RetrievalError, match="zero vectors"):
        normalize_rows(np.asarray([[0.0, 0.0]], dtype=np.float32))
    with pytest.raises(RetrievalError, match="non-finite"):
        normalize_rows(np.asarray([[np.nan, 1.0]], dtype=np.float32))
