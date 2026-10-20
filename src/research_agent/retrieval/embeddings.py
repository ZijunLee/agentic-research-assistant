"""Configurable dense-embedding providers with no import-time model loading."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .models import RetrievalError


FloatMatrix = NDArray[np.float32]


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    @property
    def max_sequence_length(self) -> int: ...

    @property
    def dimension(self) -> int: ...

    def token_count(self, text: str) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def encode_query(self, query: str) -> FloatMatrix: ...


def normalize_rows(values: NDArray[np.floating]) -> FloatMatrix:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise RetrievalError("embedding provider returned an invalid matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(matrix)) or np.any(norms == 0):
        raise RetrievalError("embedding provider returned non-finite or zero vectors")
    return np.asarray(matrix / norms, dtype=np.float32)


def validate_document_lengths(provider: EmbeddingProvider, texts: Sequence[str]) -> None:
    maximum = provider.max_sequence_length
    if maximum <= 0:
        raise RetrievalError("embedding provider has an invalid maximum sequence length")
    lengths = [provider.token_count(text) for text in texts]
    overlength = [
        (index, length) for index, length in enumerate(lengths) if length > maximum
    ]
    if overlength:
        index, length = overlength[0]
        raise RetrievalError(
            f"evidence row {index} requires {length} model tokens but the configured "
            f"provider supports {maximum}; refusing silent truncation"
        )


class SentenceTransformersEmbeddingProvider:
    """Production adapter; construction is the explicit model-load boundary."""

    provider_name = "sentence-transformers"

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        local_files_only: bool,
        trust_remote_code: bool = False,
        device: str = "cpu",
    ) -> None:
        if not model_id.strip() or not model_revision.strip():
            raise ValueError("dense retrieval requires a model ID and immutable revision")
        if trust_remote_code:
            raise ValueError("the frozen Phase 4 production provider forbids remote model code")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RetrievalError(
                "sentence-transformers is not installed; install the approved Phase 4 dependencies"
            ) from exc
        self._model_id = model_id
        self._model_revision = model_revision
        self._model = SentenceTransformer(
            model_id,
            revision=model_revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
            device=device,
        )
        self._max_sequence_length = int(self._model.max_seq_length)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._model_revision

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def dimension(self) -> int:
        return self._dimension

    def token_count(self, text: str) -> int:
        encoded = self._model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
        )
        return len(encoded["input_ids"])

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        validate_document_lengths(self, texts)
        method = getattr(self._model, "encode_document", None)
        if method is None:
            method = self._model.encode
        return normalize_rows(
            method(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

    def encode_query(self, query: str) -> FloatMatrix:
        if self.token_count(query) > self.max_sequence_length:
            raise RetrievalError("query exceeds the embedding model maximum sequence length")
        method = getattr(self._model, "encode_query", None)
        if method is None:
            method = self._model.encode
        return normalize_rows(
            method(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )
