"""Build, persist, and validate deterministic Phase 4 retrieval indexes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..models import CorpusScope, Evidence, EvidenceModality
from .bm25 import BM25Index, TOKENIZER_VERSION
from .embeddings import EmbeddingProvider, normalize_rows, validate_document_lengths
from .models import RetrievalError


INDEX_SCHEMA_VERSION = "1.0"
BM25_IMPLEMENTATION = "local_okapi_v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"cannot read retrieval artifact {path.name}") from exc


def load_evidence(evidence_path: Path) -> tuple[Evidence, ...]:
    try:
        lines = evidence_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetrievalError("cannot read Phase 3 evidence.jsonl") from exc
    records: list[Evidence] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
            evidence = Evidence(
                evidence_id=str(value["evidence_id"]),
                paper_id=str(value["paper_id"]),
                title=str(value["title"]),
                page=int(value["page"]),
                section=(str(value["section"]) if value.get("section") is not None else None),
                modality=EvidenceModality(str(value["modality"])),
                source_id=str(value["source_id"]),
                corpus_scope=CorpusScope(str(value["corpus_scope"])),
                content=str(value["content"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RetrievalError(f"invalid evidence record on line {line_number}") from exc
        if evidence.evidence_id in seen_ids:
            raise RetrievalError(f"duplicate evidence_id: {evidence.evidence_id}")
        if evidence.modality is not EvidenceModality.TEXT:
            raise RetrievalError("Phase 4 retrieval input must contain text Evidence only")
        if evidence.corpus_scope is not CorpusScope.BASE:
            raise RetrievalError("Phase 4 retrieval input must contain base Evidence only")
        seen_ids.add(evidence.evidence_id)
        records.append(evidence)
    if not records:
        raise RetrievalError("Phase 3 evidence.jsonl contains no records")
    return tuple(records)


def _validate_ingestion_source(evidence_path: Path) -> tuple[Path, str, Mapping[str, Any]]:
    manifest_path = evidence_path.parent / "ingestion_manifest.json"
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("summary", {}).get("complete") is not True:
        raise RetrievalError("Phase 3 ingestion manifest is incomplete")
    recorded = manifest.get("artifacts", {}).get("evidence_jsonl_sha256")
    actual = file_sha256(evidence_path)
    if recorded != actual:
        raise RetrievalError("Phase 3 evidence checksum does not match its ingestion manifest")
    return manifest_path, actual, manifest


@dataclass(frozen=True)
class RetrievalIndex:
    evidence: tuple[Evidence, ...]
    bm25: BM25Index
    dense_embeddings: NDArray[np.float32]
    manifest: Mapping[str, Any]
    embedding_provider: EmbeddingProvider | None = None

    @classmethod
    def load(
        cls,
        *,
        evidence_path: Path,
        index_dir: Path,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "RetrievalIndex":
        manifest_path = index_dir / "index_manifest.json"
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise RetrievalError("unsupported retrieval index schema")
        if manifest.get("complete") is not True:
            raise RetrievalError("retrieval index is incomplete")
        ingestion_path, evidence_hash, _ = _validate_ingestion_source(evidence_path)
        source = manifest.get("source", {})
        if source.get("evidence_jsonl_sha256") != evidence_hash:
            raise RetrievalError("retrieval index is stale for the current evidence.jsonl")
        if source.get("ingestion_manifest_sha256") != file_sha256(ingestion_path):
            raise RetrievalError("retrieval index is stale for the ingestion manifest")
        artifacts = manifest.get("artifacts", {})
        required = {
            "evidence_ids": index_dir / "evidence_ids.json",
            "bm25": index_dir / "bm25.json",
            "dense_embeddings": index_dir / "dense_embeddings.npy",
        }
        for name, path in required.items():
            if not path.is_file() or file_sha256(path) != artifacts.get(f"{name}_sha256"):
                raise RetrievalError(f"retrieval index artifact is missing or corrupted: {name}")
        evidence = load_evidence(evidence_path)
        evidence_ids = _load_json(required["evidence_ids"])
        expected_ids = [item.evidence_id for item in evidence]
        if evidence_ids != expected_ids or source.get("evidence_count") != len(evidence):
            raise RetrievalError("retrieval index Evidence ordering is stale or corrupted")
        raw_bm25 = _load_json(required["bm25"])
        if not isinstance(raw_bm25, dict):
            raise RetrievalError("invalid BM25 index")
        try:
            bm25 = BM25Index.from_dict(raw_bm25)
            dense = np.load(required["dense_embeddings"], allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RetrievalError("cannot load retrieval index arrays") from exc
        dense = np.asarray(dense, dtype=np.float32)
        if dense.shape != (len(evidence), int(manifest["embedding"]["dimension"])):
            raise RetrievalError("dense embedding shape does not match index metadata")
        if len(bm25.document_lengths) != len(evidence):
            raise RetrievalError("BM25 row count does not match Evidence ordering")
        norms = np.linalg.norm(dense, axis=1)
        if not np.all(np.isfinite(dense)) or not np.allclose(norms, 1.0, atol=1e-5):
            raise RetrievalError("dense embeddings are not valid normalized vectors")
        if embedding_provider is not None:
            metadata = manifest["embedding"]
            actual = {
                "provider": embedding_provider.provider_name,
                "model_id": embedding_provider.model_id,
                "model_revision": embedding_provider.model_revision,
                "max_sequence_length": embedding_provider.max_sequence_length,
                "dimension": embedding_provider.dimension,
            }
            for key, value in actual.items():
                if metadata.get(key) != value:
                    raise RetrievalError(f"embedding provider does not match index: {key}")
        return cls(evidence, bm25, dense, manifest, embedding_provider)


def build_retrieval_index(
    *,
    evidence_path: Path,
    output_dir: Path,
    embedding_provider: EmbeddingProvider,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
    candidate_depth: int = 50,
) -> RetrievalIndex:
    if output_dir.exists():
        raise RetrievalError("refusing to overwrite an existing retrieval index")
    if rrf_k <= 0 or candidate_depth <= 0:
        raise ValueError("RRF settings must be positive")
    if not embedding_provider.model_revision.strip():
        raise RetrievalError("dense index requires an explicit immutable model revision")
    ingestion_path, evidence_hash, _ = _validate_ingestion_source(evidence_path)
    evidence = load_evidence(evidence_path)
    contents = [item.content for item in evidence]
    validate_document_lengths(embedding_provider, contents)
    bm25 = BM25Index.build(contents, k1=bm25_k1, b=bm25_b)
    dense = normalize_rows(embedding_provider.encode_documents(contents))
    if dense.shape != (len(evidence), embedding_provider.dimension):
        raise RetrievalError("embedding provider returned an unexpected document matrix shape")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        ids_path = staging / "evidence_ids.json"
        bm25_path = staging / "bm25.json"
        dense_path = staging / "dense_embeddings.npy"
        ids_path.write_bytes(_json_bytes([item.evidence_id for item in evidence]))
        bm25_path.write_bytes(_json_bytes(bm25.to_dict()))
        np.save(dense_path, dense, allow_pickle=False)
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "complete": True,
            "source": {
                "evidence_jsonl": evidence_path.as_posix(),
                "evidence_jsonl_sha256": evidence_hash,
                "ingestion_manifest": ingestion_path.as_posix(),
                "ingestion_manifest_sha256": file_sha256(ingestion_path),
                "evidence_count": len(evidence),
            },
            "bm25": {
                "implementation": BM25_IMPLEMENTATION,
                "tokenizer": TOKENIZER_VERSION,
                "scored_field": "content",
                "k1": bm25_k1,
                "b": bm25_b,
                "stemming": False,
                "stop_words": False,
            },
            "embedding": {
                "provider": embedding_provider.provider_name,
                "model_id": embedding_provider.model_id,
                "model_revision": embedding_provider.model_revision,
                "max_sequence_length": embedding_provider.max_sequence_length,
                "dimension": embedding_provider.dimension,
                "dtype": "float32",
                "normalized": True,
                "document_method": "encode_document",
                "query_method": "encode_query",
                "scored_field": "content",
                "trust_remote_code": False,
            },
            "fusion": {
                "method": "rrf",
                "rrf_k": rrf_k,
                "candidate_depth": candidate_depth,
                "raw_score_combination": False,
            },
            "artifacts": {
                "evidence_ids": ids_path.name,
                "evidence_ids_sha256": file_sha256(ids_path),
                "bm25": bm25_path.name,
                "bm25_sha256": file_sha256(bm25_path),
                "dense_embeddings": dense_path.name,
                "dense_embeddings_sha256": file_sha256(dense_path),
            },
        }
        (staging / "index_manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return RetrievalIndex.load(
        evidence_path=evidence_path,
        index_dir=output_dir,
        embedding_provider=embedding_provider,
    )
