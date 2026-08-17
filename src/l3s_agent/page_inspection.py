"""Canonical Phase 3 page resolution and bounded multimodal Evidence creation."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .ingestion.models import PageRecord
from .interfaces import LLMProvider
from .models import CorpusScope, Evidence, PageInspectionResult


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_inspection_question(question: str) -> str:
    """Apply the frozen conservative identity normalization."""

    return " ".join(question.strip().split())


class CanonicalPageResolver:
    """Resolve only checksummed page images recorded by a completed Phase 3 artifact."""

    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root.resolve()
        manifest_path = self.artifact_root / "ingestion_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid ingestion manifest") from exc
        if manifest.get("summary", {}).get("complete") is not True:
            raise ValueError("page resolution requires a completed ingestion artifact")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("ingestion manifest is missing artifact metadata")
        pages_name = artifacts.get("pages_jsonl")
        expected_pages_hash = artifacts.get("pages_jsonl_sha256")
        image_root_name = artifacts.get("page_images_root")
        if not all(isinstance(item, str) and item for item in (
            pages_name,
            expected_pages_hash,
            image_root_name,
        )):
            raise ValueError("ingestion manifest has invalid page artifact metadata")
        pages_path = self._contained_path(str(pages_name), self.artifact_root)
        if _file_sha256(pages_path) != expected_pages_hash:
            raise ValueError("pages.jsonl checksum mismatch")
        self.image_root = self._contained_path(str(image_root_name), self.artifact_root)

        records: dict[tuple[str, int], PageRecord] = {}
        try:
            lines = pages_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                raw = json.loads(line)
                raw["section_headings"] = tuple(raw.get("section_headings", ()))
                raw["warnings"] = tuple(raw.get("warnings", ()))
                record = PageRecord(**raw)
                image_path = self._contained_path(record.image_path, self.artifact_root)
                if not image_path.is_relative_to(self.image_root):
                    raise ValueError("canonical page path is outside the image root")
                key = (record.paper_id, record.page)
                if key in records:
                    raise ValueError("duplicate canonical page record")
                records[key] = record
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("invalid pages.jsonl") from exc
        if not records:
            raise ValueError("canonical page map cannot be empty")
        expected_pages = manifest.get("summary", {}).get("pages")
        if expected_pages != len(records):
            raise ValueError("canonical page count does not match ingestion manifest")
        self._records: Mapping[tuple[str, int], PageRecord] = MappingProxyType(records)

    @staticmethod
    def _contained_path(relative: str, root: Path) -> Path:
        value = Path(relative)
        if value.is_absolute():
            raise ValueError("canonical page paths must be relative")
        resolved = (root / value).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("canonical page path escapes its artifact root")
        return resolved

    @property
    def records(self) -> Mapping[tuple[str, int], PageRecord]:
        return self._records

    def resolve(self, *, paper_id: str, page: int) -> tuple[PageRecord, Path]:
        if not paper_id.strip():
            raise ValueError("paper_id is required")
        if page < 1:
            raise ValueError("page inspection uses 1-based physical PDF pages")
        record = self._records.get((paper_id, page))
        if record is None:
            raise ValueError("unknown canonical paper/page")
        image_path = self._contained_path(record.image_path, self.artifact_root)
        if not image_path.is_relative_to(self.image_root):
            raise ValueError("canonical page path is outside the image root")
        if image_path.suffix.lower() != ".png" or not image_path.is_file():
            raise ValueError("canonical page image is missing or not PNG")
        if image_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise ValueError("canonical page image has an invalid PNG signature")
        if _file_sha256(image_path) != record.image_sha256:
            raise ValueError("canonical page image checksum mismatch")
        return record, image_path


class CanonicalPageInspectionTool:
    """Research-Agent-facing inspection tool over canonical rendered pages only."""

    def __init__(self, *, resolver: CanonicalPageResolver, provider: LLMProvider) -> None:
        self.resolver = resolver
        self.provider = provider

    def inspect(self, *, paper_id: str, page: int, question: str, session_id: str) -> Evidence:
        normalized_question = normalize_inspection_question(question)
        if not normalized_question or len(normalized_question) > 500:
            raise ValueError("inspection question must contain at most 500 characters")
        if not session_id.strip():
            raise ValueError("page inspection requires a session_id")
        record, image_path = self.resolver.resolve(paper_id=paper_id, page=page)
        result = self.provider.inspect_page(
            image_path=image_path,
            paper_id=paper_id,
            page=page,
            question=normalized_question,
        )
        if not isinstance(result, PageInspectionResult):
            raise TypeError("page inspection provider returned malformed data")
        if (
            result.paper_id != paper_id
            or result.page != page
            or result.question != normalized_question
        ):
            raise ValueError("page inspection result does not match the submitted provenance")
        identity_material = "\0".join(
            (session_id, paper_id, str(page), normalized_question, record.image_sha256)
        )
        identity = sha256(identity_material.encode("utf-8")).hexdigest()[:20]
        content = json.dumps(
            {
                "answer": result.answer,
                "observation": result.observation,
                "relevant_visual_elements": list(result.relevant_visual_elements),
                "limitations": list(result.limitations),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return Evidence(
            evidence_id=f"session:page_inspection:{identity}",
            paper_id=paper_id,
            title=record.title,
            page=page,
            modality=result.modality,
            source_id=f"page_inspection:{record.source_id}:p{page:04d}",
            content=content,
            corpus_scope=CorpusScope.SESSION,
            section="page inspection",
            session_id=session_id,
        )
