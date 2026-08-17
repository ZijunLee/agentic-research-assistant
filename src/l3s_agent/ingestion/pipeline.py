"""Offline, all-or-nothing ingestion of the frozen ten-paper base corpus."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import pymupdf

from ..config import ChunkingConfig, IngestionConfig
from ..models import CorpusScope, Evidence, EvidenceModality
from .chunking import chunk_page, normalize_page_text
from .models import (
    IngestionArtifact,
    IngestionWarning,
    PageRecord,
    PaperIngestionRecord,
    TextChunkRecord,
)


class IngestionError(RuntimeError):
    """Structured fatal error; frozen papers are never silently skipped."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        paper_id: str | None = None,
        page: int | None = None,
    ) -> None:
        self.code = code
        self.paper_id = paper_id
        self.page = page
        context = ""
        if paper_id:
            context += f" paper={paper_id}"
        if page is not None:
            context += f" page={page}"
        super().__init__(f"{code}:{context} {message}".strip())

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "paper_id": self.paper_id,
            "page": self.page,
            "message": str(self),
        }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_line(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _paper_id(openalex_id: str) -> str:
    suffix = openalex_id.rstrip("/").rsplit("/", 1)[-1]
    if not suffix:
        raise IngestionError("invalid_source_id", "OpenAlex work ID is blank")
    return f"paper_{suffix}"


def _load_frozen_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestionError(
            "invalid_manifest", f"cannot read frozen manifest ({type(exc).__name__})"
        ) from None
    if manifest.get("corpus_kind") != "frozen_base":
        raise IngestionError("invalid_manifest", "corpus_kind must be frozen_base")
    summary = manifest.get("summary", {})
    if summary.get("complete") is not True:
        raise IngestionError("invalid_manifest", "frozen base corpus is not complete")
    papers = manifest.get("papers")
    if not isinstance(papers, list):
        raise IngestionError("invalid_manifest", "papers must be a list")
    selected = [paper for paper in papers if paper.get("status") == "selected"]
    if len(selected) != 10 or summary.get("selected") != 10:
        raise IngestionError(
            "invalid_manifest", "Phase 3 requires exactly ten selected frozen papers"
        )
    selected.sort(
        key=lambda paper: (
            int(paper.get("decision_rank", 10**9)),
            str(paper.get("candidate", {}).get("openalex_id", "")),
        )
    )
    return manifest, selected


def _git_metadata(project_root: Path) -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    )
    return {"git_revision": revision or None, "git_worktree_dirty": dirty}


def _ingestion_code_sha256() -> str:
    digest = sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _page_dict(record: PageRecord) -> dict[str, Any]:
    return asdict(record)


def _chunk_dict(record: TextChunkRecord) -> dict[str, Any]:
    evidence = record.evidence
    return {
        "evidence_id": evidence.evidence_id,
        "paper_id": evidence.paper_id,
        "title": evidence.title,
        "page": evidence.page,
        "section": evidence.section,
        "modality": evidence.modality.value,
        "source_id": evidence.source_id,
        "corpus_scope": evidence.corpus_scope.value,
        "content": evidence.content,
        "chunk_index": record.chunk_index,
        "approx_token_count": record.approx_token_count,
        "content_sha256": record.content_sha256,
    }


def _artifact_dict(artifact: IngestionArtifact) -> dict[str, Any]:
    return {
        "schema_version": artifact.schema_version,
        "corpus_id": artifact.corpus_id,
        "corpus_scope": artifact.corpus_scope,
        "source_manifest": dict(artifact.source_manifest),
        "generator": dict(artifact.generator),
        "configuration": dict(artifact.configuration),
        "artifacts": dict(artifact.artifacts),
        "papers": [asdict(paper) for paper in artifact.papers],
        "warnings": [asdict(warning) for warning in artifact.warnings],
        "summary": dict(artifact.summary),
    }


def _warning_for_text(
    text: str, *, paper_id: str, page: int
) -> IngestionWarning | None:
    if not text:
        return IngestionWarning(
            code="empty_page_text",
            paper_id=paper_id,
            page=page,
            message="page has no extractable text; image retained and no evidence emitted",
        )
    if len(text) < 40:
        return IngestionWarning(
            code="short_page_text",
            paper_id=paper_id,
            page=page,
            message="page contains fewer than 40 extracted characters",
        )
    if text.count("\ufffd") / len(text) > 0.01:
        return IngestionWarning(
            code="replacement_characters",
            paper_id=paper_id,
            page=page,
            message="more than one percent of extracted characters are replacements",
        )
    return None


def _validate_selected_record(
    paper: Mapping[str, Any], *, project_root: Path
) -> tuple[str, str, str, int, Path, str]:
    candidate = paper.get("candidate")
    download = paper.get("download")
    if not isinstance(candidate, Mapping) or not isinstance(download, Mapping):
        raise IngestionError("invalid_manifest", "selected paper record is incomplete")
    openalex_id = str(candidate.get("openalex_id", "")).strip()
    paper_id = _paper_id(openalex_id)
    title = str(candidate.get("title", "")).strip()
    if not title:
        raise IngestionError("invalid_manifest", "selected paper title is blank", paper_id=paper_id)
    if download.get("status") != "success":
        raise IngestionError(
            "invalid_download_record", "selected PDF status is not success", paper_id=paper_id
        )
    local_path = str(download.get("local_path", "")).strip()
    expected_sha256 = str(download.get("sha256", "")).strip().lower()
    if not local_path or len(expected_sha256) != 64:
        raise IngestionError(
            "invalid_download_record", "selected PDF path or SHA-256 is invalid", paper_id=paper_id
        )
    pdf_path = Path(local_path)
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path
    if not pdf_path.is_file():
        raise IngestionError("missing_pdf", "local frozen PDF does not exist", paper_id=paper_id)
    actual_sha256 = _sha256_file(pdf_path)
    if actual_sha256 != expected_sha256:
        raise IngestionError(
            "pdf_hash_mismatch",
            f"expected {expected_sha256}, found {actual_sha256}",
            paper_id=paper_id,
        )
    rank = int(paper.get("decision_rank", 0))
    if rank < 1:
        raise IngestionError("invalid_manifest", "selected rank must be positive", paper_id=paper_id)
    return paper_id, title, openalex_id, rank, pdf_path, expected_sha256


def ingest_base_corpus(
    *,
    manifest_path: Path,
    output_dir: Path,
    project_root: Path,
    chunking: ChunkingConfig,
    ingestion: IngestionConfig,
) -> IngestionArtifact:
    """Create deterministic page/evidence artifacts without network access."""

    project_root = project_root.resolve()
    manifest_path = manifest_path if manifest_path.is_absolute() else project_root / manifest_path
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    if output_dir.exists():
        raise IngestionError(
            "output_exists",
            "refusing to overwrite an existing ingestion artifact; choose a new output path",
        )
    if not manifest_path.is_file():
        raise IngestionError("missing_manifest", "frozen base corpus manifest does not exist")
    manifest, selected = _load_frozen_manifest(manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    page_records: list[PageRecord] = []
    chunk_records: list[TextChunkRecord] = []
    paper_records: list[PaperIngestionRecord] = []
    warnings: list[IngestionWarning] = []
    image_set_digest = sha256()
    try:
        pages_root = staging / "pages"
        pages_root.mkdir()
        for selected_paper in selected:
            paper_id, title, source_id, rank, pdf_path, pdf_sha256 = _validate_selected_record(
                selected_paper, project_root=project_root
            )
            try:
                document = pymupdf.open(pdf_path)
            except Exception as exc:
                raise IngestionError(
                    "unreadable_pdf", f"PyMuPDF open failed ({type(exc).__name__})", paper_id=paper_id
                ) from None
            try:
                if not document.is_pdf:
                    raise IngestionError("invalid_pdf", "input is not a PDF", paper_id=paper_id)
                if document.needs_pass:
                    raise IngestionError(
                        "encrypted_pdf", "PDF requires a password", paper_id=paper_id
                    )
                if document.page_count <= 0:
                    raise IngestionError("zero_page_pdf", "PDF has no pages", paper_id=paper_id)

                paper_chunk_start = len(chunk_records)
                paper_warning_start = len(warnings)
                current_section: str | None = None
                paper_image_dir = pages_root / paper_id
                paper_image_dir.mkdir()
                for page_index in range(document.page_count):
                    page_number = page_index + 1
                    try:
                        page = document.load_page(page_index)
                        text = normalize_page_text(page.get_text("text", sort=True))
                    except Exception as exc:
                        raise IngestionError(
                            "page_extraction_failed",
                            f"text extraction failed ({type(exc).__name__})",
                            paper_id=paper_id,
                            page=page_number,
                        ) from None
                    warning = _warning_for_text(text, paper_id=paper_id, page=page_number)
                    page_warning_codes: tuple[str, ...] = ()
                    if warning is not None:
                        warnings.append(warning)
                        page_warning_codes = (warning.code,)

                    image_relative = Path("pages") / paper_id / f"page_{page_number:04d}.png"
                    try:
                        pixmap = page.get_pixmap(
                            dpi=ingestion.page_image_dpi,
                            colorspace=pymupdf.csRGB,
                            alpha=False,
                            annots=True,
                        )
                        image_bytes = pixmap.tobytes(ingestion.page_image_format)
                    except Exception as exc:
                        raise IngestionError(
                            "page_render_failed",
                            f"page rendering failed ({type(exc).__name__})",
                            paper_id=paper_id,
                            page=page_number,
                        ) from None
                    image_path = staging / image_relative
                    image_path.write_bytes(image_bytes)
                    image_sha256 = sha256(image_bytes).hexdigest()
                    image_set_digest.update(image_relative.as_posix().encode("utf-8"))
                    image_set_digest.update(b"\0")
                    image_set_digest.update(image_sha256.encode("ascii"))
                    image_set_digest.update(b"\0")

                    drafts, current_section, headings = chunk_page(
                        text,
                        current_section=current_section,
                        target_min_tokens=chunking.target_min_tokens,
                        target_max_tokens=chunking.target_max_tokens,
                        overlap_tokens=chunking.overlap_tokens,
                    )
                    for chunk_index, draft in enumerate(drafts, start=1):
                        evidence_id = (
                            f"base:{paper_id}:p{page_number:04d}:c{chunk_index:03d}:"
                            f"{draft.content_sha256[:12]}"
                        )
                        chunk_records.append(
                            TextChunkRecord(
                                evidence=Evidence(
                                    evidence_id=evidence_id,
                                    paper_id=paper_id,
                                    title=title,
                                    page=page_number,
                                    modality=EvidenceModality.TEXT,
                                    source_id=source_id,
                                    content=draft.content,
                                    corpus_scope=CorpusScope.BASE,
                                    section=draft.section,
                                ),
                                chunk_index=chunk_index,
                                approx_token_count=draft.approx_token_count,
                                content_sha256=draft.content_sha256,
                            )
                        )
                    page_records.append(
                        PageRecord(
                            paper_id=paper_id,
                            title=title,
                            source_id=source_id,
                            page=page_number,
                            text=text,
                            text_sha256=sha256(text.encode("utf-8")).hexdigest(),
                            section_headings=headings,
                            image_path=image_relative.as_posix(),
                            image_sha256=image_sha256,
                            image_width=pixmap.width,
                            image_height=pixmap.height,
                            warnings=page_warning_codes,
                        )
                    )
                paper_records.append(
                    PaperIngestionRecord(
                        paper_id=paper_id,
                        title=title,
                        source_id=source_id,
                        decision_rank=rank,
                        pdf_path=_relative(pdf_path, project_root),
                        pdf_sha256=pdf_sha256,
                        page_count=document.page_count,
                        chunk_count=len(chunk_records) - paper_chunk_start,
                        warning_count=len(warnings) - paper_warning_start,
                    )
                )
            finally:
                document.close()

        pages_path = staging / "pages.jsonl"
        evidence_path = staging / "evidence.jsonl"
        pages_path.write_text(
            "".join(_json_line(_page_dict(record)) + "\n" for record in page_records),
            encoding="utf-8",
        )
        evidence_path.write_text(
            "".join(_json_line(_chunk_dict(record)) + "\n" for record in chunk_records),
            encoding="utf-8",
        )
        try:
            package_version = metadata.version("l3s-scientific-agent")
        except metadata.PackageNotFoundError:
            package_version = "unknown"
        artifact = IngestionArtifact(
            schema_version="1.0",
            corpus_id=str(manifest.get("corpus_id", "")),
            corpus_scope="base",
            source_manifest={
                "path": _relative(manifest_path, project_root),
                "sha256": manifest_sha256,
                "schema_version": manifest.get("schema_version"),
            },
            generator={
                "package_version": package_version,
                "ingestion_code_sha256": _ingestion_code_sha256(),
                "pymupdf_version": pymupdf.VersionBind,
                **_git_metadata(project_root),
            },
            configuration={
                "page_numbering": "physical_pdf_index_plus_one",
                "source_id": "openalex_work_id",
                "text_extraction": "pymupdf_text_sort_true",
                "text_normalization": "conservative_v1",
                "section_strategy": "deterministic_conservative_regex_v2",
                "section_propagation": "persist_until_recognized_heading",
                "section_headings_force_chunk_boundary": False,
                "chunk_boundary_strategy": "page_paragraph_sentence_word_v2",
                "short_final_chunk_merge": "merge_backward_if_total_at_most_max",
                "token_estimator": "ceil_character_count_divided_by_4",
                "target_min_tokens": chunking.target_min_tokens,
                "target_max_tokens": chunking.target_max_tokens,
                "overlap_tokens": chunking.overlap_tokens,
                "cross_page_boundaries": False,
                "page_image_format": ingestion.page_image_format,
                "page_image_dpi": ingestion.page_image_dpi,
                "page_image_colorspace": "RGB",
                "page_image_alpha": False,
                "page_annotations_visible": True,
                "ocr": False,
            },
            artifacts={
                "pages_jsonl": "pages.jsonl",
                "pages_jsonl_sha256": _sha256_file(pages_path),
                "evidence_jsonl": "evidence.jsonl",
                "evidence_jsonl_sha256": _sha256_file(evidence_path),
                "page_images_root": "pages",
                "page_image_set_sha256": image_set_digest.hexdigest(),
            },
            papers=tuple(paper_records),
            warnings=tuple(warnings),
            summary={
                "complete": True,
                "papers": len(paper_records),
                "pages": len(page_records),
                "text_chunks": len(chunk_records),
                "page_images": len(page_records),
                "warnings": len(warnings),
            },
        )
        manifest_output = staging / "ingestion_manifest.json"
        manifest_output.write_text(
            json.dumps(_artifact_dict(artifact), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        if len(paper_records) != 10 or len(page_records) == 0:
            raise IngestionError("artifact_validation_failed", "incomplete paper/page records")
        if len(page_records) != sum(paper.page_count for paper in paper_records):
            raise IngestionError("artifact_validation_failed", "page count mismatch")
        if len(page_records) != sum(1 for _ in pages_root.glob("*/*.png")):
            raise IngestionError("artifact_validation_failed", "page image count mismatch")
        os.replace(staging, output_dir)
        return artifact
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
