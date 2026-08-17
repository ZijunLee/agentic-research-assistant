from hashlib import sha256
import json
from pathlib import Path
import re

import pymupdf
import pytest

from l3s_agent.config import load_config
from l3s_agent.ingestion.pipeline import IngestionError, ingest_base_corpus


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_pdf(
    path: Path,
    pages: list[str],
    *,
    user_password: str | None = None,
) -> None:
    document = pymupdf.open()
    for text in pages:
        page = document.new_page(width=420, height=595)
        if text:
            page.insert_textbox(
                pymupdf.Rect(36, 36, 384, 559), text, fontsize=10, fontname="helv"
            )
    save_options = {}
    if user_password:
        save_options = {
            "encryption": pymupdf.PDF_ENCRYPT_AES_256,
            "owner_pw": "owner-password",
            "user_pw": user_password,
            "permissions": 0,
        }
    document.save(path, **save_options)
    document.close()


def zero_page_pdf() -> bytes:
    objects = [
        b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n",
        b"2 0 obj\n<</Type /Pages /Count 0 /Kids []>>\nendobj\n",
    ]
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(content))
        content.extend(item)
    xref = len(content)
    content.extend(b"xref\n0 3\n0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(
        f"trailer\n<</Size 3 /Root 1 0 R>>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(content)


def make_frozen_fixture(tmp_path: Path) -> tuple[Path, list[Path]]:
    pdf_dir = tmp_path / "data" / "papers" / "base" / "fixture-corpus"
    manifest_dir = tmp_path / "data" / "manifests"
    pdf_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    paths: list[Path] = []
    papers: list[dict[str, object]] = []
    for index in range(1, 11):
        path = pdf_dir / f"paper_W{index}.pdf"
        pages = [
            "Introduction\nSolar generation responds to irradiance and cloud cover. "
            f"Synthetic paper {index}."
        ]
        if index == 1:
            pages.append("")
        write_pdf(path, pages)
        paths.append(path)
        papers.append(
            {
                "candidate": {
                    "openalex_id": f"W{index}",
                    "title": f"Synthetic Weather and Renewable Energy Paper {index}",
                },
                "status": "selected",
                "decision_rank": index,
                "download": {
                    "status": "success",
                    "local_path": path.relative_to(tmp_path).as_posix(),
                    "sha256": file_sha256(path),
                },
            }
        )
    papers.append(
        {
            "candidate": {"openalex_id": "W999", "title": "Rejected paper"},
            "status": "rejected",
            "decision_rank": 99,
            "download": {"status": "not_attempted"},
        }
    )
    manifest = {
        "schema_version": "1.1",
        "corpus_id": "synthetic-frozen-base",
        "corpus_kind": "frozen_base",
        "summary": {"complete": True, "selected": 10},
        "papers": papers,
    }
    manifest_path = manifest_dir / "base_corpus.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path, paths


def run_ingestion(tmp_path: Path, manifest_path: Path, output_name: str = "index"):
    config = load_config(environ={})
    return ingest_base_corpus(
        manifest_path=manifest_path.relative_to(tmp_path),
        output_dir=Path("data/cache") / output_name,
        project_root=tmp_path,
        chunking=config.chunking,
        ingestion=config.ingestion,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pipeline_builds_complete_deterministic_page_aware_artifact(tmp_path: Path) -> None:
    manifest_path, _ = make_frozen_fixture(tmp_path)
    manifest_before = manifest_path.read_bytes()

    artifact = run_ingestion(tmp_path, manifest_path, "index-one")
    run_ingestion(tmp_path, manifest_path, "index-two")

    first = tmp_path / "data/cache/index-one"
    second = tmp_path / "data/cache/index-two"
    assert artifact.summary == {
        "complete": True,
        "papers": 10,
        "pages": 11,
        "text_chunks": 10,
        "page_images": 11,
        "warnings": 1,
    }
    assert manifest_path.read_bytes() == manifest_before
    assert (first / "pages.jsonl").read_bytes() == (second / "pages.jsonl").read_bytes()
    assert (first / "evidence.jsonl").read_bytes() == (
        second / "evidence.jsonl"
    ).read_bytes()
    assert (first / "ingestion_manifest.json").read_bytes() == (
        second / "ingestion_manifest.json"
    ).read_bytes()

    pages = read_jsonl(first / "pages.jsonl")
    evidence = read_jsonl(first / "evidence.jsonl")
    assert {record["page"] for record in pages if record["paper_id"] == "paper_W1"} == {
        1,
        2,
    }
    assert all(record["page"] >= 1 for record in pages + evidence)
    assert not any(
        record["paper_id"] == "paper_W1" and record["page"] == 2
        for record in evidence
    )
    assert {record["source_id"] for record in evidence} == {
        f"W{index}" for index in range(1, 11)
    }
    assert all(record["corpus_scope"] == "base" for record in evidence)
    assert all(record["modality"] == "text" for record in evidence)
    assert all(
        re.fullmatch(r"base:paper_W\d+:p\d{4}:c\d{3}:[0-9a-f]{12}", record["evidence_id"])
        for record in evidence
    )
    blank_page = next(
        record
        for record in pages
        if record["paper_id"] == "paper_W1" and record["page"] == 2
    )
    assert blank_page["warnings"] == ["empty_page_text"]
    assert (first / str(blank_page["image_path"])).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    ingestion_manifest = json.loads((first / "ingestion_manifest.json").read_text())
    assert ingestion_manifest["configuration"]["page_numbering"] == (
        "physical_pdf_index_plus_one"
    )
    assert ingestion_manifest["configuration"]["source_id"] == "openalex_work_id"
    assert ingestion_manifest["configuration"]["page_image_dpi"] == 144
    assert ingestion_manifest["configuration"]["ocr"] is False
    assert ingestion_manifest["configuration"]["section_strategy"].endswith("v2")
    assert (
        ingestion_manifest["configuration"]["section_headings_force_chunk_boundary"]
        is False
    )
    assert ingestion_manifest["summary"]["pages"] == 11
    assert ingestion_manifest["warnings"][0]["code"] == "empty_page_text"
    assert ingestion_manifest["artifacts"]["pages_jsonl_sha256"] == file_sha256(
        first / "pages.jsonl"
    )
    assert ingestion_manifest["artifacts"]["evidence_jsonl_sha256"] == file_sha256(
        first / "evidence.jsonl"
    )
    assert not any(record["paper_id"] == "paper_W999" for record in pages)


def test_pipeline_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    manifest_path, _ = make_frozen_fixture(tmp_path)
    run_ingestion(tmp_path, manifest_path)
    with pytest.raises(IngestionError, match="output_exists"):
        run_ingestion(tmp_path, manifest_path)


@pytest.mark.parametrize(
    ("mode", "error_code"),
    [
        ("missing", "missing_pdf"),
        ("hash", "pdf_hash_mismatch"),
        ("invalid", "unreadable_pdf"),
        ("encrypted", "encrypted_pdf"),
        ("zero", "zero_page_pdf"),
    ],
)
def test_fatal_pdf_integrity_failures_leave_no_completed_artifact(
    tmp_path: Path, mode: str, error_code: str
) -> None:
    manifest_path, paths = make_frozen_fixture(tmp_path)
    target = paths[0]
    manifest = json.loads(manifest_path.read_text())
    first_download = manifest["papers"][0]["download"]
    if mode == "missing":
        target.unlink()
    elif mode == "hash":
        target.write_bytes(target.read_bytes() + b"changed")
    elif mode == "invalid":
        target.write_bytes(b"not a PDF")
        first_download["sha256"] = file_sha256(target)
    elif mode == "encrypted":
        target.unlink()
        write_pdf(target, ["Secret scientific page"], user_password="reader-password")
        first_download["sha256"] = file_sha256(target)
    elif mode == "zero":
        target.write_bytes(zero_page_pdf())
        first_download["sha256"] = file_sha256(target)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    with pytest.raises(IngestionError) as captured:
        run_ingestion(tmp_path, manifest_path)

    assert captured.value.code == error_code
    assert captured.value.paper_id == "paper_W1"
    assert not (tmp_path / "data/cache/index").exists()


def test_manifest_must_be_complete_frozen_ten_paper_corpus(tmp_path: Path) -> None:
    manifest_path, _ = make_frozen_fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"]["complete"] = False
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(IngestionError, match="invalid_manifest"):
        run_ingestion(tmp_path, manifest_path)
