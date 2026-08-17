from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from l3s_agent.models import CorpusScope, EvidenceModality, PageInspectionResult
from l3s_agent.page_inspection import (
    CanonicalPageInspectionTool,
    CanonicalPageResolver,
    normalize_inspection_question,
)


PNG = b"\x89PNG\r\n\x1a\nsynthetic-offline-page"


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def artifact(
    tmp_path: Path,
    *,
    image_path: str = "pages/paper-1/page_0002.png",
    page: int = 2,
) -> Path:
    root = tmp_path / "base_index"
    root.mkdir(parents=True)
    image = root / image_path
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(PNG)
    page = {
        "paper_id": "paper-1",
        "title": "NWP and wind-power forecasting",
        "source_id": "W1",
        "page": page,
        "text": "page text",
        "text_sha256": sha256(b"page text").hexdigest(),
        "section_headings": [],
        "image_path": image_path,
        "image_sha256": file_hash(image),
        "image_width": 10,
        "image_height": 20,
        "warnings": [],
    }
    pages = root / "pages.jsonl"
    pages.write_text(json.dumps(page, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest = {
        "artifacts": {
            "pages_jsonl": "pages.jsonl",
            "pages_jsonl_sha256": file_hash(pages),
            "page_images_root": "pages",
        },
        "summary": {"complete": True, "pages": 1},
    }
    (root / "ingestion_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


def result(**overrides) -> PageInspectionResult:
    values = {
        "paper_id": "paper-1",
        "page": 2,
        "question": "What does the workflow show?",
        "modality": EvidenceModality.FIGURE,
        "observation": "A diagram connects corrected NWP wind speed to power output.",
        "relevant_visual_elements": ("Figure 1 workflow",),
        "answer": "Corrected meteorological inputs feed the power model.",
        "limitations": (),
    }
    values.update(overrides)
    return PageInspectionResult(**values)


class FakeProvider:
    def __init__(self, value: PageInspectionResult) -> None:
        self.value = value
        self.calls: list[dict] = []

    def inspect_page(self, **kwargs):
        self.calls.append(kwargs)
        return self.value


def test_resolver_loads_known_page_and_checks_integrity(tmp_path: Path) -> None:
    root = artifact(tmp_path)
    resolver = CanonicalPageResolver(root)
    record, image = resolver.resolve(paper_id="paper-1", page=2)
    assert record.title == "NWP and wind-power forecasting"
    assert image == (root / "pages/paper-1/page_0002.png").resolve()
    assert len(resolver.records) == 1


@pytest.mark.parametrize(
    "paper_id,page,match",
    [
        ("missing", 2, "unknown"),
        ("paper-1", 0, "1-based"),
        ("paper-1", 3, "unknown"),
    ],
)
def test_resolver_rejects_unknown_or_invalid_pages(
    tmp_path: Path, paper_id: str, page: int, match: str
) -> None:
    resolver = CanonicalPageResolver(artifact(tmp_path))
    with pytest.raises(ValueError, match=match):
        resolver.resolve(paper_id=paper_id, page=page)


def test_resolver_rejects_path_escape_missing_and_altered_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside|escapes"):
        CanonicalPageResolver(artifact(tmp_path / "escape", image_path="../outside.png"))

    root = artifact(tmp_path / "missing")
    resolver = CanonicalPageResolver(root)
    (root / "pages/paper-1/page_0002.png").unlink()
    with pytest.raises(ValueError, match="missing"):
        resolver.resolve(paper_id="paper-1", page=2)

    root = artifact(tmp_path / "altered")
    resolver = CanonicalPageResolver(root)
    (root / "pages/paper-1/page_0002.png").write_bytes(PNG + b"altered")
    with pytest.raises(ValueError, match="checksum"):
        resolver.resolve(paper_id="paper-1", page=2)


def test_resolver_rejects_changed_pages_manifest(tmp_path: Path) -> None:
    root = artifact(tmp_path)
    with (root / "pages.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="pages.jsonl checksum"):
        CanonicalPageResolver(root)


def test_tool_passes_exactly_one_canonical_image_and_creates_session_evidence(
    tmp_path: Path,
) -> None:
    root = artifact(tmp_path)
    provider = FakeProvider(result())
    tool = CanonicalPageInspectionTool(
        resolver=CanonicalPageResolver(root), provider=provider
    )
    evidence = tool.inspect(
        paper_id="paper-1",
        page=2,
        question="  What   does the workflow show?  ",
        session_id="session-1",
    )
    assert len(provider.calls) == 1
    assert provider.calls[0] == {
        "image_path": (root / "pages/paper-1/page_0002.png").resolve(),
        "paper_id": "paper-1",
        "page": 2,
        "question": "What does the workflow show?",
    }
    assert evidence.corpus_scope is CorpusScope.SESSION
    assert evidence.session_id == "session-1"
    assert evidence.modality is EvidenceModality.FIGURE
    assert evidence.source_id == "page_inspection:W1:p0002"
    assert evidence.section == "page inspection"
    assert "image_path" not in evidence.content
    assert "answer" in json.loads(evidence.content)


def test_evidence_identity_is_deterministic_and_uses_frozen_inputs(tmp_path: Path) -> None:
    root = artifact(tmp_path)

    def inspect(question="Question?", session="session-1"):
        normalized = normalize_inspection_question(question)
        tool = CanonicalPageInspectionTool(
            resolver=CanonicalPageResolver(root),
            provider=FakeProvider(result(question=normalized)),
        )
        return tool.inspect(
            paper_id="paper-1", page=2, question=question, session_id=session
        )

    first = inspect("  Question? ")
    assert first.evidence_id == inspect("Question?").evidence_id
    assert first.evidence_id != inspect("Different?", session="session-1").evidence_id
    assert first.evidence_id != inspect("Question?", session="session-2").evidence_id

    page_three_root = artifact(
        tmp_path / "page-three",
        image_path="pages/paper-1/page_0003.png",
        page=3,
    )
    page_three_tool = CanonicalPageInspectionTool(
        resolver=CanonicalPageResolver(page_three_root),
        provider=FakeProvider(result(page=3, question="Question?")),
    )
    page_three = page_three_tool.inspect(
        paper_id="paper-1", page=3, question="Question?", session_id="session-1"
    )
    assert first.evidence_id != page_three.evidence_id

    image = root / "pages/paper-1/page_0002.png"
    image.write_bytes(PNG + b"replacement")
    raw = json.loads((root / "pages.jsonl").read_text())
    raw["image_sha256"] = file_hash(image)
    pages_text = json.dumps(raw, separators=(",", ":")) + "\n"
    (root / "pages.jsonl").write_text(pages_text, encoding="utf-8")
    manifest = json.loads((root / "ingestion_manifest.json").read_text())
    manifest["artifacts"]["pages_jsonl_sha256"] = file_hash(root / "pages.jsonl")
    (root / "ingestion_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert first.evidence_id != inspect("Question?").evidence_id


def test_tool_rejects_provider_provenance_mismatch_and_question_bounds(tmp_path: Path) -> None:
    root = artifact(tmp_path)
    for bad in (
        result(paper_id="other"),
        result(page=3),
        result(question="Other question"),
    ):
        tool = CanonicalPageInspectionTool(
            resolver=CanonicalPageResolver(root), provider=FakeProvider(bad)
        )
        with pytest.raises(ValueError, match="provenance"):
            tool.inspect(
                paper_id="paper-1",
                page=2,
                question="What does the workflow show?",
                session_id="session-1",
            )
    tool = CanonicalPageInspectionTool(
        resolver=CanonicalPageResolver(root), provider=FakeProvider(result())
    )
    with pytest.raises(ValueError, match="500"):
        tool.inspect(
            paper_id="paper-1", page=2, question="x" * 501, session_id="session-1"
        )


def test_page_inspection_result_enforces_semantic_and_serialized_bounds() -> None:
    with pytest.raises(ValueError, match="eight"):
        result(relevant_visual_elements=tuple(str(i) for i in range(9)))
    with pytest.raises(ValueError, match="4,000"):
        result(observation="x" * 4_000)
    with pytest.raises(ValueError, match="required"):
        result(answer=" ")
    with pytest.raises(ValueError, match="figure or table"):
        result(modality=EvidenceModality.TEXT)
