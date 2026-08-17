from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from l3s_agent.config import load_config
from l3s_agent.literature.builder import CorpusBuilder
from l3s_agent.literature.manifest import manifest_to_dict, write_manifest
from l3s_agent.literature.models import (
    DownloadRecord,
    DownloadStatus,
    LiteratureCandidate,
    OpenAccessInfo,
    OpenAlexRequestRecord,
)


class FakeOpenAlex:
    def __init__(self, *, secret: str | None = None) -> None:
        self._authorization = f"Bearer {secret}" if secret else None
        self.request_records: list[OpenAlexRequestRecord] = []

    def search(
        self, *, query: str, page: int, per_page: int
    ) -> list[LiteratureCandidate]:
        return []


class PagingOpenAlex:
    def __init__(self, candidate_factory: Callable[..., LiteratureCandidate]) -> None:
        self.candidate_factory = candidate_factory
        self.request_records: list[object] = []
        self.calls: list[tuple[str, int]] = []

    def search(self, *, query: str, page: int, per_page: int) -> list[LiteratureCandidate]:
        self.calls.append((query, page))
        identifier = len(self.calls)
        return [
            self.candidate_factory(
                openalex_id=f"W{identifier}",
                openalex_url=f"https://openalex.org/W{identifier}",
                doi=f"10.4/{identifier}",
                title=f"Weather and Solar Power Generation Result {identifier}",
                normalized_title=f"weather and solar power generation result {identifier}",
            )
        ]


class FakeDownloader:
    def __init__(self, *, fail_ids: set[str] | None = None) -> None:
        self.fail_ids = fail_ids or set()

    def download(
        self,
        *,
        paper_id: str,
        pdf_urls: tuple[str, ...],
        destination_dir: Path,
        openalex_content_url: str | None = None,
    ) -> DownloadRecord:
        sources = ((openalex_content_url,) if openalex_content_url else ()) + pdf_urls
        if paper_id in self.fail_ids:
            return DownloadRecord(
                status=DownloadStatus.FAILED,
                attempted_urls=sources,
                failure_reason="controlled failure",
            )
        return DownloadRecord(
            status=DownloadStatus.SUCCESS,
            attempted_urls=sources,
            successful_url=sources[0],
            local_path=destination_dir / f"{paper_id}.pdf",
            sha256="a" * 64,
            content_length=2048,
        )


def make_builder(tmp_path: Path, *, fail_ids: set[str] | None = None) -> CorpusBuilder:
    base = load_config(environ={}).literature
    config = replace(
        base,
        candidate_min=8,
        candidate_max=20,
        selection_min=8,
        selection_target=10,
        selection_max=12,
        pdf_dir=tmp_path / "pdfs",
    )
    return CorpusBuilder(
        config=config,
        openalex=FakeOpenAlex(),  # type: ignore[arg-type]
        downloader=FakeDownloader(fail_ids=fail_ids),  # type: ignore[arg-type]
        now=lambda: datetime(2026, 8, 17, tzinfo=timezone.utc),
    )


def test_builder_backfills_failed_download_and_selects_ten(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    candidates = [
        candidate_factory(
            openalex_id=f"W{index}",
            openalex_url=f"https://openalex.org/W{index}",
            doi=f"10.1/{index}",
            title=f"Weather and Solar Power Generation Study {index}",
            normalized_title=f"weather and solar power generation study {index}",
        )
        for index in range(1, 13)
    ]
    builder = make_builder(tmp_path, fail_ids={"paper_W1"})

    manifest = builder.build_manifest(candidates, corpus_id="test-corpus")

    assert manifest.summary["selected"] == 10
    assert manifest.summary["download_failures"] == 1
    assert manifest.complete is True
    selected = [paper for paper in manifest.papers if paper.status.value == "selected"]
    assert all(paper.download.sha256 == "a" * 64 for paper in selected)


def test_below_eight_downloads_marks_corpus_incomplete(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    candidates = [
        candidate_factory(
            openalex_id=f"W{index}",
            openalex_url=f"https://openalex.org/W{index}",
            doi=f"10.2/{index}",
            title=f"Weather and Wind Power Generation Study {index}",
            normalized_title=f"weather and wind power generation study {index}",
        )
        for index in range(1, 9)
    ]
    builder = make_builder(tmp_path, fail_ids={"paper_W1"})
    manifest = builder.build_manifest(candidates, corpus_id="incomplete")
    assert manifest.summary["selected"] == 7
    assert manifest.complete is False


def test_manifest_is_stable_and_never_overwritten(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    candidates = [
        candidate_factory(
            openalex_id=f"W{index}",
            openalex_url=f"https://openalex.org/W{index}",
            doi=f"10.3/{index}",
            title=f"Weather and Solar Power Generation Study {index}",
            normalized_title=f"weather and solar power generation study {index}",
        )
        for index in range(1, 9)
    ]
    manifest = make_builder(tmp_path).build_manifest(candidates, corpus_id="stable")
    output = tmp_path / "manifest.json"

    write_manifest(manifest, output)
    first = output.read_bytes()
    serialized = manifest_to_dict(manifest)
    assert serialized["corpus_kind"] == "frozen_base"
    assert serialized["papers"][0]["candidate"]["open_access"]["pdf_url"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_manifest(manifest, output)
    assert output.read_bytes() == first


def test_candidate_collection_uses_one_fallback_page_and_deterministic_cap(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    base = load_config(environ={}).literature
    config = replace(
        base,
        queries=("solar weather", "wind climate"),
        candidate_min=3,
        candidate_max=4,
        selection_min=2,
        selection_target=2,
        selection_max=3,
        per_query=1,
        pdf_dir=tmp_path / "pdfs",
    )
    openalex = PagingOpenAlex(candidate_factory)
    builder = CorpusBuilder(
        config=config,
        openalex=openalex,  # type: ignore[arg-type]
        downloader=FakeDownloader(),  # type: ignore[arg-type]
    )

    candidates = builder.collect_candidates()

    assert len(candidates) == 4
    assert openalex.calls == [
        ("solar weather", 1),
        ("wind climate", 1),
        ("solar weather", 2),
        ("wind climate", 2),
    ]


def test_manifest_never_serializes_client_credentials(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    secret = "manifest-secret-key"
    base = load_config(environ={}).literature
    config = replace(
        base,
        candidate_min=1,
        candidate_max=10,
        selection_min=1,
        selection_target=1,
        selection_max=2,
        pdf_dir=tmp_path / "pdfs",
    )
    openalex = FakeOpenAlex(secret=secret)
    openalex.request_records.append(
        OpenAlexRequestRecord(
            query="weather solar",
            page=1,
            per_page=10,
            result_count=1,
            response_sha256="b" * 64,
        )
    )
    builder = CorpusBuilder(
        config=config,
        openalex=openalex,  # type: ignore[arg-type]
        downloader=FakeDownloader(),  # type: ignore[arg-type]
    )
    manifest = builder.build_manifest(
        [candidate_factory()], corpus_id="credential-safe"
    )

    serialized = json.dumps(manifest_to_dict(manifest), sort_keys=True)
    assert secret not in serialized
    assert "Authorization" not in serialized


def test_oa_openalex_content_only_candidate_is_downloadable_without_rescoring(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    base = candidate_factory()
    candidate = replace(
        base,
        open_access=OpenAccessInfo(
            is_oa=True,
            status="green",
            landing_page_url="https://repository.example/work",
            pdf_urls=(),
            has_content_pdf=True,
            content_pdf_url="https://content.openalex.org/works/W1.pdf",
        ),
    )
    config = replace(
        load_config(environ={}).literature,
        candidate_min=1,
        candidate_max=2,
        selection_min=1,
        selection_target=1,
        selection_max=2,
        pdf_dir=tmp_path / "pdfs",
    )
    builder = CorpusBuilder(
        config=config,
        openalex=FakeOpenAlex(),  # type: ignore[arg-type]
        downloader=FakeDownloader(),  # type: ignore[arg-type]
    )

    manifest = builder.build_manifest([candidate], corpus_id="content-only")

    assert manifest.summary["selected"] == 1
    assert manifest.papers[0].download.successful_url == (
        "https://content.openalex.org/works/W1.pdf"
    )
    assert manifest.papers[0].score is not None
    assert manifest.papers[0].score.accessibility == 10


def test_openalex_content_does_not_weaken_oa_requirement(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    candidate = candidate_factory(
        open_access=OpenAccessInfo(
            is_oa=False,
            status="closed",
            landing_page_url=None,
            pdf_urls=(),
            has_content_pdf=True,
            content_pdf_url="https://content.openalex.org/works/W1.pdf",
        )
    )
    builder = make_builder(tmp_path)

    manifest = builder.build_manifest([candidate], corpus_id="closed-content")

    assert manifest.summary["selected"] == 0
    assert manifest.papers[0].download.status is DownloadStatus.FAILED
    assert "not OA" in manifest.papers[0].reason
