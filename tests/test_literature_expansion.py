from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from l3s_agent.config import LiteratureConfig, load_config
from l3s_agent.literature.builder import CorpusBuilder
from l3s_agent.literature.discovery import domain_signals, score_candidate
from l3s_agent.literature.models import (
    DownloadRecord,
    DownloadStatus,
    LiteratureCandidate,
    OpenAlexRequestRecord,
    QueryMatch,
)


class ScriptedOpenAlex:
    def __init__(
        self,
        responses: Mapping[tuple[str, int], Sequence[LiteratureCandidate]],
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int, int]] = []
        self.request_records: list[OpenAlexRequestRecord] = []

    def search(
        self, *, query: str, page: int, per_page: int
    ) -> list[LiteratureCandidate]:
        self.calls.append((query, page, per_page))
        batch = list(self.responses.get((query, page), ()))
        self.request_records.append(
            OpenAlexRequestRecord(
                query=query,
                page=page,
                per_page=per_page,
                result_count=len(batch),
                response_sha256=f"{len(self.calls):064x}",
            )
        )
        return batch


class SelectiveDownloader:
    def __init__(self, successful_ids: set[str]) -> None:
        self.successful_ids = successful_ids
        self.calls: list[str] = []

    def download(
        self,
        *,
        paper_id: str,
        pdf_urls: tuple[str, ...],
        destination_dir: Path,
        openalex_content_url: str | None = None,
    ) -> DownloadRecord:
        openalex_id = paper_id.removeprefix("paper_")
        self.calls.append(openalex_id)
        sources = ((openalex_content_url,) if openalex_content_url else ()) + pdf_urls
        if openalex_id not in self.successful_ids:
            return DownloadRecord(
                status=DownloadStatus.FAILED,
                attempted_urls=sources,
                failure_reason="controlled acquisition failure",
            )
        return DownloadRecord(
            status=DownloadStatus.SUCCESS,
            attempted_urls=sources,
            successful_url=sources[0],
            local_path=destination_dir / f"{paper_id}.pdf",
            sha256="a" * 64,
            content_length=2048,
        )


def expansion_config(tmp_path: Path, *, max_pages: int = 4) -> LiteratureConfig:
    return replace(
        load_config(environ={}).literature,
        candidate_min=30,
        candidate_max=50,
        expansion_increment=10,
        expansion_candidate_max=90,
        expansion_max_pages=max_pages,
        selection_min=8,
        selection_target=10,
        selection_max=12,
        pdf_dir=tmp_path / "pdfs",
    )


def make_candidates(
    candidate_factory: Callable[..., LiteratureCandidate],
    *,
    prefix: str,
    count: int,
    query: str,
    rank_start: int = 1,
) -> list[LiteratureCandidate]:
    return [
        candidate_factory(
            openalex_id=f"{prefix}{index}",
            openalex_url=f"https://openalex.org/{prefix}{index}",
            doi=f"10.5555/{prefix.casefold()}{index}",
            title=f"Weather Effects on Solar Power Generation Study {prefix}{index}",
            source_url=f"https://publisher.example/{prefix}{index}",
            query_matches=(QueryMatch(query, rank_start + index - 1),),
        )
        for index in range(1, count + 1)
    ]


def focused_responses(
    candidate_factory: Callable[..., LiteratureCandidate],
    config: LiteratureConfig,
    *,
    count: int = 40,
) -> dict[tuple[str, int], list[LiteratureCandidate]]:
    responses: dict[tuple[str, int], list[LiteratureCandidate]] = {}
    per_query: list[list[LiteratureCandidate]] = [[], [], []]
    for offset in range(count):
        query_index = offset % len(config.expansion_queries)
        query = config.expansion_queries[query_index]
        identifier = offset + 1
        per_query[query_index].append(
            candidate_factory(
                openalex_id=f"E{identifier}",
                openalex_url=f"https://openalex.org/E{identifier}",
                doi=f"10.7777/e{identifier}",
                title=f"Meteorological Effects on Wind Power Generation Study E{identifier}",
                abstract=(
                    "Wind power generation changes with meteorological conditions "
                    "and wind speed."
                ),
                topics=("Wind Power", "Meteorological Conditions"),
                keywords=("wind speed", "power generation"),
                query_matches=(QueryMatch(query, len(per_query[query_index]) + 1),),
            )
        )
    for query, batch in zip(config.expansion_queries, per_query):
        responses[(query, 1)] = batch
    return responses


@pytest.mark.parametrize("validated", [8, 9])
def test_initial_eight_or_nine_is_complete_without_expansion(
    tmp_path: Path,
    candidate_factory: Callable[..., LiteratureCandidate],
    validated: int,
) -> None:
    config = expansion_config(tmp_path)
    initial = make_candidates(
        candidate_factory, prefix="I", count=50, query=config.queries[0]
    )
    openalex = ScriptedOpenAlex({})
    downloader = SelectiveDownloader({f"I{index}" for index in range(1, validated + 1)})

    manifest = CorpusBuilder(
        config=config, openalex=openalex, downloader=downloader  # type: ignore[arg-type]
    ).build_manifest(initial, corpus_id=f"initial-{validated}")

    assert manifest.summary["selected"] == validated
    assert manifest.complete is True
    assert openalex.calls == []
    assert manifest.discovery_expansion["triggered"] is False
    assert manifest.discovery_expansion["stop_reason"] == "initial_minimum_satisfied"


def test_expansion_uses_fixed_query_order_and_stops_at_ten_validated_pdfs(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    config = expansion_config(tmp_path)
    initial = make_candidates(
        candidate_factory, prefix="I", count=50, query=config.queries[0]
    )
    responses = focused_responses(candidate_factory, config, count=12)
    openalex = ScriptedOpenAlex(responses)
    downloader = SelectiveDownloader(
        {f"I{index}" for index in range(1, 8)} | {f"E{index}" for index in range(1, 13)}
    )

    manifest = CorpusBuilder(
        config=config, openalex=openalex, downloader=downloader  # type: ignore[arg-type]
    ).build_manifest(initial, corpus_id="expanded-target")

    expected_calls = [(query, 1, config.per_query) for query in config.expansion_queries]
    assert openalex.calls == expected_calls
    assert manifest.summary["selected"] == 10
    assert manifest.summary["unique_candidates"] == 60
    assert manifest.complete is True
    assert manifest.discovery_expansion["triggered"] is True
    assert manifest.discovery_expansion["stop_reason"] == "target_reached"
    rounds = manifest.discovery_expansion["rounds"]
    assert isinstance(rounds, list)
    assert rounds[1]["new_unique_candidates"] == 10
    assert rounds[1]["cumulative_validated_pdfs"] == 10


def test_expansion_stops_at_ninety_unique_candidates_when_downloads_keep_failing(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    config = expansion_config(tmp_path, max_pages=1)
    initial = make_candidates(
        candidate_factory, prefix="I", count=50, query=config.queries[0]
    )
    openalex = ScriptedOpenAlex(focused_responses(candidate_factory, config, count=40))
    downloader = SelectiveDownloader({f"I{index}" for index in range(1, 8)})

    manifest = CorpusBuilder(
        config=config, openalex=openalex, downloader=downloader  # type: ignore[arg-type]
    ).build_manifest(initial, corpus_id="budget-exhausted")

    assert manifest.summary["unique_candidates"] == 90
    assert manifest.summary["selected"] == 7
    assert manifest.complete is False
    assert manifest.discovery_expansion["stop_reason"] == "maximum_unique_budget_reached"
    rounds = manifest.discovery_expansion["rounds"]
    assert isinstance(rounds, list)
    assert [round_record["new_unique_candidates"] for round_record in rounds[1:]] == [
        10,
        10,
        10,
        10,
    ]
    assert len(set(downloader.calls)) == 90


def test_focused_search_exhaustion_is_bounded_and_recorded(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    config = expansion_config(tmp_path, max_pages=2)
    initial = make_candidates(
        candidate_factory, prefix="I", count=50, query=config.queries[0]
    )
    openalex = ScriptedOpenAlex({})
    downloader = SelectiveDownloader({f"I{index}" for index in range(1, 8)})

    manifest = CorpusBuilder(
        config=config, openalex=openalex, downloader=downloader  # type: ignore[arg-type]
    ).build_manifest(initial, corpus_id="search-exhausted")

    assert len(openalex.calls) == 2 * len(config.expansion_queries)
    assert manifest.summary["unique_candidates"] == 50
    assert manifest.discovery_expansion["stop_reason"] == "focused_search_exhausted"
    rounds = manifest.discovery_expansion["rounds"]
    assert isinstance(rounds, list)
    assert [page["query"] for page in rounds[1]["query_pages"]] == list(
        config.expansion_queries
    ) * 2


def test_later_duplicate_hit_retains_provenance_without_changing_frozen_score(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    config = expansion_config(tmp_path, max_pages=1)
    initial = make_candidates(
        candidate_factory, prefix="I", count=50, query=config.queries[0]
    )
    initial[0] = replace(
        initial[0], query_matches=(QueryMatch(config.queries[0], 15),)
    )
    duplicate = replace(
        initial[0], query_matches=(QueryMatch(config.expansion_queries[0], 1),)
    )
    responses = focused_responses(candidate_factory, config, count=10)
    responses[(config.expansion_queries[0], 1)].insert(0, duplicate)
    openalex = ScriptedOpenAlex(responses)
    downloader = SelectiveDownloader(
        {f"I{index}" for index in range(1, 8)} | {f"E{index}" for index in range(1, 11)}
    )
    expected_frozen = score_candidate(initial[0], domain_signals(initial[0]), config)

    manifest = CorpusBuilder(
        config=config, openalex=openalex, downloader=downloader  # type: ignore[arg-type]
    ).build_manifest(initial, corpus_id="frozen-score")

    existing = next(
        paper for paper in manifest.papers if paper.candidate.openalex_id == "I1"
    )
    assert existing.score == expected_frozen
    assert {match.query for match in existing.candidate.query_matches} == {
        config.queries[0],
        config.expansion_queries[0],
    }
    provenance = manifest.discovery_expansion["candidate_provenance"]
    assert isinstance(provenance, dict)
    existing_provenance = provenance["I1"]
    assert existing_provenance["first_admission_query_matches"] == [
        {"query": config.queries[0], "rank": 15, "relevance_score": None}
    ]
    assert {item["query"] for item in existing_provenance["retrievals"]} == {
        config.queries[0],
        config.expansion_queries[0],
    }
    rounds = manifest.discovery_expansion["rounds"]
    assert isinstance(rounds, list)
    assert rounds[1]["duplicate_hits"] == 1
    assert rounds[1]["new_unique_candidates"] == 10
    assert downloader.calls.count("I1") == 1


def test_new_expansion_candidate_uses_the_unchanged_scoring_formula(
    tmp_path: Path, candidate_factory: Callable[..., LiteratureCandidate]
) -> None:
    config = expansion_config(tmp_path, max_pages=1)
    initial = make_candidates(
        candidate_factory, prefix="I", count=50, query=config.queries[0]
    )
    responses = focused_responses(candidate_factory, config, count=10)
    expansion_candidate = responses[(config.expansion_queries[0], 1)][0]
    openalex = ScriptedOpenAlex(responses)
    downloader = SelectiveDownloader(
        {f"I{index}" for index in range(1, 8)} | {f"E{index}" for index in range(1, 11)}
    )
    expected = score_candidate(
        expansion_candidate, domain_signals(expansion_candidate), config
    )

    manifest = CorpusBuilder(
        config=config, openalex=openalex, downloader=downloader  # type: ignore[arg-type]
    ).build_manifest(initial, corpus_id="new-score")

    decision = next(
        paper
        for paper in manifest.papers
        if paper.candidate.openalex_id == expansion_candidate.openalex_id
    )
    assert decision.score == expected
    provenance = manifest.discovery_expansion["candidate_provenance"]
    assert isinstance(provenance, dict)
    assert provenance[expansion_candidate.openalex_id]["first_admission_round"] == 1
    assert manifest.rules["ranking"] == {
        "query_relevance": 35,
        "domain_relevance": 35,
        "accessibility": 20,
        "metadata_completeness": 5,
        "recency": 5,
    }
