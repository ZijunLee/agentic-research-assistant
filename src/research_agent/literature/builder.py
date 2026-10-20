"""Bounded, deterministic construction of the frozen base literature corpus."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from ..config import LiteratureConfig
from .discovery import (
    deduplicate_candidates,
    exclusion_reasons,
    normalize_queries,
    score_candidate,
)
from .download import PDFDownloader
from .models import (
    CandidateDecision,
    CorpusManifest,
    DecisionStatus,
    DownloadRecord,
    DownloadStatus,
    DuplicateRecord,
    LiteratureCandidate,
    QueryMatch,
    ScoreBreakdown,
)
from .openalex import WORK_FIELDS, OpenAlexClient


class CorpusBuilder:
    def __init__(
        self,
        *,
        config: LiteratureConfig,
        openalex: OpenAlexClient,
        downloader: PDFDownloader,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config
        self.openalex = openalex
        self.downloader = downloader
        self.now = now
        self.used_queries = normalize_queries(config.queries)

    def collect_candidates(
        self, queries: Sequence[str] | None = None
    ) -> list[LiteratureCandidate]:
        normalized_queries = normalize_queries(queries or self.config.queries)
        if not normalized_queries:
            raise ValueError("at least one literature query is required")
        self.used_queries = normalized_queries

        by_query: list[list[LiteratureCandidate]] = []
        for query in normalized_queries:
            by_query.append(
                self.openalex.search(query=query, page=1, per_page=self.config.per_query)
            )
        candidates = self._cap_candidate_pool(self._round_robin(by_query))
        unique, _ = deduplicate_candidates(candidates)

        # One documented fallback page per query. It is used both to reach the
        # minimum candidate pool and to approach the deterministic 50-paper cap.
        if len(unique) < self.config.candidate_max:
            fallback: list[list[LiteratureCandidate]] = []
            for query in normalized_queries:
                fallback.append(
                    self.openalex.search(query=query, page=2, per_page=self.config.per_query)
                )
            candidates = self._cap_candidate_pool(
                candidates + self._round_robin(fallback)
            )
            unique, _ = deduplicate_candidates(candidates)
        return candidates

    def build_manifest(
        self,
        candidates: Sequence[LiteratureCandidate],
        *,
        corpus_id: str,
        generator: Mapping[str, object] | None = None,
    ) -> CorpusManifest:
        admitted, duplicates = deduplicate_candidates(candidates)
        admitted = admitted[: self.config.candidate_max]
        frozen_scores: dict[str, ScoreBreakdown] = {}
        eligibility_reasons: dict[str, tuple[str, ...]] = {}
        provenance: dict[str, dict[str, object]] = {}
        for candidate in admitted:
            self._freeze_candidate(
                candidate,
                admission_round=0,
                frozen_scores=frozen_scores,
                eligibility_reasons=eligibility_reasons,
                provenance=provenance,
            )

        downloads: dict[str, DownloadRecord] = {}
        initial_processed, initial_validated = self._acquire_ranked(
            admitted,
            frozen_scores,
            eligibility_reasons,
            downloads,
            corpus_id=corpus_id,
        )
        raw_results = len(candidates)
        initial_eligible = sum(not eligibility_reasons[item.openalex_id] for item in admitted)
        rounds: list[dict[str, object]] = [
            {
                "round": 0,
                "kind": "initial",
                "query_pages": [],
                "raw_results": raw_results,
                "duplicate_hits": len(duplicates),
                "overflow_candidates": 0,
                "new_unique_candidates": len(admitted),
                "new_unique_candidate_ids": [
                    candidate.openalex_id for candidate in admitted
                ],
                "cumulative_unique_candidates": len(admitted),
                "scientifically_eligible_added": initial_eligible,
                "candidates_processed_for_acquisition": initial_processed,
                "validated_pdfs_added": initial_validated,
                "cumulative_validated_pdfs": self._validated_count(downloads),
            }
        ]

        triggered = self._validated_count(downloads) < self.config.selection_min
        stop_reason = "initial_minimum_satisfied"
        expansion_buffer: list[tuple[LiteratureCandidate, int, int]] = []
        next_page = 1
        search_exhausted = False
        round_index = 0
        expansion_duplicates: list[DuplicateRecord] = []

        while (
            triggered
            and self._validated_count(downloads) < self.config.selection_target
            and len(admitted) < self.config.expansion_candidate_max
        ):
            round_index += 1
            new_unique = 0
            new_unique_ids: list[str] = []
            duplicate_hits = 0
            eligible_added = 0
            round_raw = 0
            query_pages: list[dict[str, object]] = []

            while (
                new_unique < self.config.expansion_increment
                and len(admitted) < self.config.expansion_candidate_max
            ):
                if not expansion_buffer:
                    if next_page > self.config.expansion_max_pages:
                        search_exhausted = True
                        break
                    batches: list[list[LiteratureCandidate]] = []
                    for query in self.config.expansion_queries:
                        before = len(self.openalex.request_records)
                        batch = self.openalex.search(
                            query=query,
                            page=next_page,
                            per_page=self.config.per_query,
                        )
                        batches.append(batch)
                        round_raw += len(batch)
                        query_pages.append(
                            self._request_provenance(
                                query=query,
                                page=next_page,
                                result_count=len(batch),
                                record_index=before,
                            )
                        )
                    for candidate in self._round_robin(batches):
                        expansion_buffer.append((candidate, round_index, next_page))
                    next_page += 1
                    if not expansion_buffer:
                        continue

                candidate, retrieval_round, retrieval_page = expansion_buffer.pop(0)
                duplicate_index, duplicate = self._existing_duplicate(admitted, candidate)
                if duplicate_index is not None:
                    duplicate_hits += 1
                    existing = admitted[duplicate_index]
                    admitted[duplicate_index] = replace(
                        existing,
                        query_matches=self._merge_query_matches(
                            existing.query_matches, candidate.query_matches
                        ),
                    )
                    self._record_retrieval(
                        provenance[existing.openalex_id],
                        candidate,
                        retrieval_round=retrieval_round,
                        retrieval_page=retrieval_page,
                    )
                    if duplicate is not None:
                        expansion_duplicates.append(duplicate)
                    continue

                admitted.append(candidate)
                self._freeze_candidate(
                    candidate,
                    admission_round=round_index,
                    frozen_scores=frozen_scores,
                    eligibility_reasons=eligibility_reasons,
                    provenance=provenance,
                    retrieval_round=retrieval_round,
                    retrieval_page=retrieval_page,
                )
                new_unique += 1
                new_unique_ids.append(candidate.openalex_id)
                if not eligibility_reasons[candidate.openalex_id]:
                    eligible_added += 1

            raw_results += round_raw
            processed, validated_added = self._acquire_ranked(
                admitted,
                frozen_scores,
                eligibility_reasons,
                downloads,
                corpus_id=corpus_id,
            )
            rounds.append(
                {
                    "round": round_index,
                    "kind": "focused_expansion",
                    "query_pages": query_pages,
                    "raw_results": round_raw,
                    "duplicate_hits": duplicate_hits,
                    "overflow_candidates": len(expansion_buffer),
                    "new_unique_candidates": new_unique,
                    "new_unique_candidate_ids": new_unique_ids,
                    "cumulative_unique_candidates": len(admitted),
                    "scientifically_eligible_added": eligible_added,
                    "candidates_processed_for_acquisition": processed,
                    "validated_pdfs_added": validated_added,
                    "cumulative_validated_pdfs": self._validated_count(downloads),
                }
            )
            if self._validated_count(downloads) >= self.config.selection_target:
                stop_reason = "target_reached"
                break
            if len(admitted) >= self.config.expansion_candidate_max:
                stop_reason = "maximum_unique_budget_reached"
                break
            if search_exhausted and not expansion_buffer:
                stop_reason = "focused_search_exhausted"
                break

        if triggered and stop_reason == "initial_minimum_satisfied":
            if self._validated_count(downloads) >= self.config.selection_target:
                stop_reason = "target_reached"
            elif len(admitted) >= self.config.expansion_candidate_max:
                stop_reason = "maximum_unique_budget_reached"
            else:
                stop_reason = "focused_search_exhausted"

        duplicates.extend(expansion_duplicates)
        ordered_eligible = self._ordered_eligible(
            admitted, frozen_scores, eligibility_reasons
        )
        decisions: list[CandidateDecision] = []
        for rank, candidate in enumerate(ordered_eligible, start=1):
            download = downloads.get(
                candidate.openalex_id,
                DownloadRecord(status=DownloadStatus.NOT_ATTEMPTED),
            )
            if download.status is DownloadStatus.SUCCESS:
                status = DecisionStatus.SELECTED
                reason = "selected by deterministic rank with validated OA PDF"
            elif download.status is DownloadStatus.FAILED:
                status = DecisionStatus.REJECTED
                reason = (
                    "OA PDF download or validation failed"
                    if download.attempted_urls
                    else "not OA with a downloadable PDF source"
                )
            else:
                status = DecisionStatus.REJECTED
                reason = "ranked below automatic selection target"
            decisions.append(
                CandidateDecision(
                    candidate=candidate,
                    status=status,
                    reason=reason,
                    score=frozen_scores[candidate.openalex_id],
                    decision_rank=rank,
                    download=download,
                )
            )

        for candidate in admitted:
            reasons = eligibility_reasons[candidate.openalex_id]
            if reasons:
                decisions.append(
                    CandidateDecision(
                        candidate=candidate,
                        status=DecisionStatus.REJECTED,
                        reason="; ".join(reasons),
                        score=frozen_scores[candidate.openalex_id],
                    )
                )
        for duplicate in duplicates:
            decisions.append(
                CandidateDecision(
                    candidate=duplicate.candidate,
                    status=DecisionStatus.REJECTED,
                    reason=f"duplicate by {duplicate.method}",
                    score=None,
                    duplicate_of=duplicate.duplicate_of,
                    deduplication_method=duplicate.method,
                    deduplication_similarity=duplicate.similarity,
                )
            )

        selected_count = self._validated_count(downloads)
        download_failures = sum(
            result.status is DownloadStatus.FAILED for result in downloads.values()
        )
        complete = (
            len(admitted) >= self.config.candidate_min
            and selected_count >= self.config.selection_min
        )
        timestamp = self.now().astimezone(timezone.utc).isoformat()
        request_records = [
            {
                "query": record.query,
                "page": record.page,
                "per_page": record.per_page,
                "result_count": record.result_count,
                "response_sha256": record.response_sha256,
                "cache_path": record.cache_path.as_posix() if record.cache_path else None,
                "parse_failures": list(record.parse_failures),
            }
            for record in self.openalex.request_records
        ]
        return CorpusManifest(
            schema_version="1.1",
            corpus_id=corpus_id,
            corpus_kind="frozen_base",
            topic=self.config.topic,
            modalities=self.config.modalities,
            created_at=timestamp,
            generator={"selection_strategy": "phase2-expansion-v1", **(generator or {})},
            openalex={
                "endpoint": self.config.openalex_api_url,
                "retrieved_at": timestamp,
                "queries": list(
                    self.used_queries
                    + (self.config.expansion_queries if triggered else ())
                ),
                "request_parameters": {
                    "per_page": self.config.per_query,
                    "sort": None,
                    "default_order": "relevance_score:desc",
                    "select": WORK_FIELDS,
                    "fallback_pages": 1,
                },
                "requests": request_records,
            },
            rules={
                "candidate_min": self.config.candidate_min,
                "candidate_max": self.config.candidate_max,
                "expansion_increment": self.config.expansion_increment,
                "expansion_candidate_max": self.config.expansion_candidate_max,
                "expansion_max_pages": self.config.expansion_max_pages,
                "selection_min": self.config.selection_min,
                "selection_target": self.config.selection_target,
                "selection_max": self.config.selection_max,
                "deduplication": "openalex-doi-title-fuzzy-author-year-v1",
                "fuzzy_title_threshold": 0.97,
                "ranking": {
                    "query_relevance": self.config.ranking.query_relevance,
                    "domain_relevance": self.config.ranking.domain_relevance,
                    "accessibility": self.config.ranking.accessibility,
                    "metadata_completeness": self.config.ranking.metadata_completeness,
                    "recency": self.config.ranking.recency,
                },
                "recency_reference_year": self.config.recency_reference_year,
                "quotas": "none",
            },
            discovery_expansion={
                "trigger_min_validated_pdfs": self.config.selection_min,
                "target_validated_pdfs": self.config.selection_target,
                "initial_unique_budget": self.config.candidate_max,
                "increment_size": self.config.expansion_increment,
                "maximum_unique_budget": self.config.expansion_candidate_max,
                "focused_queries": list(self.config.expansion_queries),
                "triggered": triggered,
                "score_policy": "frozen_at_first_admission",
                "candidate_provenance": provenance,
                "rounds": rounds,
                "stop_reason": stop_reason,
            },
            summary={
                "raw_results": raw_results,
                "unique_candidates": len(admitted),
                "duplicates": len(duplicates),
                "eligible": len(ordered_eligible),
                "selected": selected_count,
                "rejected": len(decisions) - selected_count,
                "download_failures": download_failures,
                "complete": complete,
            },
            papers=tuple(decisions),
        )

    def _freeze_candidate(
        self,
        candidate: LiteratureCandidate,
        *,
        admission_round: int,
        frozen_scores: dict[str, ScoreBreakdown],
        eligibility_reasons: dict[str, tuple[str, ...]],
        provenance: dict[str, dict[str, object]],
        retrieval_round: int = 0,
        retrieval_page: int | None = None,
    ) -> None:
        reasons, signals = exclusion_reasons(candidate, self.config.allowed_work_types)
        frozen_scores[candidate.openalex_id] = score_candidate(
            candidate, signals, self.config
        )
        eligibility_reasons[candidate.openalex_id] = reasons
        entry: dict[str, object] = {
            "first_admission_round": admission_round,
            "score_frozen_at_admission": True,
            "first_admission_query_matches": [
                self._query_match_record(match) for match in candidate.query_matches
            ],
            "retrievals": [],
        }
        provenance[candidate.openalex_id] = entry
        self._record_retrieval(
            entry,
            candidate,
            retrieval_round=retrieval_round,
            retrieval_page=retrieval_page,
        )

    def _acquire_ranked(
        self,
        candidates: Sequence[LiteratureCandidate],
        frozen_scores: Mapping[str, ScoreBreakdown],
        eligibility_reasons: Mapping[str, tuple[str, ...]],
        downloads: dict[str, DownloadRecord],
        *,
        corpus_id: str,
    ) -> tuple[int, int]:
        processed = 0
        validated_before = self._validated_count(downloads)
        for candidate in self._ordered_eligible(
            candidates, frozen_scores, eligibility_reasons
        ):
            if self._validated_count(downloads) >= self.config.selection_target:
                break
            if candidate.openalex_id in downloads:
                continue
            processed += 1
            has_pdf_source = bool(
                candidate.open_access.content_pdf_url
                or candidate.open_access.pdf_urls
            )
            if not candidate.open_access.is_oa or not has_pdf_source:
                downloads[candidate.openalex_id] = DownloadRecord(
                    status=DownloadStatus.FAILED,
                    failure_reason="not OA with a downloadable PDF source",
                )
                continue
            downloads[candidate.openalex_id] = self.downloader.download(
                paper_id=candidate.paper_id,
                pdf_urls=candidate.open_access.pdf_urls,
                destination_dir=self.config.pdf_dir / corpus_id,
                openalex_content_url=candidate.open_access.content_pdf_url,
            )
        return processed, self._validated_count(downloads) - validated_before

    @staticmethod
    def _validated_count(downloads: Mapping[str, DownloadRecord]) -> int:
        return sum(item.status is DownloadStatus.SUCCESS for item in downloads.values())

    @staticmethod
    def _ordered_eligible(
        candidates: Sequence[LiteratureCandidate],
        frozen_scores: Mapping[str, ScoreBreakdown],
        eligibility_reasons: Mapping[str, tuple[str, ...]],
    ) -> list[LiteratureCandidate]:
        eligible = [
            candidate
            for candidate in candidates
            if not eligibility_reasons[candidate.openalex_id]
        ]
        eligible.sort(
            key=lambda candidate: (
                -frozen_scores[candidate.openalex_id].total,
                -frozen_scores[candidate.openalex_id].accessibility,
                candidate.normalized_title,
                candidate.openalex_id,
            )
        )
        return eligible

    @staticmethod
    def _merge_query_matches(
        left: Sequence[QueryMatch], right: Sequence[QueryMatch]
    ) -> tuple[QueryMatch, ...]:
        merged = {
            (match.query, match.rank, match.relevance_score): match
            for match in (*left, *right)
        }
        return tuple(
            sorted(merged.values(), key=lambda match: (match.query, match.rank))
        )

    @staticmethod
    def _query_match_record(match: QueryMatch) -> dict[str, object]:
        return {
            "query": match.query,
            "rank": match.rank,
            "relevance_score": match.relevance_score,
        }

    def _record_retrieval(
        self,
        entry: dict[str, object],
        candidate: LiteratureCandidate,
        *,
        retrieval_round: int,
        retrieval_page: int | None,
    ) -> None:
        retrievals = entry["retrievals"]
        assert isinstance(retrievals, list)
        for match in candidate.query_matches:
            page = retrieval_page or ((match.rank - 1) // self.config.per_query) + 1
            record = {
                "round": retrieval_round,
                "query": match.query,
                "page": page,
                "rank": match.rank,
                "relevance_score": match.relevance_score,
            }
            if record not in retrievals:
                retrievals.append(record)

    @staticmethod
    def _existing_duplicate(
        admitted: Sequence[LiteratureCandidate], candidate: LiteratureCandidate
    ) -> tuple[int | None, DuplicateRecord | None]:
        for index, existing in enumerate(admitted):
            unique, duplicates = deduplicate_candidates([existing, candidate])
            if len(unique) == 1:
                if existing.openalex_id == candidate.openalex_id:
                    return index, None
                duplicate = duplicates[0]
                return index, DuplicateRecord(
                    candidate=candidate,
                    duplicate_of=existing.paper_id,
                    method=duplicate.method,
                    similarity=duplicate.similarity,
                )
        return None, None

    def _request_provenance(
        self, *, query: str, page: int, result_count: int, record_index: int
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "query": query,
            "page": page,
            "result_count": result_count,
        }
        records = self.openalex.request_records
        if len(records) > record_index:
            record = records[-1]
            result.update(
                {
                    "response_sha256": record.response_sha256,
                    "cache_path": record.cache_path.as_posix()
                    if record.cache_path
                    else None,
                    "parse_failures": list(record.parse_failures),
                }
            )
        return result

    @staticmethod
    def _round_robin(batches: Sequence[Sequence[LiteratureCandidate]]) -> list[LiteratureCandidate]:
        result: list[LiteratureCandidate] = []
        max_length = max((len(batch) for batch in batches), default=0)
        for index in range(max_length):
            for batch in batches:
                if index < len(batch):
                    result.append(batch[index])
        return result

    def _cap_candidate_pool(
        self, candidates: Sequence[LiteratureCandidate]
    ) -> list[LiteratureCandidate]:
        capped: list[LiteratureCandidate] = []
        for candidate in candidates:
            trial = capped + [candidate]
            unique, _ = deduplicate_candidates(trial)
            if len(unique) > self.config.candidate_max:
                break
            capped.append(candidate)
            if len(unique) == self.config.candidate_max:
                break
        return capped
