"""Deterministic query, relevance, deduplication, and ranking rules."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from difflib import SequenceMatcher
from typing import Iterable, Sequence

from ..config import LiteratureConfig
from .models import (
    DomainSignals,
    DuplicateRecord,
    LiteratureCandidate,
    ScoreBreakdown,
)


RENEWABLE_PATTERNS = {
    "renewable-energy": r"\brenewab\w*\b",
    "solar": r"\bsolar\w*\b|\bphotovolta\w*\b|\bpv\b",
    "wind-energy": r"\bwind\s+(?:energy|power|generation|farm|turbine)s?\b",
    "renewable-generation": r"\b(?:clean|green)\s+(?:energy|power)\b",
}
METEOROLOGICAL_PATTERNS = {
    "weather": r"\bweather\w*\b|\bmeteorolog\w*\b",
    "climate": r"\bclimat\w*\b",
    "irradiance": r"\birradia\w*\b|\binsolation\b",
    "temperature": r"\btemperature\w*\b|\bthermal\b",
    "cloud": r"\bcloud\w*\b|\bovercast\b",
    "humidity": r"\bhumid\w*\b",
    "wind-speed": r"\bwind\s+(?:speed|variab\w*|condition)s?\b",
    "precipitation": r"\bprecipitat\w*\b|\brainfall\b|\bsnow\w*\b",
    "atmosphere": r"\batmospher\w*\b|\bpressure\b",
    "extreme-weather": r"\b(?:storm|drought|heatwave|extreme\s+weather)\w*\b",
}
OUTCOME_PATTERNS = {
    "generation": r"\bgenerat\w*\b|\bpower\s+output\b|\benergy\s+yield\b",
    "forecasting": r"\bforecast\w*\b|\bpredict\w*\b",
    "reliability": r"\breliab\w*\b|\bresilien\w*\b",
    "variability": r"\bvariab\w*\b|\bfluctuat\w*\b|\bintermitten\w*\b",
    "performance": r"\bperformance\b|\befficien\w*\b",
}
PHYSICAL_WEATHER_PATTERNS = {
    key: pattern for key, pattern in METEOROLOGICAL_PATTERNS.items() if key != "climate"
}
# "Thermal" is useful as a broad ranking signal, but is too ambiguous for
# eligibility (for example, a photovoltaic-device topic is not weather evidence).
PHYSICAL_WEATHER_PATTERNS["temperature"] = r"\btemperature\w*\b"
PHYSICAL_WEATHER_PATTERNS["climatic-conditions"] = r"\bclimatic\s+conditions?\b"
ENERGY_OUTCOME_PATTERN = re.compile(
    r"\b(?:generat\w*|power(?:\s+output)?|output|energy\s+yield|energy\s+supply|"
    r"production|reliab\w*|resilien\w*|variab\w*|fluctuat\w*|intermitten\w*)\b"
)
RENEWABLE_CONTEXT_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in RENEWABLE_PATTERNS.values())
)
PHYSICAL_WEATHER_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in PHYSICAL_WEATHER_PATTERNS.values())
)
MITIGATION_CONTEXT_PATTERN = re.compile(
    r"\b(?:climate\s+change\s+mitigation|greenhouse\s+gas|emissions?|"
    r"paris\s+agreement|decarbon\w*|energy\s+transition|global\s+average\s+temperature)\b"
)
CLIMATE_TARGET_PATTERN = (
    r"(?:renewab\w*|solar\w*|photovolta\w*|\bpv\b|wind\s+(?:energy|power)|"
    r"energy\s+systems?|generat\w*|power\s+output|energy\s+supply|production|"
    r"reliab\w*|resilien\w*)"
)
CLIMATE_LINK_PATTERNS = (
    re.compile(
        rf"\bclimat\w*\b.{{0,60}}\b(?:impact\w*|affect\w*|influenc\w*|"
        rf"driv\w*|depend\w*|vulnerab\w*|effect\w*)\b.{{0,100}}{CLIMATE_TARGET_PATTERN}"
    ),
    re.compile(
        rf"\b(?:impact\w*|effect\w*)\s+of\s+\bclimat\w*\b.{{0,100}}"
        rf"{CLIMATE_TARGET_PATTERN}"
    ),
    re.compile(
        rf"{CLIMATE_TARGET_PATTERN}.{{0,60}}\b(?:under|due\s+to|dependent\s+on|"
        rf"driven\s+by|in(?:\s+real)?)\b.{{0,40}}\bclimat\w*\b"
    ),
)
WIND_FORECAST_TARGET_PATTERN = re.compile(r"\bwind\s+(?:power\s+)?forecast\w*\b")
ENGLISH_MARKERS = {
    "the",
    "and",
    "of",
    "in",
    "for",
    "with",
    "from",
    "under",
    "using",
    "on",
    "to",
}


def normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def normalize_doi(doi: str) -> str:
    normalized = doi.strip().casefold()
    normalized = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", normalized)
    return normalized.strip()


def normalized_first_author_surname(candidate: LiteratureCandidate) -> str | None:
    if not candidate.authors:
        return None
    name = unicodedata.normalize("NFKC", candidate.authors[0].display_name).casefold()
    tokens = re.findall(r"\w+", name)
    return tokens[-1] if tokens else None


def normalize_queries(queries: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned = " ".join(query.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return tuple(result)


def _signal_points(
    sources: Sequence[tuple[str, str, int]], patterns: dict[str, str], cap: int
) -> tuple[int, list[str]]:
    points = 0
    matched: list[str] = []
    for source_name, text, weight in sources:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        for label, pattern in patterns.items():
            if re.search(pattern, normalized):
                points += weight
                matched.append(f"{source_name}:{label}")
    return min(cap, points), matched


def domain_signals(candidate: LiteratureCandidate) -> DomainSignals:
    sources: list[tuple[str, str, int]] = [("title", candidate.title, 3)]
    if candidate.abstract:
        sources.append(("abstract", candidate.abstract, 1))
    for topic in candidate.topics:
        sources.append(("topic", topic, 2))
    for keyword in candidate.keywords:
        sources.append(("keyword", keyword, 2))
    for match in candidate.query_matches:
        sources.append(("query", match.query, 1))

    renewable, renewable_matches = _signal_points(sources, RENEWABLE_PATTERNS, 14)
    meteorological, weather_matches = _signal_points(sources, METEOROLOGICAL_PATTERNS, 14)
    outcomes, outcome_matches = _signal_points(sources, OUTCOME_PATTERNS, 7)
    matched = tuple(dict.fromkeys(renewable_matches + weather_matches + outcome_matches))
    return DomainSignals(
        renewable_points=renewable,
        meteorological_points=meteorological,
        outcome_points=outcomes,
        matched_signals=matched,
    )


def _normalized(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _substantive_sources(candidate: LiteratureCandidate) -> tuple[str, ...]:
    return tuple(
        text
        for text in (
            candidate.title,
            candidate.abstract or "",
            *candidate.topics,
            *candidate.keywords,
        )
        if text
    )


def _climate_output_link(text: str) -> bool:
    normalized = _normalized(text)
    return any(pattern.search(normalized) for pattern in CLIMATE_LINK_PATTERNS)


def _physical_weather_evidence(text: str) -> bool:
    normalized = _normalized(text)
    if PHYSICAL_WEATHER_PATTERN.search(normalized):
        return not MITIGATION_CONTEXT_PATTERN.search(normalized)
    return _climate_output_link(normalized)


def _renewable_energy_outcome(text: str) -> bool:
    normalized = _normalized(text)
    if WIND_FORECAST_TARGET_PATTERN.search(normalized):
        return True
    if not RENEWABLE_CONTEXT_PATTERN.search(normalized):
        return False
    return bool(ENERGY_OUTCOME_PATTERN.search(normalized))


def _relationship_units(candidate: LiteratureCandidate) -> tuple[str, ...]:
    units = [candidate.title]
    if candidate.abstract:
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", candidate.abstract)
            if sentence.strip()
        ]
        units.extend(sentences)
        units.extend(
            f"{left} {right}" for left, right in zip(sentences, sentences[1:])
        )
    return tuple(units)


def _has_weather_energy_link(candidate: LiteratureCandidate) -> bool:
    sources = _substantive_sources(candidate)
    renewable_present = any(
        RENEWABLE_CONTEXT_PATTERN.search(_normalized(text)) for text in sources
    )
    for unit in _relationship_units(candidate):
        if _renewable_energy_outcome(unit) and _physical_weather_evidence(unit):
            return True
        if (
            renewable_present
            and ENERGY_OUTCOME_PATTERN.search(_normalized(unit))
            and _physical_weather_evidence(unit)
        ):
            return True
        if renewable_present and _climate_output_link(unit):
            return True

    # Sparse OpenAlex records may lack an abstract. An explicit renewable-power
    # target in the title can be corroborated by physical-weather topics/keywords.
    metadata = (*candidate.topics, *candidate.keywords)
    if _renewable_energy_outcome(candidate.title) and any(
        _physical_weather_evidence(text) for text in metadata
    ):
        return True
    return False


def _three_axis_relevance(candidate: LiteratureCandidate) -> tuple[bool, bool, bool, bool]:
    sources = _substantive_sources(candidate)
    renewable = any(
        RENEWABLE_CONTEXT_PATTERN.search(_normalized(text)) for text in sources
    )
    meteorological = any(_physical_weather_evidence(text) for text in sources)
    outcome_sources = (candidate.title, candidate.abstract or "", *candidate.keywords)
    outcome = renewable and any(
        ENERGY_OUTCOME_PATTERN.search(_normalized(text))
        or WIND_FORECAST_TARGET_PATTERN.search(_normalized(text))
        for text in outcome_sources
    )
    if renewable and any(_climate_output_link(unit) for unit in _relationship_units(candidate)):
        outcome = True
        meteorological = True
    linkage = renewable and meteorological and outcome and _has_weather_energy_link(candidate)
    return renewable, meteorological, outcome, linkage


def is_clearly_english(candidate: LiteratureCandidate) -> bool:
    text = " ".join(part for part in (candidate.title, candidate.abstract or "") if part)
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    ascii_ratio = sum(character.isascii() for character in letters) / len(letters)
    tokens = set(re.findall(r"[a-z]+", text.casefold()))
    return ascii_ratio >= 0.9 and bool(tokens & ENGLISH_MARKERS)


def exclusion_reasons(
    candidate: LiteratureCandidate, allowed_work_types: Sequence[str]
) -> tuple[tuple[str, ...], DomainSignals]:
    reasons: list[str] = []
    if not candidate.title.strip():
        reasons.append("missing title")
    if candidate.is_retracted:
        reasons.append("retracted work")
    if candidate.work_type not in set(allowed_work_types):
        reasons.append(f"ineligible work type: {candidate.work_type or 'unknown'}")
    if candidate.language and candidate.language not in {"en", "eng"}:
        reasons.append(f"known non-English language: {candidate.language}")
    elif candidate.language is None and not is_clearly_english(candidate):
        reasons.append("unknown language without clear English title/abstract")

    signals = domain_signals(candidate)
    # Query provenance contributes to scoring but cannot satisfy scientific
    # eligibility. Eligibility requires all three substantive axes and a
    # deterministic weather-to-renewable-output linkage.
    renewable, meteorological, outcome, linkage = _three_axis_relevance(candidate)
    missing: list[str] = []
    if not renewable:
        missing.append("renewable-energy context")
    if not meteorological:
        missing.append("physical weather/climate factor")
    if not outcome:
        missing.append("renewable generation/output outcome")
    if missing:
        reasons.append("missing relevance axis: " + ", ".join(missing))
    elif not linkage:
        reasons.append("no weather-to-renewable-output relationship")
    return tuple(reasons), signals


def score_candidate(
    candidate: LiteratureCandidate,
    signals: DomainSignals,
    config: LiteratureConfig,
) -> ScoreBreakdown:
    ranks = [match.rank for match in candidate.query_matches]
    best_rank = min(ranks) if ranks else config.per_query
    rank_fraction = max(0.0, 1.0 - ((min(best_rank, 15) - 1) / 15.0))
    best_rank_score = 28.0 * rank_fraction
    coverage_score = 7.0 * min(len(candidate.query_matches), 3) / 3.0
    query_score = min(float(config.ranking.query_relevance), best_rank_score + coverage_score)

    domain_raw = (
        signals.renewable_points + signals.meteorological_points + signals.outcome_points
    )
    domain_score = config.ranking.domain_relevance * min(35, domain_raw) / 35.0

    if candidate.open_access.is_oa and candidate.open_access.pdf_urls:
        access_score = float(config.ranking.accessibility)
    elif candidate.open_access.is_oa and candidate.open_access.landing_page_url:
        access_score = config.ranking.accessibility / 2.0
    else:
        access_score = 0.0

    metadata_items = (
        bool(candidate.authors),
        candidate.year is not None,
        bool(candidate.abstract),
        bool(candidate.source_url),
        bool(candidate.doi),
    )
    metadata_score = config.ranking.metadata_completeness * sum(metadata_items) / 5.0

    if candidate.year is None:
        recency_score = 0.0
    else:
        age = config.recency_reference_year - candidate.year
        if age <= 6:
            recency_score = float(config.ranking.recency)
        elif age <= 16:
            recency_score = config.ranking.recency * 0.8
        elif age <= 26:
            recency_score = config.ranking.recency * 0.6
        else:
            recency_score = config.ranking.recency * 0.4

    total = query_score + domain_score + access_score + metadata_score + recency_score
    return ScoreBreakdown(
        query_relevance=round(query_score, 4),
        domain_relevance=round(domain_score, 4),
        accessibility=round(access_score, 4),
        metadata_completeness=round(metadata_score, 4),
        recency=round(recency_score, 4),
        total=round(total, 4),
        matched_signals=signals.matched_signals,
    )


def _duplicate_match(
    left: LiteratureCandidate, right: LiteratureCandidate
) -> tuple[str, float | None] | None:
    if left.openalex_id == right.openalex_id:
        return ("openalex_id", None)
    if left.doi and right.doi and left.doi == right.doi:
        return ("doi", None)
    doi_conflict = bool(left.doi and right.doi and left.doi != right.doi)
    if not doi_conflict and left.normalized_title == right.normalized_title:
        return ("normalized_title", 1.0)
    if doi_conflict or not left.normalized_title or not right.normalized_title:
        return None
    similarity = SequenceMatcher(None, left.normalized_title, right.normalized_title).ratio()
    left_surname = normalized_first_author_surname(left)
    right_surname = normalized_first_author_surname(right)
    year_close = (
        left.year is not None
        and right.year is not None
        and abs(left.year - right.year) <= 1
    )
    if similarity >= 0.97 and left_surname and left_surname == right_surname and year_close:
        return ("fuzzy_title_author_year", similarity)
    return None


def _representative_key(candidate: LiteratureCandidate) -> tuple[int, int, str]:
    metadata = sum(
        (
            bool(candidate.open_access.pdf_urls),
            bool(candidate.authors),
            candidate.year is not None,
            bool(candidate.abstract),
            bool(candidate.source_url),
            bool(candidate.doi),
        )
    )
    return (int(bool(candidate.open_access.pdf_urls)), metadata, candidate.openalex_id)


def deduplicate_candidates(
    candidates: Sequence[LiteratureCandidate],
) -> tuple[list[LiteratureCandidate], list[DuplicateRecord]]:
    if not candidates:
        return [], []
    parents = list(range(len(candidates)))
    edges: dict[tuple[int, int], tuple[str, float | None]] = {}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left_index, left in enumerate(candidates):
        for right_index in range(left_index):
            match = _duplicate_match(left, candidates[right_index])
            if match:
                union(left_index, right_index)
                edges[(min(left_index, right_index), max(left_index, right_index))] = match

    groups: dict[int, list[int]] = {}
    for index in range(len(candidates)):
        groups.setdefault(find(index), []).append(index)

    unique: list[tuple[int, LiteratureCandidate]] = []
    duplicates: list[DuplicateRecord] = []
    for member_indices in groups.values():
        representative_index = max(
            member_indices, key=lambda index: _representative_key(candidates[index])
        )
        representative = candidates[representative_index]
        merged_matches = sorted(
            {
                (match.query, match.rank, match.relevance_score): match
                for index in member_indices
                for match in candidates[index].query_matches
            }.values(),
            key=lambda match: (match.query, match.rank),
        )
        representative = replace(representative, query_matches=tuple(merged_matches))
        unique.append((min(member_indices), representative))
        for index in member_indices:
            if index == representative_index:
                continue
            direct = _duplicate_match(candidates[index], candidates[representative_index])
            if direct is None:
                related_edges = [
                    value
                    for pair, value in edges.items()
                    if index in pair and pair[0] in member_indices and pair[1] in member_indices
                ]
                direct = related_edges[0] if related_edges else ("transitive", None)
            if direct[0] == "openalex_id":
                # Repeated query hits are represented once with merged QueryMatch
                # provenance, not as duplicate paper entries with the same ID.
                continue
            duplicates.append(
                DuplicateRecord(
                    candidate=candidates[index],
                    duplicate_of=representative.paper_id,
                    method=direct[0],
                    similarity=direct[1],
                )
            )

    unique.sort(key=lambda item: item[0])
    duplicates.sort(key=lambda item: item.candidate.openalex_id)
    return [candidate for _, candidate in unique], duplicates


def rank_eligible_candidates(
    candidates: Sequence[LiteratureCandidate], config: LiteratureConfig
) -> tuple[
    list[tuple[LiteratureCandidate, ScoreBreakdown]],
    list[tuple[LiteratureCandidate, tuple[str, ...], DomainSignals]],
]:
    eligible: list[tuple[LiteratureCandidate, ScoreBreakdown]] = []
    rejected: list[tuple[LiteratureCandidate, tuple[str, ...], DomainSignals]] = []
    for candidate in candidates:
        reasons, signals = exclusion_reasons(candidate, config.allowed_work_types)
        if reasons:
            rejected.append((candidate, reasons, signals))
        else:
            eligible.append((candidate, score_candidate(candidate, signals, config)))
    eligible.sort(
        key=lambda item: (
            -item[1].total,
            -item[1].accessibility,
            item[0].normalized_title,
            item[0].openalex_id,
        )
    )
    return eligible, rejected
