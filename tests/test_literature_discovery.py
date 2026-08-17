from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path

from l3s_agent.config import load_config
from l3s_agent.literature.discovery import (
    deduplicate_candidates,
    domain_signals,
    exclusion_reasons,
    normalize_doi,
    normalize_queries,
    rank_eligible_candidates,
    score_candidate,
)
from l3s_agent.literature.models import LiteratureCandidate, OpenAccessInfo, QueryMatch


def test_query_normalization_is_ordered_and_deterministic() -> None:
    assert normalize_queries(["  weather   solar ", "weather solar", "wind climate"]) == (
        "weather solar",
        "wind climate",
    )


def test_domain_relevance_uses_topics_abstract_and_query_provenance(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Environmental Drivers of Power-System Variability",
        abstract="Atmospheric conditions affect the output in the studied system.",
        topics=("Photovoltaic Systems", "Climate Variability"),
        keywords=("energy yield",),
        query_matches=(QueryMatch("meteorological variables solar power generation", 2),),
    )

    reasons, signals = exclusion_reasons(candidate, ("article",))

    assert reasons == ()
    assert signals.renewable_points >= 3
    assert signals.meteorological_points >= 3
    assert any(signal.startswith("topic:") for signal in signals.matched_signals)


def test_known_non_english_and_non_research_types_are_rejected(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(language="de", work_type="book")
    reasons, _ = exclusion_reasons(candidate, ("article", "review"))
    assert "known non-English language: de" in reasons
    assert "ineligible work type: book" in reasons


def test_unknown_language_with_clearly_english_text_remains_eligible(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(language=None)
    reasons, _ = exclusion_reasons(candidate, ("article",))
    assert not any("language" in reason for reason in reasons)


def test_doi_and_exact_title_deduplication(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    first = candidate_factory(openalex_id="W1", doi=normalize_doi("https://doi.org/10.1/X"))
    same_doi = candidate_factory(openalex_id="W2", doi="10.1/x")
    same_title = candidate_factory(openalex_id="W3", doi=None)

    unique, duplicates = deduplicate_candidates([first, same_doi, same_title])

    assert len(unique) == 1
    assert {duplicate.method for duplicate in duplicates} == {"doi", "normalized_title"}


def test_repeated_openalex_hit_merges_query_provenance_without_duplicate_entry(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    first = candidate_factory(
        openalex_id="W1", query_matches=(QueryMatch("solar weather", 1),)
    )
    second = candidate_factory(
        openalex_id="W1", query_matches=(QueryMatch("renewable climate", 2),)
    )

    unique, duplicates = deduplicate_candidates([first, second])

    assert len(unique) == 1
    assert {match.query for match in unique[0].query_matches} == {
        "solar weather",
        "renewable climate",
    }
    assert duplicates == []


def test_conservative_fuzzy_dedup_records_similarity(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    first = candidate_factory(
        openalex_id="W1",
        doi=None,
        title=(
            "Weather Effects on Solar Power Generation under Variable "
            "Meteorological Conditions: A Regional Study"
        ),
    )
    second_title = (
        "Weather Effects on Solar Power Generation under Variable "
        "Meteorological Condition: A Regional Study"
    )
    second = candidate_factory(
        openalex_id="W2",
        doi=None,
        title=second_title,
    )

    unique, duplicates = deduplicate_candidates([first, second])

    assert len(unique) == 1
    assert duplicates[0].method == "fuzzy_title_author_year"
    assert duplicates[0].similarity is not None
    assert duplicates[0].similarity >= 0.97


def test_different_dois_prevent_title_or_fuzzy_merge(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    first = candidate_factory(openalex_id="W1", doi="10.1/one")
    second = candidate_factory(openalex_id="W2", doi="10.1/two")
    unique, duplicates = deduplicate_candidates([first, second])
    assert len(unique) == 2
    assert duplicates == []


def test_ranking_weights_total_100_and_citations_do_not_change_score(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    config = load_config(environ={}).literature
    low_citations = candidate_factory(openalex_id="W1", cited_by_count=0)
    high_citations = candidate_factory(openalex_id="W2", cited_by_count=100000)

    ranked, rejected = rank_eligible_candidates([low_citations, high_citations], config)

    assert rejected == []
    assert ranked[0][1].total == ranked[1][1].total
    assert sum(
        (
            config.ranking.query_relevance,
            config.ranking.domain_relevance,
            config.ranking.accessibility,
            config.ranking.metadata_completeness,
            config.ranking.recency,
        )
    ) == 100


def test_direct_pdf_scores_above_oa_landing_only(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    config = load_config(environ={}).literature
    direct = candidate_factory(openalex_id="W1")
    landing_only = candidate_factory(
        openalex_id="W2",
        open_access=OpenAccessInfo(True, "green", "https://repo/work", ()),
    )
    ranked, _ = rank_eligible_candidates([landing_only, direct], config)
    assert ranked[0][0].openalex_id == "W1"
    assert ranked[0][1].accessibility == 20
    assert ranked[1][1].accessibility == 10


def test_openalex_content_availability_does_not_change_accessibility_score(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    config = load_config(environ={}).literature
    without_content = candidate_factory(openalex_id="W1")
    with_content = candidate_factory(
        openalex_id="W2",
        open_access=replace(
            without_content.open_access,
            has_content_pdf=True,
            content_pdf_url="https://content.openalex.org/works/W2.pdf",
        ),
    )

    ranked, rejected = rank_eligible_candidates([without_content, with_content], config)

    assert rejected == []
    assert ranked[0][1].accessibility == ranked[1][1].accessibility == 20
    assert ranked[0][1].total == ranked[1][1].total


def test_query_provenance_cannot_supply_missing_weather_axis(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Maximum Power Point Tracking of Multiple Photovoltaic Arrays",
        abstract="A control algorithm improves photovoltaic array efficiency.",
        topics=("Photovoltaic System Optimization Techniques",),
        keywords=("maximum power point tracking",),
        query_matches=(QueryMatch("weather solar power generation", 1),),
    )

    reasons, _ = exclusion_reasons(candidate, ("article",))

    assert any("physical weather/climate factor" in reason for reason in reasons)


def test_climate_mitigation_transition_is_not_physical_weather_evidence(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Renewable Electricity Generation for a Sustainable Energy Transition",
        abstract=(
            "The Paris Agreement and greenhouse gas emissions motivate renewable "
            "energy transition policies that limit global average temperature."
        ),
        topics=("Global Energy and Sustainability Research",),
        keywords=("climate change mitigation", "renewable energy"),
    )

    reasons, _ = exclusion_reasons(candidate, ("article",))

    assert any("physical weather/climate factor" in reason for reason in reasons)


def test_weather_resource_forecast_without_energy_output_is_rejected(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Hourly solar irradiance prediction using weather forecasts",
        abstract=None,
        topics=("Solar Radiation and Photovoltaics",),
        keywords=("solar irradiance", "meteorology", "photovoltaic system"),
    )

    reasons, _ = exclusion_reasons(candidate, ("article",))

    assert any("renewable generation/output outcome" in reason for reason in reasons)


def test_generic_efficiency_under_weather_is_not_an_energy_output_relationship(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Efficiency optimization of photovoltaic arrays",
        abstract="Weather conditions are considered while improving generic efficiency.",
        topics=("Photovoltaic System Optimization Techniques",),
        keywords=("photovoltaic system", "weather"),
    )

    reasons, _ = exclusion_reasons(candidate, ("article",))

    assert any("renewable generation/output outcome" in reason for reason in reasons)


def test_explicit_climate_impact_on_renewable_supply_is_eligible(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Climate change impacts on renewable energy supply",
        abstract=None,
        topics=("Energy and Environment Impacts",),
        keywords=("renewable energy", "climate change", "wind power"),
    )

    reasons, _ = exclusion_reasons(candidate, ("article",))

    assert reasons == ()


def test_adjacent_sentences_can_link_weather_to_renewable_variability(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    candidate = candidate_factory(
        title="Complementarity of renewable energy sources",
        abstract=(
            "Energy demand is increasingly supplied by renewable energy sources. "
            "These weather-driven sources have substantial temporal variability."
        ),
        topics=("Hybrid Renewable Energy Systems",),
        keywords=("renewable energy",),
        work_type="review",
    )

    reasons, _ = exclusion_reasons(candidate, ("review",))

    assert reasons == ()


def test_saved_live_metadata_retains_24_and_rejects_8(
    candidate_factory: Callable[..., LiteratureCandidate],
) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "current_eligible_relevance.json"
    records = json.loads(fixture_path.read_text(encoding="utf-8"))
    config = load_config(environ={}).literature
    candidates: list[LiteratureCandidate] = []
    expected_by_id: dict[str, bool] = {}
    expected_scores: dict[str, float] = {}
    for record in records:
        candidate = candidate_factory(
            openalex_id=record["openalex_id"],
            openalex_url=f"https://openalex.org/{record['openalex_id']}",
            doi=f"10.9999/{record['openalex_id'].lower()}",
            title=record["title"],
            abstract=record["abstract"],
            topics=tuple(record["topics"]),
            keywords=tuple(record["keywords"]),
            query_matches=tuple(
                QueryMatch(query, index + 1)
                for index, query in enumerate(record["queries"])
            ),
        )
        candidates.append(candidate)
        expected_by_id[candidate.openalex_id] = record["expected_eligible"]
        expected_scores[candidate.openalex_id] = score_candidate(
            candidate, domain_signals(candidate), config
        ).total

    ranked, rejected = rank_eligible_candidates(candidates, config)
    retained_ids = {candidate.openalex_id for candidate, _ in ranked}
    rejected_ids = {candidate.openalex_id for candidate, _, _ in rejected}

    assert len(retained_ids) == 24
    assert len(rejected_ids) == 8
    assert retained_ids == {
        identifier for identifier, expected in expected_by_id.items() if expected
    }
    assert rejected_ids == {
        identifier for identifier, expected in expected_by_id.items() if not expected
    }
    assert all(
        score.total == expected_scores[candidate.openalex_id]
        for candidate, score in ranked
    )
