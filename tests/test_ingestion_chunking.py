import pytest

from l3s_agent.ingestion.chunking import (
    ChunkDraft,
    approximate_token_count,
    chunk_page,
    chunk_section,
    merge_short_final_chunk,
    normalize_page_text,
    recognize_heading,
)


def overlap_characters(left: str, right: str) -> int:
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def test_token_estimator_is_frozen_character_count_rule() -> None:
    assert approximate_token_count("") == 0
    assert approximate_token_count("abcd") == 1
    assert approximate_token_count("abcde") == 2


def test_long_section_chunks_near_bounds_with_modest_overlap() -> None:
    text = " ".join(
        f"Solar generation sentence {index} responds to irradiance and cloud variability."
        for index in range(180)
    )

    chunks = chunk_section(
        text,
        section="Results",
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )

    assert len(chunks) > 1
    assert all(chunk.approx_token_count <= 800 for chunk in chunks)
    assert all(chunk.approx_token_count >= 500 for chunk in chunks[:-1])
    overlap = overlap_characters(chunks[0].content, chunks[1].content)
    assert 300 <= overlap <= 420


def test_short_page_emits_one_short_chunk() -> None:
    chunks, section, headings = chunk_page(
        "A short scientific page.",
        current_section=None,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )
    assert len(chunks) == 1
    assert chunks[0].approx_token_count < 500
    assert section is None
    assert headings == ()


def test_section_state_is_deterministic_and_persists_across_pages() -> None:
    page_one, current, headings = chunk_page(
        "Preface text before a heading.\n\n1. Introduction\nSolar weather context.",
        current_section=None,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )
    page_two, current, second_headings = chunk_page(
        "The analysis continues on the next physical page.",
        current_section=current,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )
    page_three, current, third_headings = chunk_page(
        "Conclusion\nThe final result.",
        current_section=current,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )

    assert len(page_one) == 1
    assert page_one[0].section is None
    assert "1. Introduction" in page_one[0].content
    assert headings == ("1. Introduction",)
    assert page_two[0].section == "1. Introduction"
    assert second_headings == ()
    assert page_three[0].section == "Conclusion"
    assert current == "Conclusion"
    assert third_headings == ("Conclusion",)


def test_normalization_is_conservative_and_deterministic() -> None:
    source = "line one  \r\n\r\n\r\nline two-\r\nbreak\t\r\n"
    assert normalize_page_text(source) == "line one\n\nline two-\nbreak"


@pytest.mark.parametrize(
    "candidate",
    [
        "JTECH-D-13-00104.1",
        "760 Environmental Chemistry Letters (2023) 21:741–764",
        "VOLUME 8, 2020",
        "= ±0.01 Δ LOLP",
        "450",
        "TABLE IX",
        "1.4 Data RMSE(m/s) MAE(m/s) Improvement",
        "4 Source 1 Source 2",
        "8 of",
        "10 papers",
        "30 articles",
        "I. INTRODUCTION extends from an hour ahead to 24 hours ahead and "
        "continues as a substantial body sentence",
    ],
)
def test_known_false_heading_examples_are_rejected(candidate: str) -> None:
    assert recognize_heading(candidate) is None


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("Introduction", "Introduction"),
        ("2 Methods", "2 Methods"),
        ("3.1 Data", "3.1 Data"),
        ("II. RESULTS", "II. RESULTS"),
    ],
)
def test_conservative_heading_examples_are_retained(candidate: str, expected: str) -> None:
    assert recognize_heading(candidate) == expected


def test_false_headings_do_not_fragment_page_text() -> None:
    body = " ".join("weather driven solar generation" for _ in range(350))
    text = (
        f"{body}\nJTECH-D-13-00104.1\n"
        "= ±0.01 Δ LOLP\n"
        "760 Environmental Chemistry Letters (2023) 21:741–764"
    )

    chunks, section, headings = chunk_page(
        text,
        current_section=None,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )

    assert headings == ()
    assert section is None
    assert all(chunk.approx_token_count <= 800 for chunk in chunks)
    assert all(chunk.approx_token_count >= 500 for chunk in chunks)
    assert any("JTECH-D-13-00104.1" in chunk.content for chunk in chunks)


def test_heading_metadata_does_not_force_a_chunk_boundary() -> None:
    before = " ".join("weather context" for _ in range(60))
    after = " ".join("solar generation evidence" for _ in range(70))

    chunks, final_section, headings = chunk_page(
        f"{before}\n\nResults\n{after}",
        current_section=None,
        target_min_tokens=500,
        target_max_tokens=800,
        overlap_tokens=100,
    )

    assert len(chunks) == 1
    assert chunks[0].section is None
    assert "Results" in chunks[0].content
    assert headings == ("Results",)
    assert final_section == "Results"


def test_short_final_chunk_merges_backward_when_combined_is_within_maximum() -> None:
    previous_content = " ".join("solar" for _ in range(400))
    overlap = previous_content[-400:]
    final_content = overlap + " " + " ".join("wind" for _ in range(100))
    chunks = [
        ChunkDraft("Results", previous_content, approximate_token_count(previous_content), "a"),
        ChunkDraft("Results", final_content, approximate_token_count(final_content), "b"),
    ]

    merged = merge_short_final_chunk(
        chunks, target_min_tokens=500, target_max_tokens=800
    )

    assert len(merged) == 1
    assert merged[0].approx_token_count <= 800
    assert merged[0].content.startswith(previous_content)
    assert merged[0].content.endswith("wind")


def test_chunking_is_deterministic_for_identical_input() -> None:
    text = "Results\n" + " ".join(
        f"Sentence {index} describes wind power under changing weather."
        for index in range(180)
    )
    arguments = {
        "current_section": None,
        "target_min_tokens": 500,
        "target_max_tokens": 800,
        "overlap_tokens": 100,
    }
    assert chunk_page(text, **arguments) == chunk_page(text, **arguments)
