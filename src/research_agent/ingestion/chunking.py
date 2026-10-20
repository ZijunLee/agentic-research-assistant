"""Deterministic, page-local text normalization and chunk construction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import ceil
import re


COMMON_HEADINGS = {
    "abstract",
    "acknowledgment",
    "acknowledgments",
    "acknowledgement",
    "acknowledgements",
    "conclusion",
    "conclusions",
    "data",
    "discussion",
    "introduction",
    "materials and methods",
    "method",
    "methodology",
    "methods",
    "references",
    "results",
    "results and discussion",
    "summary",
    "supplementary material",
}
NUMBERED_HEADING = re.compile(
    r"^(?P<number>(?:\d{1,2}(?:\.\d+)*\.?)|(?:[IVXLC]{1,6}\.))\s+"
    r"(?P<label>[A-Za-z][^.!?]{0,78})$"
)
SENTENCE_END = re.compile(r"[.!?](?:[\"')\]])?\s")
FALSE_HEADING_MARKERS = {
    "articles",
    "doi",
    "journal",
    "mae",
    "papers",
    "present address",
    "proceedings",
    "received",
    "rmse",
    "source",
    "table",
    "transactions",
    "volume",
}


@dataclass(frozen=True)
class ChunkDraft:
    section: str | None
    content: str
    approx_token_count: int
    content_sha256: str


def approximate_token_count(text: str) -> int:
    """Approximate tokens deterministically as ceil(character count / 4)."""

    return ceil(len(text) / 4) if text else 0


def normalize_page_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    result: list[str] = []
    blank = False
    for line in lines:
        if line:
            result.append(line)
            blank = False
        elif not blank:
            result.append("")
            blank = True
    return "\n".join(result).strip()


def recognize_heading(line: str) -> str | None:
    candidate = " ".join(line.strip().split())
    if not candidate:
        return None
    lowered = candidate.casefold().rstrip(":")
    if lowered in COMMON_HEADINGS:
        return candidate.rstrip(":")
    match = NUMBERED_HEADING.match(candidate)
    if match is None:
        return None
    label = match.group("label")
    normalized_label = label.casefold()
    if not 1 <= len(label.split()) <= 8:
        return None
    if any(marker in normalized_label for marker in FALSE_HEADING_MARKERS):
        return None
    if any(symbol in label for symbol in ("=", "±", "Δ", "|")):
        return None
    if any(character.isdigit() for character in label):
        return None
    if len(label.split()) == 1 and len(label) < 4:
        return None
    if len(candidate) <= 80:
        return candidate
    return None


def _prepare_page_text(
    text: str,
) -> tuple[str, tuple[int, ...], tuple[tuple[int, str], ...], tuple[str, ...]]:
    """Canonicalize text while retaining paragraph and heading positions."""

    parts: list[str] = []
    paragraph_boundaries: list[int] = []
    heading_markers: list[tuple[int, str]] = []
    headings: list[str] = []
    length = 0
    paragraph_break = False
    for line in text.splitlines():
        candidate = " ".join(line.split())
        if not candidate:
            paragraph_break = bool(parts)
            continue
        if parts:
            if paragraph_break:
                paragraph_boundaries.append(length)
            length += 1
        heading = recognize_heading(candidate)
        if heading:
            heading_markers.append((length, heading))
            headings.append(heading)
        parts.append(candidate)
        length += len(candidate)
        paragraph_break = False
    return (
        " ".join(parts),
        tuple(paragraph_boundaries),
        tuple(heading_markers),
        tuple(headings),
    )


def _choose_end(
    text: str,
    start: int,
    minimum: int,
    maximum: int,
    paragraph_boundaries: tuple[int, ...],
) -> int:
    if len(text) - start <= maximum:
        return len(text)
    lower = min(len(text), start + minimum)
    upper = min(len(text), start + maximum)
    paragraph_ends = [
        boundary for boundary in paragraph_boundaries if lower <= boundary <= upper
    ]
    if paragraph_ends:
        return paragraph_ends[-1]
    window = text[lower:upper]
    sentence_ends = [match.end() for match in SENTENCE_END.finditer(window)]
    if sentence_ends:
        return lower + sentence_ends[-1]
    whitespace = text.rfind(" ", lower, upper)
    return whitespace if whitespace > start else upper


def _chunk_intervals(
    canonical: str,
    *,
    target_min_tokens: int,
    target_max_tokens: int,
    overlap_tokens: int,
    paragraph_boundaries: tuple[int, ...] = (),
) -> list[tuple[int, int]]:
    if not canonical:
        return []
    minimum_chars = target_min_tokens * 4
    maximum_chars = target_max_tokens * 4
    overlap_chars = overlap_tokens * 4
    intervals: list[tuple[int, int]] = []
    start = 0
    while start < len(canonical):
        end = _choose_end(
            canonical, start, minimum_chars, maximum_chars, paragraph_boundaries
        )
        if canonical[start:end].strip():
            intervals.append((start, end))
        if end >= len(canonical):
            break
        next_start = max(start + 1, end - overlap_chars)
        whitespace = canonical.find(" ", next_start, end)
        start = whitespace + 1 if whitespace >= 0 else next_start
    return intervals


def _draft(section: str | None, content: str) -> ChunkDraft:
    digest = sha256(content.encode("utf-8")).hexdigest()
    return ChunkDraft(
        section=section,
        content=content,
        approx_token_count=approximate_token_count(content),
        content_sha256=digest,
    )


def _overlap_size(left: str, right: str) -> int:
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def merge_short_final_chunk(
    chunks: list[ChunkDraft], *, target_min_tokens: int, target_max_tokens: int
) -> list[ChunkDraft]:
    """Merge an avoidable short final remainder without exceeding the maximum."""

    if len(chunks) < 2 or chunks[-1].approx_token_count >= target_min_tokens:
        return chunks
    previous, final = chunks[-2:]
    overlap = _overlap_size(previous.content, final.content)
    remainder = final.content[overlap:].lstrip()
    combined = previous.content if not remainder else f"{previous.content} {remainder}"
    if approximate_token_count(combined) > target_max_tokens:
        return chunks
    return [*chunks[:-2], _draft(previous.section, combined)]


def chunk_section(
    text: str,
    *,
    section: str | None,
    target_min_tokens: int,
    target_max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkDraft]:
    canonical, paragraph_boundaries, _, _ = _prepare_page_text(text)
    chunks = [
        _draft(section, canonical[start:end].strip())
        for start, end in _chunk_intervals(
            canonical,
            target_min_tokens=target_min_tokens,
            target_max_tokens=target_max_tokens,
            overlap_tokens=overlap_tokens,
            paragraph_boundaries=paragraph_boundaries,
        )
    ]
    return merge_short_final_chunk(
        chunks,
        target_min_tokens=target_min_tokens,
        target_max_tokens=target_max_tokens,
    )


def chunk_page(
    text: str,
    *,
    current_section: str | None,
    target_min_tokens: int,
    target_max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[ChunkDraft], str | None, tuple[str, ...]]:
    canonical, paragraph_boundaries, heading_markers, headings = _prepare_page_text(text)
    intervals = _chunk_intervals(
        canonical,
        target_min_tokens=target_min_tokens,
        target_max_tokens=target_max_tokens,
        overlap_tokens=overlap_tokens,
        paragraph_boundaries=paragraph_boundaries,
    )
    chunks: list[ChunkDraft] = []
    for start, end in intervals:
        section = current_section
        for marker_position, marker_heading in heading_markers:
            if marker_position > start:
                break
            section = marker_heading
        chunks.append(_draft(section, canonical[start:end].strip()))
    chunks = merge_short_final_chunk(
        chunks,
        target_min_tokens=target_min_tokens,
        target_max_tokens=target_max_tokens,
    )
    final_section = current_section
    if heading_markers:
        final_section = heading_markers[-1][1]
    return chunks, final_section, headings
