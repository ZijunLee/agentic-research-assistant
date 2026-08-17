"""Small transparent BM25 Okapi implementation for the 345-chunk corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Mapping, Sequence


TOKENIZER_VERSION = "nfkc_casefold_alnum_v1"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize deterministically without stemming or a stop-word list."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(_TOKEN_PATTERN.findall(normalized))


@dataclass(frozen=True)
class BM25Index:
    document_term_frequencies: tuple[Mapping[str, int], ...]
    document_lengths: tuple[int, ...]
    document_frequencies: Mapping[str, int]
    average_document_length: float
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        count = len(self.document_term_frequencies)
        if count == 0 or len(self.document_lengths) != count:
            raise ValueError("BM25 requires aligned non-empty documents")
        if any(length <= 0 for length in self.document_lengths):
            raise ValueError("BM25 documents must contain at least one token")
        if self.average_document_length <= 0 or self.k1 <= 0 or not 0 <= self.b <= 1:
            raise ValueError("invalid BM25 parameters")

    @classmethod
    def build(
        cls, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75
    ) -> "BM25Index":
        if not documents:
            raise ValueError("BM25 requires at least one document")
        frequencies: list[Mapping[str, int]] = []
        lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()
        for document in documents:
            tokens = tokenize(document)
            if not tokens:
                raise ValueError("BM25 documents must contain at least one token")
            term_frequencies = Counter(tokens)
            frequencies.append(dict(sorted(term_frequencies.items())))
            lengths.append(len(tokens))
            document_frequencies.update(term_frequencies.keys())
        return cls(
            document_term_frequencies=tuple(frequencies),
            document_lengths=tuple(lengths),
            document_frequencies=dict(sorted(document_frequencies.items())),
            average_document_length=sum(lengths) / len(lengths),
            k1=k1,
            b=b,
        )

    def scores(self, query: str) -> tuple[float, ...]:
        query_terms = set(tokenize(query))
        scores = [0.0] * len(self.document_lengths)
        if not query_terms:
            return tuple(scores)
        document_count = len(self.document_lengths)
        for term in sorted(query_terms):
            document_frequency = self.document_frequencies.get(term, 0)
            if document_frequency == 0:
                continue
            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            for index, term_frequencies in enumerate(self.document_term_frequencies):
                frequency = term_frequencies.get(term, 0)
                if frequency == 0:
                    continue
                length_ratio = self.document_lengths[index] / self.average_document_length
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[index] += inverse_document_frequency * (
                    frequency * (self.k1 + 1) / denominator
                )
        return tuple(scores)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": self.k1,
            "b": self.b,
            "average_document_length": self.average_document_length,
            "document_lengths": list(self.document_lengths),
            "document_frequencies": dict(self.document_frequencies),
            "document_term_frequencies": [
                dict(frequencies) for frequencies in self.document_term_frequencies
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "BM25Index":
        if value.get("schema_version") != "1.0":
            raise ValueError("unsupported BM25 schema")
        if value.get("tokenizer_version") != TOKENIZER_VERSION:
            raise ValueError("unsupported BM25 tokenizer")
        raw_frequencies = value.get("document_term_frequencies")
        raw_lengths = value.get("document_lengths")
        raw_document_frequencies = value.get("document_frequencies")
        if not isinstance(raw_frequencies, list) or not isinstance(raw_lengths, list):
            raise ValueError("invalid BM25 document records")
        if not isinstance(raw_document_frequencies, dict):
            raise ValueError("invalid BM25 document frequencies")
        return cls(
            document_term_frequencies=tuple(
                {str(term): int(count) for term, count in frequencies.items()}
                for frequencies in raw_frequencies
                if isinstance(frequencies, dict)
            ),
            document_lengths=tuple(int(length) for length in raw_lengths),
            document_frequencies={
                str(term): int(count) for term, count in raw_document_frequencies.items()
            },
            average_document_length=float(value["average_document_length"]),
            k1=float(value["k1"]),
            b=float(value["b"]),
        )
