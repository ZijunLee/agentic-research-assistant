"""Deterministic PDF ingestion for the frozen base corpus."""

from .pipeline import IngestionError, ingest_base_corpus

__all__ = ["IngestionError", "ingest_base_corpus"]
