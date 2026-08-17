"""Automated literature discovery and reproducible corpus construction."""

from .builder import CorpusBuilder
from .models import CorpusManifest, LiteratureCandidate
from .openalex import OpenAlexClient

__all__ = ["CorpusBuilder", "CorpusManifest", "LiteratureCandidate", "OpenAlexClient"]

