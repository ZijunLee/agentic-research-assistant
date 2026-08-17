"""Public contracts for the L3S scientific research assistant."""

from .config import AppConfig, load_config
from .models import (
    Claim,
    CorpusScope,
    Evidence,
    EvidenceModality,
    PageInspectionResult,
    PaperRecord,
    ResearchDraft,
    VerificationResult,
    VerifierInput,
    VerifierStatus,
)

__all__ = [
    "AppConfig",
    "Claim",
    "CorpusScope",
    "Evidence",
    "EvidenceModality",
    "PageInspectionResult",
    "PaperRecord",
    "ResearchDraft",
    "VerificationResult",
    "VerifierInput",
    "VerifierStatus",
    "load_config",
]
