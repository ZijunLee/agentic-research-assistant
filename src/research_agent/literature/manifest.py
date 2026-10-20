"""Stable, Git-trackable corpus manifest serialization."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import CorpusManifest, DecisionStatus, DownloadStatus


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value):
        return {key: _primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_primitive(item) for item in value]
    return value


def manifest_to_dict(manifest: CorpusManifest) -> dict[str, Any]:
    data = _primitive(manifest)
    papers = data["papers"]
    papers.sort(
        key=lambda paper: (
            0 if paper["status"] == DecisionStatus.SELECTED.value else 1,
            paper["decision_rank"] if paper["decision_rank"] is not None else 10**9,
            paper["candidate"]["openalex_id"],
        )
    )
    for paper in papers:
        oa = paper["candidate"]["open_access"]
        oa["pdf_url"] = oa["pdf_urls"][0] if oa["pdf_urls"] else None
    return data


def validate_manifest(manifest: CorpusManifest) -> None:
    selected = [paper for paper in manifest.papers if paper.status is DecisionStatus.SELECTED]
    for paper in selected:
        download = paper.download
        if download.status is not DownloadStatus.SUCCESS:
            raise ValueError(f"selected paper {paper.candidate.paper_id} has no successful PDF")
        if not download.sha256 or len(download.sha256) != 64:
            raise ValueError(f"selected paper {paper.candidate.paper_id} has invalid SHA-256")
        try:
            int(download.sha256, 16)
        except ValueError as exc:
            raise ValueError(
                f"selected paper {paper.candidate.paper_id} has invalid SHA-256"
            ) from exc
        if download.local_path is None:
            raise ValueError(f"selected paper {paper.candidate.paper_id} has no local PDF path")
    summary_selected = int(manifest.summary.get("selected", -1))
    if summary_selected != len(selected):
        raise ValueError("manifest selected count does not match paper decisions")


def write_manifest(manifest: CorpusManifest, path: Path) -> None:
    """Validate and atomically write a new manifest without overwriting it."""

    validate_manifest(manifest)
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite frozen corpus manifest: {path}; choose a new output path"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest_to_dict(manifest), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
