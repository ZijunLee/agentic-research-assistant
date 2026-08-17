"""Command-line entry point for the live Phase 2 corpus build."""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..config import load_config
from .builder import CorpusBuilder
from .download import PDFDownloader
from .manifest import write_manifest
from .openalex import OpenAlexClient


def _git_metadata() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    return {"git_revision": revision or None, "git_worktree_dirty": dirty}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--corpus-id", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    literature = config.literature
    api_key = os.environ.get("OPENALEX_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "OPENALEX_API_KEY is not configured. Add it to the environment; "
            "never place the key in tracked configuration."
        )
    output = args.output or config.paths.base_corpus_manifest
    if output.exists():
        raise SystemExit(
            f"Refusing to overwrite existing frozen manifest {output}. "
            "Choose a new --output candidate path."
        )
    timestamp = datetime.now(timezone.utc)
    corpus_id = args.corpus_id or f"weather-renewable-energy-base-{timestamp:%Y%m%dT%H%M%SZ}"

    with OpenAlexClient(
        api_key=api_key,
        api_url=literature.openalex_api_url,
        timeout_seconds=literature.request_timeout_seconds,
        max_attempts=literature.max_http_attempts,
        cache_dir=literature.openalex_cache_dir / corpus_id,
    ) as openalex, PDFDownloader(
        timeout_seconds=literature.request_timeout_seconds,
        min_pdf_bytes=literature.min_pdf_bytes,
        max_pdf_bytes=literature.max_pdf_bytes,
        openalex_api_key=api_key,
    ) as downloader:
        builder = CorpusBuilder(config=literature, openalex=openalex, downloader=downloader)
        candidates = builder.collect_candidates(args.queries)
        manifest = builder.build_manifest(
            candidates,
            corpus_id=corpus_id,
            generator={
                "package_version": "0.1.0",
                "command": "python -m l3s_agent.literature.cli",
                **_git_metadata(),
            },
        )
        write_manifest(manifest, output)

    print(f"Manifest: {output}")
    print(f"Selected PDFs: {manifest.summary['selected']}")
    print(f"Complete: {manifest.complete}")
    return 0 if manifest.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
