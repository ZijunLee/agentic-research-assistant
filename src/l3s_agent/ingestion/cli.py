"""Build offline page-aware artifacts from the frozen base corpus."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ..config import load_config
from .pipeline import IngestionError, ingest_base_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.toml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    try:
        artifact = ingest_base_corpus(
            manifest_path=args.manifest or config.paths.base_corpus_manifest,
            output_dir=args.output or config.paths.base_index_dir,
            project_root=args.project_root,
            chunking=config.chunking,
            ingestion=config.ingestion,
        )
    except IngestionError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Ingestion artifact: {args.output or config.paths.base_index_dir}")
    print(f"Papers: {artifact.summary['papers']}")
    print(f"Pages: {artifact.summary['pages']}")
    print(f"Text chunks: {artifact.summary['text_chunks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
