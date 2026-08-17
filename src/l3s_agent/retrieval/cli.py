"""Build, query, or evaluate the Phase 4 retrieval index."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from ..config import load_config
from ..models import to_primitive
from .embeddings import SentenceTransformersEmbeddingProvider
from .engine import RetrievalEngine
from .evaluation import evaluate_modes, load_gold
from .index import RetrievalIndex, build_retrieval_index
from .models import RetrievalError, RetrievalMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "query", "evaluate"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", type=Path, default=Path("config/default.toml"))
        child.add_argument(
            "--evidence", type=Path, default=Path("data/cache/base_index/evidence.jsonl")
        )
        child.add_argument(
            "--index", type=Path, default=Path("data/cache/retrieval/base")
        )
    query = subparsers.choices["query"]
    query.add_argument("query")
    query.add_argument("--mode", choices=[mode.value for mode in RetrievalMode], default="hybrid")
    query.add_argument("--top-k", type=int, default=5)
    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--gold", type=Path, default=Path("evaluation/retrieval_gold.json"))
    return parser


def _provider(config):
    if config.embedding.model is None or config.embedding.revision is None:
        raise RetrievalError(
            "configure an explicitly verified embedding model revision before dense indexing"
        )
    return SentenceTransformersEmbeddingProvider(
        model_id=config.embedding.model,
        model_revision=config.embedding.revision,
        local_files_only=config.embedding.local_only,
        trust_remote_code=config.embedding.trust_remote_code,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    try:
        if args.command == "build":
            index = build_retrieval_index(
                evidence_path=args.evidence,
                output_dir=args.index,
                embedding_provider=_provider(config),
                bm25_k1=config.retrieval.bm25_k1,
                bm25_b=config.retrieval.bm25_b,
                rrf_k=config.retrieval.rrf_k,
                candidate_depth=config.retrieval.candidate_depth,
            )
            print(json.dumps(index.manifest, indent=2, sort_keys=True))
            return 0
        mode = RetrievalMode(getattr(args, "mode", "hybrid"))
        provider = None if mode is RetrievalMode.BM25 else _provider(config)
        index = RetrievalIndex.load(
            evidence_path=args.evidence,
            index_dir=args.index,
            embedding_provider=provider,
        )
        engine = RetrievalEngine(index)
        if args.command == "query":
            results = [to_primitive(asdict(item)) for item in engine.search(args.query, args.top_k, mode)]
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 0
        if provider is None:
            provider = _provider(config)
            index = RetrievalIndex.load(
                evidence_path=args.evidence,
                index_dir=args.index,
                embedding_provider=provider,
            )
            engine = RetrievalEngine(index)
        gold = load_gold(
            args.gold,
            expected_evidence_sha256=index.manifest["source"]["evidence_jsonl_sha256"],
        )
        print(json.dumps(evaluate_modes(engine, gold), indent=2, default=str))
        return 0
    except (OSError, ValueError, RetrievalError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
