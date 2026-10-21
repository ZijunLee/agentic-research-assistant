"""Benchmark the Evidence Verifier against SciFact human labels.

    # cost-free check: builds every input, estimates tokens, makes no API call
    python -m research_agent.benchmarks.verifier_eval run --dry-run
    # live run (OPENAI_API_KEY from the environment or from a .env file); resumable
    python -m research_agent.benchmarks.verifier_eval run --env-file .env \
        --models gpt-4.1-2025-04-14 gpt-4.1-mini-2025-04-14 --variants original injection
    python -m research_agent.benchmarks.verifier_eval report

Binary agreement treats verifier PASS as "supported" and SciFact SUPPORT as the positive
human label; the safety-relevant error is a false PASS on a CONTRADICT or NEI pair.
Confidence intervals resample claims (not pairs), because pairs of one claim are correlated.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..config import load_config
from ..models import VerifierInput, VerifierStatus
from . import scifact

STATUSES = tuple(s.value for s in VerifierStatus)
DEFAULT_MODELS = ("gpt-4.1-2025-04-14",)
RAW_DIR = Path("results/verifier_scifact")
REPORT_DIR = Path("evaluation/verifier_scifact")
DATA_DIR = Path("data/external/scifact")
MAX_ATTEMPTS = 5
TRANSIENT_STATUS = {408, 409, 429, 500, 502, 503, 504}
TRANSIENT_TYPES = {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}


# --------------------------------------------------------------------------- environment

def read_env_file(path: Path | None) -> dict[str, str]:
    """Process environment overlaid with KEY=VALUE lines of an optional .env file (values never printed)."""
    env = dict(os.environ)
    if path is None:
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            env[key.strip()] = value
    return env


# --------------------------------------------------------------------------- running

VerifyFn = Callable[[VerifierInput], tuple[Any, Mapping[str, int]]]


def openai_verify_factory(model: str, env: Mapping[str, str], config_path: str) -> Callable[[], VerifyFn]:
    """One provider per worker thread; the production provider, prompt and validation are reused."""
    from ..providers.openai import OpenAIProviderError, OpenAIResponsesProvider

    llm = load_config(config_path, environ=env).llm
    # SDK retries stay disabled (a project rule); transient failures are retried explicitly in verify()
    llm = dataclasses.replace(llm, verifier_model=model, timeout_seconds=120)
    if llm.text_model == llm.verifier_model:  # provider requires distinct roles; only verify() is used here
        llm = dataclasses.replace(llm, text_model=llm.text_model + "-unused")

    def make() -> VerifyFn:
        usage: dict[str, int] = {}

        def sink(event: Mapping[str, Any]) -> None:
            if event.get("marker") == "SAFE_PROVIDER_CALL_RETURNED":
                usage.clear()
                usage.update(event.get("usage") or {})

        provider = OpenAIResponsesProvider(llm, environ=env, event_sink=sink)

        def verify(verifier_input: VerifierInput):
            for attempt in range(MAX_ATTEMPTS):
                usage.clear()
                try:
                    result = provider.verify(verifier_input)
                    return result, {**usage, "attempts": attempt + 1}
                except OpenAIProviderError as exc:
                    transient = exc.status_code in TRANSIENT_STATUS or exc.error_type in TRANSIENT_TYPES
                    if not transient or attempt == MAX_ATTEMPTS - 1:
                        raise
                    time.sleep(min(60.0, 2.0 * 2 ** attempt))
            raise AssertionError("unreachable")

        return verify

    return make


def _row(pair: scifact.Pair, variant: str, model: str, result: Any, usage: Mapping[str, int],
         error: str | None, seconds: float) -> dict:
    row = {"pair_id": pair.pair_id, "claim_id": pair.claim_id, "doc_id": pair.doc_id, "label": pair.label,
           "variant": variant, "model": model, "status": None, "findings": [], "error": error,
           "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
           "attempts": usage.get("attempts", 1),
           "seconds": round(seconds, 2)}
    if result is not None:
        row["status"] = result.status.value
        row["findings"] = [{"status": f.status.value, "claim_id": f.claim_id, "reason": f.reason,
                            "requested_evidence": f.requested_evidence} for f in result.findings]
    return row


def raw_path(raw_dir: Path, model: str, variant: str) -> Path:
    return raw_dir / model / f"{variant}.jsonl"


def load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["pair_id"]] = row  # later lines (retries) win
    return rows


def run_variant(pairs: list[scifact.Pair], corpus: dict[int, scifact.Abstract], *, model: str, variant: str,
                make_verify: Callable[[], VerifyFn], raw_dir: Path, workers: int = 8, seed: int = 0,
                log: Callable[[str], None] = print) -> Path:
    """Verify every pair not yet answered successfully; appends to a resumable JSONL file."""
    path = raw_path(raw_dir, model, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {pid for pid, row in load_rows(path).items() if row["status"] is not None}
    todo = [p for p in pairs if p.pair_id not in done]
    log(f"{model} / {variant}: {len(done)} done, {len(todo)} to run")
    if not todo:
        return path

    local = threading.local()
    lock = threading.Lock()

    def work(pair: scifact.Pair) -> dict:
        if not hasattr(local, "verify"):
            local.verify = make_verify()
        verifier_input = scifact.build_input(pair, corpus, variant, seed=seed)
        start = time.time()
        try:
            result, usage = local.verify(verifier_input)
            return _row(pair, variant, model, result, usage, None, time.time() - start)
        except Exception as exc:  # recorded and retried on the next run
            return _row(pair, variant, model, None, {}, f"{type(exc).__name__}: {exc}"[:500], time.time() - start)

    failures = 0
    with open(path, "a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, p) for p in todo]
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            failures += row["error"] is not None
            with lock:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            if i % 50 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} (errors so far: {failures})")
    return path


# --------------------------------------------------------------------------- metrics

def cohen_kappa(y: np.ndarray, p: np.ndarray) -> float:
    po = float(np.mean(y == p))
    pe = float(np.mean(y) * np.mean(p) + (1 - np.mean(y)) * (1 - np.mean(p)))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def _binary_metrics(labels: list[str], passed: np.ndarray) -> dict[str, float]:
    labels_arr = np.array(labels)
    y = labels_arr == "SUPPORT"
    out = {
        "n": int(len(labels)),
        "accuracy": float(np.mean(y == passed)),
        "kappa": cohen_kappa(y, passed),
        "pass_rate": float(np.mean(passed)),
        "pass_precision": float(np.mean(y[passed])) if passed.any() else float("nan"),
    }
    for label in scifact.LABELS:
        mask = labels_arr == label
        out[f"pass_rate_{label}"] = float(np.mean(passed[mask])) if mask.any() else float("nan")
    rejected = labels_arr != "SUPPORT"
    out["false_pass_rate"] = float(np.mean(passed[rejected])) if rejected.any() else float("nan")
    return out


def cluster_bootstrap(rows: list[dict], stat: Callable[[list[dict]], float], *, n_boot: int = 1000,
                      seed: int = 0) -> tuple[float, float]:
    by_claim: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_claim[row["claim_id"]].append(row)
    clusters = list(by_claim.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = [r for i in rng.integers(0, len(clusters), len(clusters)) for r in clusters[i]]
        v = stat(sample)
        if not np.isnan(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarise(rows: list[dict], *, n_boot: int = 1000) -> dict:
    ok = [r for r in rows if r["status"] is not None]
    passed = np.array([r["status"] == "PASS" for r in ok], dtype=bool)
    summary = _binary_metrics([r["label"] for r in ok], passed) if ok else {"n": 0}
    summary["errors"] = len(rows) - len(ok)
    if ok:
        def kappa_of(sample):
            return _binary_metrics([r["label"] for r in sample],
                                   np.array([r["status"] == "PASS" for r in sample]))["kappa"]

        def false_pass_of(sample):
            return _binary_metrics([r["label"] for r in sample],
                                   np.array([r["status"] == "PASS" for r in sample]))["false_pass_rate"]

        summary["kappa_ci95"] = cluster_bootstrap(ok, kappa_of, n_boot=n_boot)
        summary["false_pass_rate_ci95"] = cluster_bootstrap(ok, false_pass_of, n_boot=n_boot)
        confusion = {label: Counter() for label in scifact.LABELS}
        for r in ok:
            confusion[r["label"]][r["status"]] += 1
        summary["confusion"] = {label: {s: confusion[label][s] for s in STATUSES} for label in scifact.LABELS}
    summary["input_tokens"] = int(sum(r["input_tokens"] for r in rows))
    summary["output_tokens"] = int(sum(r["output_tokens"] for r in rows))
    return summary


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts b and c."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_shift(base: dict[str, dict], other: dict[str, dict]) -> dict:
    """How a perturbation changes verdicts on the same pairs (only pairs answered in both runs)."""
    common = [pid for pid in base if pid in other and base[pid]["status"] and other[pid]["status"]]
    out: dict[str, Any] = {"n_paired": len(common)}
    if not common:
        return out
    flips = [(base[p]["status"] == "PASS") != (other[p]["status"] == "PASS") for p in common]
    out["pass_flip_rate"] = float(np.mean(flips))
    for label in scifact.LABELS:
        ids = [p for p in common if base[p]["label"] == label]
        if ids:
            b = np.mean([base[p]["status"] == "PASS" for p in ids])
            o = np.mean([other[p]["status"] == "PASS" for p in ids])
            out[f"delta_pass_rate_{label}"] = float(o - b)
    # false PASS on CONTRADICT/NEI pairs: newly passed (b) vs no longer passed (c)
    rejected = [p for p in common if base[p]["label"] != "SUPPORT"]
    b = sum(base[p]["status"] != "PASS" and other[p]["status"] == "PASS" for p in rejected)
    c = sum(base[p]["status"] == "PASS" and other[p]["status"] != "PASS" for p in rejected)
    out["false_pass_new"], out["false_pass_removed"] = int(b), int(c)
    out["false_pass_mcnemar_p"] = mcnemar_exact(int(b), int(c))
    return out


# --------------------------------------------------------------------------- report

def build_report(raw_dir: Path, *, n_boot: int = 1000) -> tuple[dict, str]:
    results: dict[str, dict] = {}
    for model_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()) if raw_dir.exists() else []:
        model = model_dir.name
        runs = {f.stem: load_rows(f) for f in sorted(model_dir.glob("*.jsonl"))}
        results[model] = {}
        for variant, rows in runs.items():
            entry = summarise(list(rows.values()), n_boot=n_boot)
            if variant != "original" and "original" in runs:
                entry["vs_original"] = paired_shift(runs["original"], rows)
            results[model][variant] = entry

    def pct(x):
        return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"

    lines = ["# Evidence Verifier on SciFact (dev split, claim × cited abstract)", "",
             "Positive class: SciFact SUPPORT ↔ verifier PASS. False PASS = PASS on a CONTRADICT or NEI pair. "
             "95% CIs: bootstrap over claims. Calls use the production verifier prompt and model defaults "
             "(no temperature set), so verdicts are not deterministic: the `shuffled` row is a practical "
             "noise floor for the perturbation comparisons.", ""]
    lines += ["## Agreement with human labels", "",
              "| model | variant | n | κ (95% CI) | accuracy | PASS on SUPPORT | false PASS (95% CI) "
              "| PASS on CONTRADICT | PASS on NEI | errors | tokens in / out |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    order = {v: i for i, v in enumerate(scifact.VARIANTS)}
    for model, variants in results.items():
        for variant in sorted(variants, key=lambda v: order.get(v, 99)):
            s = variants[variant]
            if not s.get("n"):
                continue
            k_lo, k_hi = s["kappa_ci95"]
            f_lo, f_hi = s["false_pass_rate_ci95"]
            lines.append(
                f"| {model} | {variant} | {s['n']} | {s['kappa']:.3f} ({k_lo:.2f}–{k_hi:.2f}) | {pct(s['accuracy'])} "
                f"| {pct(s['pass_rate_SUPPORT'])} | {pct(s['false_pass_rate'])} ({pct(f_lo)}–{pct(f_hi)}) "
                f"| {pct(s['pass_rate_CONTRADICT'])} | {pct(s['pass_rate_NEI'])} | {s['errors']} "
                f"| {s['input_tokens']:,} / {s['output_tokens']:,} |")
    lines += ["", "## Perturbations: verdict changes on the same pairs", "",
              "| model | variant | paired n | PASS flips | Δ PASS on SUPPORT | Δ PASS on CONTRADICT | Δ PASS on NEI "
              "| false PASS new / removed (McNemar p) |",
              "|---|---|---|---|---|---|---|---|"]
    for model, variants in results.items():
        for variant in sorted(variants, key=lambda v: order.get(v, 99)):
            shift = variants[variant].get("vs_original")
            if not shift or not shift.get("n_paired"):
                continue
            lines.append(
                f"| {model} | {variant} | {shift['n_paired']} | {pct(shift['pass_flip_rate'])} | "
                + " | ".join(f"{100 * shift.get(f'delta_pass_rate_{l}', float('nan')):+.1f} pp" for l in scifact.LABELS)
                + f" | {shift['false_pass_new']} / {shift['false_pass_removed']} (p={shift['false_pass_mcnemar_p']:.2g}) |")
    lines += ["", "## Confusion matrices (rows: SciFact label, columns: verifier status)", ""]
    for model, variants in results.items():
        for variant in sorted(variants, key=lambda v: order.get(v, 99)):
            s = variants[variant]
            if "confusion" not in s:
                continue
            lines += [f"**{model} / {variant}**", "", "| label | " + " | ".join(STATUSES) + " |",
                      "|---|" + "---|" * len(STATUSES)]
            for label in scifact.LABELS:
                lines.append(f"| {label} | " + " | ".join(str(s["confusion"][label][st]) for st in STATUSES) + " |")
            lines.append("")
    return results, "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI

def _load(args) -> tuple[list[scifact.Pair], dict[int, scifact.Abstract]]:
    data = scifact.download(Path(args.data_dir))
    pairs = scifact.load_pairs(data / "claims_dev.jsonl")
    corpus = scifact.load_corpus(data / "corpus.jsonl")
    if args.limit > 0:  # stratified prefix: same label mix, deterministic
        by_label: dict[str, list] = defaultdict(list)
        for p in pairs:
            by_label[p.label].append(p)
        share = args.limit / len(pairs)
        pairs = [p for label in scifact.LABELS for p in by_label[label][: max(1, round(share * len(by_label[label])))]]
    return pairs, corpus


def cmd_run(args) -> None:
    pairs, corpus = _load(args)
    counts = Counter(p.label for p in pairs)
    print(f"{len(pairs)} pairs {dict(counts)}; variants {args.variants}; models {args.models}")
    if args.dry_run:
        for variant in args.variants:
            chars = [len(json.dumps(dataclasses.asdict(scifact.build_input(p, corpus, variant, seed=args.seed)),
                                    default=str)) for p in pairs]
            # ~4 characters per token for the context, plus ~600 tokens of instructions and schema
            est = sum(chars) / 4 + 600 * len(pairs)
            print(f"  {variant}: ~{est / 1e6:.2f}M input tokens per model (estimate)")
        return
    env = read_env_file(Path(args.env_file) if args.env_file else None)
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (export it or put it in the --env-file)")
    for model in args.models:
        make = openai_verify_factory(model, env, args.config)
        for variant in args.variants:
            run_variant(pairs, corpus, model=model, variant=variant, make_verify=make,
                        raw_dir=Path(args.raw_dir), workers=args.workers, seed=args.seed)
    cmd_report(args)


def cmd_report(args) -> None:
    results, markdown = build_report(Path(args.raw_dir), n_boot=args.n_boot)
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out}/report.md and summary.json")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--raw-dir", default=str(RAW_DIR))
        p.add_argument("--report-dir", default=str(REPORT_DIR))
        p.add_argument("--n-boot", type=int, default=1000)
        if name == "run":
            p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
            p.add_argument("--variants", nargs="+", default=list(scifact.VARIANTS), choices=scifact.VARIANTS)
            p.add_argument("--data-dir", default=str(DATA_DIR))
            p.add_argument("--config", default="config/default.toml")
            p.add_argument("--env-file", default=None)
            p.add_argument("--limit", type=int, default=0, help="stratified subset size (0 = all 340 pairs)")
            p.add_argument("--workers", type=int, default=8)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    {"run": cmd_run, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
