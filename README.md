# An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy

Small take-home prototype for an L3S technical interview.

The goal is to build an agentic scientific-research assistant that can automatically collect scientific papers, retrieve multimodal evidence, perform a small ML analysis, and answer scientific questions with explicit provenance and uncertainty.

## Status

**Phases 1–6 implemented through a bounded local scientific-analysis tool over
the frozen corpus, checksummed hybrid retrieval index, and canonical PDF pages.**

The repository contains typed contracts, reproducible literature discovery,
page-aware PDF ingestion, BM25/dense/hybrid retrieval, a bounded two-role agent
runtime, stateless OpenAI Responses adapter, canonical page inspection, and a
fixed contemporaneous solar-generation regression experiment. Production text,
multimodal, and bounded scientific-analysis paths have approved smoke-test
observations; these are not benchmarks.

## Core design

The runtime system uses two agents:

1. **Research Agent**
   - decides which research tools are needed
   - searches literature
   - retrieves indexed evidence
   - inspects PDF pages / figures / tables
   - invokes Python or ML analysis when appropriate
   - produces evidence-grounded draft claims

2. **Evidence Verifier**
   - checks whether claims are supported by cited evidence
   - detects missing evidence
   - detects conflicting evidence
   - prevents unsupported overclaiming

Search, RAG, multimodal inspection, Python analysis, and ML analysis are tools rather than separate agents.

## Scientific scope

Broad topic:

**Weather and climate impacts on renewable-energy systems**

Primary literature focus:

- solar energy
- wind energy
- meteorological variables
- renewable-energy generation
- renewable-energy forecasting
- reliability and extreme-weather effects

Focused ML case study:

**contemporaneous Berlin-area weather and solar irradiance -> observed regional
50Hertz solar generation prediction**

This is neither future forecasting nor site-level PV prediction, and predictive
importance is not interpreted causally.

## Evaluation

The evaluation is organized around three questions:

1. Did the agent choose the right research action?
2. Did it retrieve the right scientific evidence?
3. Did the final claim actually follow from that evidence?

The planned evaluation set contains approximately 12 questions spanning:

- text RAG
- literature expansion
- multimodal figure/table inspection
- ML/data analysis
- mixed literature + ML reasoning
- missing evidence
- conflicting evidence

## Repository documentation

Before implementation, read:

- `AGENTS.md`
- `docs/PROJECT_SPEC.md`
- `docs/REQUIREMENTS_MAPPING.md`
- `docs/EVALUATION_PLAN.md`

Development-agent usage and implementation decisions are recorded in:

- `docs/DEVELOPMENT_LOG.md`

## Environment

Dedicated Conda environment:

```bash
l3s_agent_311
```

The repository should use only this environment for Python execution and dependency installation.

A reproducible environment definition is stored in:

```text
environment.yml
```

Create or update the dedicated environment, then run the offline tests:

```bash
conda env update -n l3s_agent_311 -f environment.yml
conda run -n l3s_agent_311 python -m pytest
```

Runtime dependencies include `httpx`, PyMuPDF, NumPy, scikit-learn, Sentence
Transformers, and Transformers. Tests use `pytest` and do not download an
embedding model.

## Configuration

Non-secret defaults are stored in `config/default.toml`. Phase 5B configures
`gpt-5.6-terra` for Research Agent decisions/drafting and the fixed
`gpt-4.1-2025-04-14` snapshot for the separate Evidence Verifier call. Phase 4
uses `Alibaba-NLP/gte-modernbert-base` as the local embedding model at verified
immutable revision
`e7f32e3c00f91d699e8c43b53106206bcc72bb22`.

The configuration freezes these boundaries:

- BM25 and dense retrieval combined by Reciprocal Rank Fusion
- chunks never crossing PDF page boundaries
- exactly two verifier calls at most
- separate frozen base-corpus and temporary session-evidence paths
- one approved local Berlin ML dataset; raw data remains outside Git

The Python contracts live under `src/l3s_agent/`:

- `models.py`: papers, evidence, claims, drafts, and verification data
- `interfaces.py`: configurable provider and Research Agent tool protocols
- `tracing.py`: tool/verifier traces and structured failures
- `config.py`: TOML loading, environment overrides, and invariant checks

`PageInspectionTool` is the Research-Agent-facing page-inspection capability.
It uses `LLMProvider.inspect_page` as the lower-level multimodal provider call;
the provider method is not exposed to the Research Agent as a separate tool.

The configured Phase 5A limits—six total tool calls, ten Research Agent
decisions, one literature search, two page inspections, one Python call, and
three follow-up tool calls—are provisional MVP safety limits, not scientifically
justified values. The maximum of two verifier calls is architecturally frozen.

## Phase 2 literature corpus

The frozen scientific topic is **Weather and climate impacts on renewable
energy**, with solar and wind as the main modalities. Phase 2:

1. runs six deterministic OpenAlex queries or accepts explicit queries;
2. collects 30–50 unique candidates using one bounded fallback page;
3. evaluates relevance across title, abstract, OpenAlex topics/keywords, and
   matched-query provenance;
4. deduplicates by OpenAlex ID, DOI, normalized title, and a conservative
   title/author/year fallback;
5. ranks candidates using the approved 35/35/20/5/5 score;
6. downloads and byte-validates only automatically selected OA PDFs;
7. backfills failed downloads until the target of ten is reached; and
8. writes a Git-trackable manifest with decisions, provenance, failures, and
   SHA-256 checksums.

Raw API caches and PDFs remain outside Git. The corpus builder never silently
overwrites an existing frozen manifest and imposes no solar/wind, conflict,
regional, or evaluation-answer quotas.

### Frozen base corpus

The official, Git-trackable base-corpus manifest is:

```text
data/manifests/base_corpus.json
```

It records ten automatically selected papers: four solar-focused, four
wind-focused, and two cross-modality/general-renewable papers. Five primarily
address forecasting; five address climate impacts, weather-driven variability,
performance, or reliability.

The initial 50-candidate pool yielded six validated PDFs. Three focused
discovery-expansion rounds increased the accumulated pool to 80 unique
candidates and reached ten validated PDFs, with stop reason `target_reached`.
All ten local PDFs passed SHA-256 verification against the frozen manifest.

During development, OpenAlex-hosted content downloads returned HTTP 401 with
the available API credentials. OpenAlex content therefore remains an
opportunistic first source, while publisher and repository OA fallbacks were
necessary to construct the complete corpus.

Paper `W3126094341` is retained as an automatically selected but scientifically
marginal corpus item: its implemented PV forecasting experiment relies mainly
on historical PV power rather than explicit meteorological predictors. It must
not be used as primary gold evidence for meteorological-effect claims.

Candidate manifests such as `base_corpus_candidate_v3.json` are local
development artifacts and are excluded from Git. Freezing the v3 result did
not change its selected papers, scores, rankings, download records, checksums,
or discovery provenance.

Live OpenAlex access requires an uncommitted environment variable:

```bash
export OPENALEX_API_KEY="..."
```

Build a new candidate corpus with:

```bash
conda run -n l3s_agent_311 python -m l3s_agent.literature.cli \
  --output data/manifests/base_corpus_candidate.json
```

The command exits with status 2 if fewer than eight validated PDFs can be
collected after the bounded fallback procedure. It does not parse, chunk,
render, inspect, or index PDFs.

## Phase 3 PDF ingestion

Phase 3 provides an offline, all-or-nothing ingestion command for the frozen
ten-paper base corpus:

```bash
conda run -n l3s_agent_311 python -m l3s_agent.ingestion.cli
```

Before reading a paper, ingestion verifies its local path and manifest SHA-256,
then opens it with PyMuPDF and requires a readable PDF with at least one page.
Every physical PDF page is represented internally as `page_index + 1`, rendered
to a 144-DPI RGB PNG with annotations visible, and extracted without OCR.

Text chunks remain within a single page, target approximately 500–800 tokens
using the deterministic `ceil(character_count / 4)` estimate, and use about 100
tokens of page-local overlap. Conservative section headings persist until the
next recognized heading, but remain metadata only and do not force chunk
boundaries. The final corrected section strategy is
`deterministic_conservative_regex_v2`; section labels are descriptive,
optional metadata and are not intended to be a strong retrieval-ranking
feature. OpenAlex work IDs remain `source_id` values.

The ignored `data/cache/base_index/` output contains:

```text
ingestion_manifest.json
pages.jsonl
evidence.jsonl
pages/{paper_id}/page_0001.png
```

The ingestion manifest records input, code, configuration, page-image, and
JSONL checksums. Existing output is never silently replaced. Phase 3 does not
perform retrieval, embedding, vision reasoning, LLM calls, or orchestration.

The finalized frozen-corpus ingestion contains 10 papers, 207 physical PDF
pages, 207 rendered page images, and 345 page-local text chunks. Its canonical
artifact path is `data/cache/base_index/`.

## Phase 4 retrieval

Phase 4 reads the finalized `evidence.jsonl` without changing or rechunking its
345 Evidence records. It implements three modes over `Evidence.content` only:

- local BM25 Okapi with `k1=1.5` and `b=0.75`;
- normalized Sentence Transformers embeddings with NumPy dot-product search;
- hybrid Reciprocal Rank Fusion using rank positions only, `rrf_k=60`, and a
  component depth of 50.

Titles and conservative section labels remain result provenance, not ranking
features. Hybrid ties resolve deterministically by fused score, best component
rank, then evidence ID. Results retain the complete Evidence object plus the
available BM25 rank/score, dense rank/score, and RRF score. The Phase 1
`RetrievalTool` adapter projects these rich results back to Evidence records and
explicitly rejects session-evidence retrieval in Phase 4.

Build and query are separate commands:

```bash
conda run -n l3s_agent_311 python -m l3s_agent.retrieval.cli build
conda run -n l3s_agent_311 python -m l3s_agent.retrieval.cli query \
  "How does cloud cover affect photovoltaic power forecasting?" --mode hybrid
```

The default model has an 8192-token context window, 768-dimensional embeddings,
and is loaded without remote executable model code. Its verified immutable
revision is recorded in configuration, and `local_only=true` prevents index
commands from silently downloading a model.

Derived artifacts under `data/cache/retrieval/base/` contain evidence-ID order,
BM25 statistics, normalized float32 embeddings, and checksummed metadata; they
do not duplicate evidence content and remain outside Git. Loading rejects a
changed source `evidence.jsonl`, incompatible configuration/provider metadata,
or altered index files. The canonical production index is
`data/cache/retrieval/base/`; Phase 5B loads it without rebuilding or tuning it.

The tracked `evaluation/retrieval_gold.json` contains six manually verified,
natural scientific questions spanning solar, wind, climate-impact, and
cross-modality topics. Relevance labels are one or more `(paper_id, physical
1-based page)` pairs—not exact chunk IDs—and contain no answers or manufactured
conflicts. The evaluator reports Hit@3, Hit@5, Page Recall@3, Page Recall@5,
MRR, and first relevant rank for BM25, dense, and hybrid modes:

```bash
conda run -n l3s_agent_311 python -m l3s_agent.retrieval.cli evaluate
```

Evaluation requires a built retrieval index. No retrieval metrics are claimed
yet.

## Phase 5A bounded agent runtime

Phase 5A adds a lightweight custom runtime under `src/l3s_agent/runtime/` with
exactly two reasoning roles: a Research Agent/orchestrator and a tool-free
Evidence Verifier. It uses typed actions for retrieval, literature discovery,
page inspection, Python analysis, drafting, and stopping; no prose parsing or
agent framework is involved.

The state machine is bounded:

```text
gather evidence -> draft -> verify #1
  PASS -> complete
  otherwise -> bounded follow-up/revision -> verify #2 -> complete
```

Structural draft errors such as unknown Evidence IDs are rejected before
verification and do not consume the two-call scientific-verification budget.
After a final non-PASS result, the runtime returns the draft together with the
unresolved verifier status, findings, and explicit uncertainty.

Base and session Evidence remain separate. Literature search returns discovery
metadata only and cannot become citable Evidence without a future ingestion
step. Fake page inspection produces session-scoped figure/table Evidence, while
Python analysis produces a distinct non-Evidence `AnalysisResult` with its own
stable computed-result provenance.

Every typed decision—including rejected decisions—is traced alongside tool
calls, tool results, Evidence IDs, sanitized failures, and complete verifier
results. Tests inject a fixed clock and scripted providers, require no network,
and include an offline integration with the existing Phase 4 retrieval adapter.
Phase 5A does not initialize a production LLM, multimodal model, runtime
literature ingestion pipeline, or real Python/ML analysis tool.

## Phase 5B production provider wiring

Phase 5B adds a concrete `OpenAIResponsesProvider` behind the existing
`LLMProvider` Protocol. Action selection, drafting, and verification are
separate stateless Responses API calls using strict Pydantic Structured Outputs;
there is no shared conversation state, `previous_response_id`, function-tool
execution, prose/JSON repair, or hidden SDK retry. Research calls use
`gpt-5.6-terra`; verification uses the separately configured
`gpt-4.1-2025-04-14` snapshot and receives only the draft, claims, and cited
Evidence.

Paper text is serialized as explicitly untrusted scientific data. Action
selection receives 600-character Evidence previews, while drafting and
verification receive full admitted/cited Evidence subject to the explicit
200,000-character request bound. The deterministic runtime still validates
actions, budgets, Evidence IDs, tool-call IDs, and the exact two-verifier limit.
The runtime verifier is a reliability mechanism, not the later independent
system-evaluation judge.

The Phase 5B production smoke test used only hybrid base-corpus retrieval;
literature search, page inspection, and Python analysis were unavailable for
that frozen run. The production factory validates and loads the existing
retrieval index, initializes the frozen GTE revision locally with remote model
code disabled, and uses MPS for query embeddings.

Real LLM calls require an uncommitted environment variable:

```bash
export OPENAI_API_KEY="..."
```

After separate approval of API billing and smoke testing, run one question with:

```bash
conda run -n l3s_agent_311 python -m l3s_agent.runtime.cli \
  "How does numerical weather prediction contribute to wind-power forecasting?"
```

The CLI prints only the final answer, claim citations/provenance, verifier
status, and terminal status by default. Add `--show-trace` for the sanitized
`ExecutionTrace`. Prompts, provider request objects, hidden reasoning, and
credentials are never printed.

In the production smoke test, the observed route was `retrieve (5 new) ->
retrieve (3 new, 2 duplicate) -> retrieve (0 new, 4 duplicate) -> draft ->
verifier PASS`. The runtime admitted 8 unique Evidence records, structurally
grounded 6 claims with valid cited Evidence IDs, received verifier `PASS`, and
selected no unavailable tool. This is a smoke-test observation, not a benchmark.

## Phase 5C canonical page inspection

Phase 5C adds `inspect_page` as a bounded Research Agent tool following the
principle “text retrieves the page; vision interprets the page.” It resolves
only `(paper_id, 1-based page)` entries from the completed Phase 3 artifact,
checks `pages.jsonl` and the selected PNG SHA-256, and never accepts a model- or
user-supplied filesystem path. One image and one bounded question are sent to
the existing stateless provider, and the typed interpretation becomes exactly
one session-scoped figure/table Evidence record with deterministic provenance.

The independent verifier may check cited derived visual Evidence as bounded
text, but it does not independently re-read the page image. Images, data URLs,
prompts, raw provider objects, and inspection content are excluded from safe
runtime events. Phase 5C passed a provider-level multimodal smoke test. In one
visually dependent evaluation, the Research Agent naturally retrieved the
relevant paper, inspected its canonical physical page, admitted one
session-scoped visual Evidence record with paper/page provenance, drafted with
mixed text and visual citations, and received verifier `PASS`. This is a smoke
test observation, not a benchmark or a claim that vision is always necessary.

## Phase 6 bounded scientific analysis

Phase 6 implements one predefined `RUN_PYTHON` operation:

```json
{"analysis": "berlin_weather_solar_v1"}
```

It compares a training-mean baseline, Ridge regression, and fixed
histogram-gradient-boosting model using 2018 for training and 2019 for held-out
testing, with no shuffle, cross-validation, hyperparameter search, target
normalization, lags, or future values. It reports held-out MAE, RMSE, R², and
noncausal permutation importance. Computed results remain non-Evidence
`AnalysisResult` values. Literature/page `Evidence` and computed
`AnalysisResult` are separate scientific-support classes: computed claims cite
a deterministic `analysis_result_id`, while `tool_trace` separately records the
producing Python call. The verifier checks claim/result consistency, including
metrics and limitations; it does not rerun or independently reproduce the
analysis.

In the approved production smoke, the Research Agent naturally selected
`RUN_PYTHON`, admitted the deterministic AnalysisResult, drafted four affirmative
computed claims citing its `analysis_result_id`, included the producing call in
`tool_trace`, and received verifier `PASS`. The verifier checked claim/result
consistency rather than independently reproducing the computation. This is a
smoke/evaluation observation, not a benchmark.

Held-out results were:

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---:|---:|---:|
| Training-mean baseline | 661.328 | 757.954 | -0.0251 |
| Ridge | 211.282 | 288.315 | 0.8517 |
| HistGradientBoosting | 190.356 | 278.180 | 0.8619 |

The target mean shifted from 974.655 MW in training to 856.106 MW in testing
(-12.163%). The leading held-out permutation-importance variables were
`Clearsky.GHI`, `GHI`, `Wind.Direction`, `Precipitable.Water`, and
`Surface.Albedo`; these are predictive, not causal, rankings.

The raw research dataset is intentionally untracked because no redistribution
or reproducible public acquisition record is present in this repository. A
local run requires:

```text
data/ml/berlin/Berlin_solar_regression.csv
```

The loader requires the frozen 24-column schema, 36,296 rows, target
`X50Hertz..MW.`, and SHA-256
`eda6fccb75d8e76d9ae56e806e20fcb12f017041e02d463d60a94817ee5656d8`.
If the file is absent or does not match, the bounded tool fails clearly without
training. Offline tests use only a small synthetic fixture.

## Secrets

API keys and local secrets should be stored in:

```text
.env
```

Do **not** commit `.env`.

Use `.env.example` as the template for required environment variables.

## Planned implementation areas

The implementation is expected to include:

- automated literature discovery (implemented)
- OA paper collection and checksums (implemented)
- PDF ingestion and page rendering (implemented and finalized for the frozen corpus)
- provenance-aware evidence objects
- hybrid retrieval (implemented with a canonical real dense index)
- bounded Research Agent orchestration (production provider wired and smoke-tested)
- Evidence Verifier (separate production model/context wired and smoke-tested)
- multimodal page inspection (implemented and smoke-tested)
- small solar-generation ML experiment (implemented as a bounded local tool)
- small page-level retrieval evaluator (implemented); final agent evaluation harness
- saved execution traces and evaluation results

## Important constraints

- Keep the prototype small and testable.
- Do not manually curate the final literature corpus.
- Do not fabricate papers, citations, metrics, or experimental results.
- Prefer scientific reliability over UI polish.
- Preserve paper/page provenance throughout the system.
- Return explicit uncertainty when evidence is insufficient or conflicting.

## Running the project

The corpus-building, ingestion, retrieval, and bounded Research Agent entry
points are available as documented above.
