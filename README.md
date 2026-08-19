# An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy

This compact prototype helps investigate weather and climate
impacts on solar and wind energy. A Research Agent adaptively chooses bounded
tools for page-aware retrieval, multimodal PDF-page inspection, and local
scientific analysis; a separate Evidence Verifier checks whether submitted
claims are grounded. The focus is scientific provenance and explicit
limitations rather than a polished UI.

## Quick start

### A. Fresh clone: API-free review

Create the Python 3.11 environment defined by `environment.yml`. Its name is
`l3s_agent_311`.

```bash
conda env create -f environment.yml

conda run -n l3s_agent_311 python -m pytest

conda run -n l3s_agent_311 \
  python -m l3s_agent.system_evaluation validate

conda run -n l3s_agent_311 \
  python -m json.tool evaluation/system_results_summary.json
```

These commands need no API key, PDFs, retrieval index, model cache, or Berlin
dataset. Tests use synthetic fixtures and do not download an embedding model.

### B. Prepared local scientific artifacts

Retrieval and Research Agent demos additionally require ignored local artifacts
that are **not included in Git**:

```text
data/papers/base/...                         frozen PDFs recorded by the manifest
data/cache/base_index/evidence.jsonl        page-aware text Evidence
data/cache/base_index/pages/...             rendered page images
data/cache/retrieval/base/index_manifest.json
local Hugging Face cache                    exact frozen GTE model revision
```

Check the canonical artifacts before running a demo:

```bash
test -f data/cache/base_index/evidence.jsonl
test -f data/cache/base_index/ingestion_manifest.json
test -f data/cache/retrieval/base/index_manifest.json
```

The production Research Agent also requires an Apple system with MPS available,
because its current query-embedding path explicitly uses `device="mps"`.

### C. Berlin computed analysis

The Berlin dataset is **not included in this repository**. The computed-analysis
demo additionally requires:

```text
data/ml/berlin/Berlin_solar_regression.csv
```

Expected SHA-256:

```text
eda6fccb75d8e76d9ae56e806e20fcb12f017041e02d463d60a94817ee5656d8
```

The loader also requires the frozen 24-column schema, 36,296 rows, and target
`X50Hertz..MW.`. Literature-only and visual demos do not require this CSV.

## Architecture

The runtime has exactly two reasoning roles—no third agent:

1. **Research Agent**: chooses and sequences available tools, then produces a
   structured, support-grounded draft.
2. **Evidence Verifier**: receives the claims and referenced support in a
   separate stateless call and checks support, provenance, missing evidence,
   and conflicting evidence.

```text
Question
  → Research Agent chooses bounded tools
  → scientific support
  → grounded ResearchDraft
  → Evidence Verifier
  → answer or explicit limitation
```

Execution is bounded by typed actions, provisional MVP tool budgets, explicit
rejection of unavailable tools, and at most two verifier calls:

```text
gather support → draft → verify #1
  PASS → complete
  otherwise → bounded follow-up/revision → verify #2 → stop
```

Tool availability is intentionally explicit:

| Context | Available capabilities |
|---|---|
| Offline corpus construction | automated OpenAlex literature discovery, relevance filtering, OA PDF collection |
| Frozen production runtime | `retrieve_evidence`, `inspect_page`, `run_python` |
| Not enabled in frozen production runtime | `search_literature` |

## Scientific support and provenance

The system keeps three scientific-support classes distinct:

| Support class | Identity and scope | Supports | Verification boundary |
|---|---|---|---|
| Base text/page `Evidence` | Frozen paper, 1-based physical page, page-local chunk, source, and corpus provenance | Literature claims | Verifier checks the cited paper text and provenance supplied to it |
| Visual session `Evidence` | Derived from one checksummed canonical paper/page image; session-scoped with the same paper/page provenance | Figure and table claims | Verifier sees the derived visual Evidence text, not the raw image |
| `AnalysisResult` | Local computed result identified by dataset/spec/result identity through `analysis_result_id` | Computed claims | Verifier checks claim/result consistency; it does **not** reproduce the computation |

`tool_trace` records execution provenance—such as the Python call that produced
an `AnalysisResult`. It is not scientific support and cannot replace an
Evidence or AnalysisResult citation.

All PDF page provenance uses physical 1-based page order. Text chunks never
cross page boundaries.

## Demo commands

### Query the prepared retrieval index

BM25 requires the prepared index but does not compute a query embedding:

```bash
conda run -n l3s_agent_311 \
  python -m l3s_agent.retrieval.cli query \
  "How does numerical weather prediction contribute to wind-power forecasting?" \
  --mode bm25 \
  --top-k 5
```

Hybrid retrieval additionally requires the exact GTE revision already present
in the local model cache:

```bash
conda run -n l3s_agent_311 \
  python -m l3s_agent.retrieval.cli query \
  "How does numerical weather prediction contribute to wind-power forecasting?" \
  --mode hybrid \
  --top-k 5
```

The frozen embedding model is `Alibaba-NLP/gte-modernbert-base` at immutable
revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`, loaded with
`trust_remote_code=false`.

### Rebuild prepared artifacts

Ingestion requires all ten PDFs at the paths and hashes recorded by
`data/manifests/base_corpus.json`:

```bash
conda run -n l3s_agent_311 \
  python -m l3s_agent.ingestion.cli
```

Index construction additionally requires the exact embedding revision already
cached locally:

```bash
conda run -n l3s_agent_311 \
  python -m l3s_agent.retrieval.cli build
```

Both commands refuse silent replacement of completed artifacts.

### Run the live Research Agent

`OPENAI_API_KEY` must already be exported into the child process environment.
The application does not automatically load `.env`. If an ignored `.env` is
used as a shell convenience, load it explicitly without printing its contents:

```bash
set -a
source .env
set +a
```

The live command requires the canonical ingestion/retrieval artifacts, the
cached GTE revision, and current MPS support:

```bash
conda run -n l3s_agent_311 \
  python -m l3s_agent.runtime.cli \
  "How does numerical weather prediction contribute to wind-power forecasting?"
```

The Berlin question also requires the private CSV described above:

```bash
conda run -n l3s_agent_311 \
  python -m l3s_agent.runtime.cli \
  "Using the available Berlin weather and solar-generation data, how well can contemporaneous weather conditions predict 50Hertz regional solar generation, and which variables are most predictive?"
```

The CLI reports the answer, structured claims and citations, verifier status,
and terminal status. `--show-trace` adds the sanitized execution trace.

### Demonstrated capability examples

These questions correspond to observed capability categories; they are not
promises of deterministic routing on every live run.

- **Text-sufficient retrieval:** “How does numerical weather prediction
  contribute to wind-power forecasting, and how is NWP wind speed processed
  before producing the final power forecast?”
- **Visually dependent comparison:** “How do the clearness-index distributions
  and loss-of-load probability patterns differ between Romania and Dubai in
  January versus July?”
- **Computed analysis:** “Using the available Berlin weather and
  solar-generation data, how well can contemporaneous weather conditions
  predict 50Hertz regional solar generation, and which variables are most
  predictive?”
- **Mixed literature and computation:** “Which atmospheric and irradiance
  variables does the literature identify as important for photovoltaic power
  prediction, and do the Berlin–50Hertz results show a similar predictive
  pattern?”

## Evaluation snapshot

Evaluation separately asks whether the agent chose a useful action, retrieval
found manually verified paper/pages, and final claims followed from their cited
support. Offline retrieval cases are not pooled with live answer observations.

### Frozen six-query retrieval evaluation

| Mode | Hit@5 | MRR |
|---|---:|---:|
| BM25 | 0.833 | 0.611 |
| Dense | 0.500 | 0.417 |
| Hybrid RRF | 0.667 | 0.583 |

BM25 was strongest on this six-query, terminology-heavy frozen gold set. This
is an observed result, not evidence of universal BM25 superiority, and no
retrieval parameters or gold labels were tuned after measurement.

The three Gate 4A retrieval diagnostics expose page-level limitations:

- **T01:** correct paper retrieved, but no annotated gold page appeared in the
  top five.
- **T02:** correct paper retrieved, but no annotated gold page appeared in the
  top five.
- **T03:** a gold page appeared at rank 1.

### Seven real end-to-end observations

| Case | Type | Expected → actual tools | Support | Verifier | Human review | Outcome |
|---|---|---|---|---|---|---|
| M01 | historical | retrieve → retrieve | text | PASS | — | text-sufficient route |
| M02 | historical | retrieve + inspect → retrieve + inspect | text + visual | PASS | — | visual route |
| A01 | historical | Python → Python | computed | PASS | — | terminal pass |
| X01 | new live | retrieve + Python → retrieve + Python | text + computed | PASS | SUPPORTED | scoped mixed-provenance answer |
| O01 | new live | cautious/retrieve → retrieve | none cited | PASS | INSUFFICIENT_EVIDENCE | safe refusal; six retrievals |
| S01 | new live | retrieve → retrieve | text | PASS | PARTIALLY_SUPPORTED | grounded claims, incomplete wind coverage |
| I01 | new live | retrieve → retrieve | text | PASS | SUPPORTED | rejected a universal winner |

Verifier status was PASS in all seven documented observations. Terminal status
was recorded for five and was `pass` in 5/5; M01 and M02 are outside that
denominator. The four new-live cases were 4/4 routing-appropriate and 4/4
structurally grounded, with no unavailable-tool attempt and no verifier #2.
Historical human labels remain null rather than being inferred from verifier
status.

S01 makes the evaluation boundary explicit: its verifier status was **PASS**,
while human review was **PARTIALLY_SUPPORTED**. Verifier PASS means the claims
that were submitted were supported. It does not establish complete question
coverage, independent ground truth, or computational reproduction.

O01 is a successful robustness observation: the system declined to manufacture
coral-science claims from an unrelated renewable-energy corpus. Its lack of an
affirmative answer is appropriate, although using six retrieval calls before
stopping was inefficient.

The full frozen design and safe aggregate are in
`docs/EVALUATION_PLAN.md` and `evaluation/system_results_summary.json`.

## What this demo does not establish

- The automatically selected base corpus contains ten papers; it is not broad
  coverage of renewable-energy science.
- Retrieval evaluation uses six gold queries. T01 and T02 retrieved the correct
  papers but missed the annotated pages in the top five.
- The seven real observations are illustrative, not a statistically
  representative benchmark or a combined accuracy score.
- Live literature expansion is not enabled in the frozen production runtime.
- O01 required six retrieval calls before stopping safely.
- S01 did not fully cover wind-specific reliability despite supported submitted
  claims.
- The verifier checks derived visual Evidence text and does not independently
  re-read page images.
- The verifier checks AnalysisResult consistency and does not rerun Python.
- The Berlin data is private and absent from Git. Its task is regional,
  contemporaneous, and predictive—not site-level PV modeling, future
  forecasting, or causal inference.
- The current production embedding path requires prepared local artifacts, the
  exact local model cache, and Apple MPS.
- Single-run token and timing values are operational observations, not latency,
  cost, or efficiency benchmarks.

## Reproducibility and local artifacts

The Git-tracked source of truth includes:

- `data/manifests/base_corpus.json`: ten automatically selected OA papers with
  source URLs and PDF checksums;
- `evaluation/retrieval_gold.json`: six page-level retrieval questions;
- `evaluation/system_cases.json`: frozen Gate 4A case definitions;
- `evaluation/system_results_summary.json`: safe aggregate results;
- configuration, source, synthetic fixtures, and tests.

Exact full local reproduction additionally needs untracked artifacts:

- the frozen PDFs;
- the completed Phase 3 ingestion artifact and rendered pages;
- the completed Phase 4 retrieval index;
- the exact GTE model cache;
- the private Berlin CSV for computed-analysis runs.

The frozen manifest is sufficient to audit identifiers, source URLs, and
checksums, but the current corpus command performs fresh OpenAlex discovery; it
is not a bit-for-bit manifest replay command. OpenAlex results may change over
time. Live OpenAI calls are external and stochastic and likewise cannot be
reproduced bit-for-bit. Detailed live records remain in ignored local cache;
the repository tracks their safe aggregate.

## Compact implementation details

### Corpus, ingestion, and retrieval

- Six deterministic topic queries plus bounded focused expansion produced 80
  unique candidates and ten validated OA PDFs without manual paper selection.
- The frozen corpus contains four solar, four wind, and two cross-modality or
  general-renewable papers. Five focus on forecasting and five on climate
  impacts, variability, performance, or reliability.
- Phase 3 verified PDF SHA-256 values, extracted all 207 physical pages,
  rendered 207 checksummed 144-DPI RGB PNGs, and created 345 page-local chunks.
- Phase 4 scores `Evidence.content` only with BM25 (`k1=1.5`, `b=0.75`),
  normalized dense embeddings, and rank-only RRF (`rrf_k=60`, candidate depth
  50). Section labels are optional descriptive metadata, not ranking features.

### Berlin scientific analysis

The bounded operation `berlin_weather_solar_v1` predicts observed 50Hertz
regional solar generation from contemporaneous meteorological conditions. It
uses 2018 for training and 2019 for held-out testing without shuffling, lags,
future values, cross-validation, or hyperparameter search.

| Model | MAE (MW) | RMSE (MW) | R² |
|---|---:|---:|---:|
| Training-mean baseline | 661.328 | 757.954 | -0.0251 |
| Ridge | 211.282 | 288.315 | 0.8517 |
| HistGradientBoosting | 190.356 | 278.180 | 0.8619 |

The leading held-out permutation-importance variables were `Clearsky.GHI`,
`GHI`, `Wind.Direction`, `Precipitable.Water`, and `Surface.Albedo`. These are
predictive, noncausal rankings; correlated variables may redistribute
importance.

## Configuration, security, and deeper documentation

Non-secret defaults live in `config/default.toml`; environment overrides are
listed in `.env.example`. The Research Agent uses `gpt-5.6-terra`, while the
separate verifier uses `gpt-4.1-2025-04-14`. Provider calls are stateless and
use typed Structured Outputs with SDK retries disabled.

Never commit `.env`, API credentials, the Berlin CSV, PDFs, model caches, or
derived artifacts. Paper content is treated as untrusted scientific data, not
executable instructions. Prompts, provider payloads, credentials, hidden
reasoning, full Evidence content, and full AnalysisResult values are excluded
from safe tracked evaluation summaries.

Design and implementation details:

- `AGENTS.md`
- `docs/PROJECT_SPEC.md`
- `docs/REQUIREMENTS_MAPPING.md`
- `docs/EVALUATION_PLAN.md`
- `docs/DEVELOPMENT_LOG.md`

Core Python contracts:

- `src/l3s_agent/models.py`: Evidence, AnalysisResult, claims, drafts, and
  verification results;
- `src/l3s_agent/interfaces.py`: provider and Research Agent tool protocols;
- `src/l3s_agent/tracing.py`: action, tool, verifier, and failure provenance;
- `src/l3s_agent/runtime/`: bounded orchestration and production CLI.
