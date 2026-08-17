# An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy

Small take-home prototype for an L3S technical interview.

The goal is to build an agentic scientific-research assistant that can automatically collect scientific papers, retrieve multimodal evidence, perform a small ML analysis, and answer scientific questions with explicit provenance and uncertainty.

## Status

**Phase 1 contracts and Phase 2 literature-corpus builder implemented, with the
official ten-paper base corpus frozen.**

The repository contains typed contracts plus deterministic OpenAlex discovery,
rule-based relevance ranking, deduplication, OA PDF download/validation, and a
reproducible corpus manifest. PDF parsing, retrieval, LLM calls, orchestration,
and ML analysis are not implemented yet.

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

**weather variables -> solar generation prediction**

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

Phase 2 adds `httpx` for bounded HTTP calls and offline mock-transport tests.

## Phase 1 configuration

Non-secret defaults are stored in `config/default.toml`. Concrete LLM and local
sentence-transformers model names are intentionally unset. Configure them later
through TOML or the documented `L3S_*` environment variables after model
selection is approved.

The configuration freezes these boundaries:

- BM25 and dense retrieval combined by Reciprocal Rank Fusion
- chunks never crossing PDF page boundaries
- exactly two verifier calls at most
- separate frozen base-corpus and temporary session-evidence paths
- no ML dataset until separately approved

The Python contracts live under `src/l3s_agent/`:

- `models.py`: papers, evidence, claims, drafts, and verification data
- `interfaces.py`: configurable provider and Research Agent tool protocols
- `tracing.py`: tool/verifier traces and structured failures
- `config.py`: TOML loading, environment overrides, and invariant checks

`PageInspectionTool` is the Research-Agent-facing page-inspection capability.
It uses `LLMProvider.inspect_page` as the lower-level multimodal provider call;
the provider method is not exposed to the Research Agent as a separate tool.

The configured limits of two search rounds and twelve tool calls are provisional
MVP safety limits for bounded execution, not scientifically justified values.

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
- PDF ingestion and page rendering
- provenance-aware evidence objects
- hybrid retrieval
- Research Agent orchestration
- Evidence Verifier
- multimodal page inspection
- small solar-generation ML experiment
- evaluation harness
- saved execution traces and evaluation results

## Important constraints

- Keep the prototype small and testable.
- Do not manually curate the final literature corpus.
- Do not fabricate papers, citations, metrics, or experimental results.
- Prefer scientific reliability over UI polish.
- Preserve paper/page provenance throughout the system.
- Return explicit uncertainty when evidence is insufficient or conflicting.

## Running the project

The Phase 2 corpus-building CLI is available as documented above. There is no
runtime research-agent CLI yet; that will be added only after later approved
phases implement retrieval and the lightweight custom orchestration loop.
