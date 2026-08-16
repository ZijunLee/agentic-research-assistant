# An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy

Small take-home prototype for an L3S technical interview.

The goal is to build an agentic scientific-research assistant that can automatically collect scientific papers, retrieve multimodal evidence, perform a small ML analysis, and answer scientific questions with explicit provenance and uncertainty.

## Status

**Phase 1 contracts and package scaffolding implemented.**

The repository currently contains the frozen design plus typed configuration,
schemas, provider/tool interfaces, trace/failure contracts, and unit tests. It
does not yet contain literature, PDF, retrieval, LLM, orchestration, or ML
implementations.

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

Create or update the dedicated environment, then run the Phase 1 tests:

```bash
conda env update -n l3s_agent_311 -f environment.yml
conda run -n l3s_agent_311 python -m pytest
```

Phase 1 adds only `pytest`; all package code otherwise uses the Python standard
library.

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

## Secrets

API keys and local secrets should be stored in:

```text
.env
```

Do **not** commit `.env`.

Use `.env.example` as the template for required environment variables.

## Planned implementation areas

The implementation is expected to include:

- automated literature discovery
- OA paper collection
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

There is no runtime CLI yet. It will be added after the approved implementation
phases introduce working tools and the lightweight custom orchestration loop.
