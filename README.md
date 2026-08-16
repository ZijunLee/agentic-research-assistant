# An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy

Small take-home prototype for an L3S technical interview.

The goal is to build an agentic scientific-research assistant that can automatically collect scientific papers, retrieve multimodal evidence, perform a small ML analysis, and answer scientific questions with explicit provenance and uncertainty.

## Status

**Design frozen; implementation not started yet.**

The current repository contains the architecture and evaluation specification that the implementation should follow.

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

Implementation commands will be added after the first working version is built.

For now, this repository is the frozen design baseline for Codex-assisted implementation.
