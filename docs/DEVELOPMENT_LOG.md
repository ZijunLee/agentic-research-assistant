# Development Log

This file records meaningful implementation decisions and the use of coding / AI agents during development.

The goal is to make the development process transparent and reproducible, especially because the take-home explicitly asks how coding / AI agents were used while building the system.

---

## Frozen design decisions before implementation

Date: 2026-08-16

Human-controlled decisions made before coding:

- Project title: **An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy**
- Target users: environmental / energy scientists
- Scientific scope: weather and climate impacts on renewable-energy systems, with solar and wind as the main literature focus
- ML case study: weather variables -> solar generation prediction
- Runtime architecture: **two agents**
  - Research Agent
  - Evidence Verifier
- Literature search, RAG, page inspection, Python analysis, and ML analysis remain tools rather than separate agents
- Final corpus should be collected automatically rather than manually curated
- Target corpus size: approximately 8-12 automatically selected open-access papers
- Retrieval design: hybrid lexical + dense retrieval where feasible
- Multimodal design: text retrieves the relevant page; vision interprets the page
- Verifier statuses:
  - PASS
  - NEED_MORE_EVIDENCE
  - UNSUPPORTED_CLAIM
  - CONFLICTING_EVIDENCE
- Maximum Research Agent <-> Verifier rounds: 2
- Evaluation should test three layers:
  1. tool / research-action selection
  2. evidence retrieval
  3. scientific claim support
- Approximate evaluation-set size: 12 questions
- No polished UI is required; CLI or simple Python entry point is sufficient
- Evaluation results must never be fabricated

Detailed design is frozen in:

- `AGENTS.md`
- `docs/PROJECT_SPEC.md`
- `docs/REQUIREMENTS_MAPPING.md`
- `docs/EVALUATION_PLAN.md`

---

## Development-agent usage policy

Codex may be used to accelerate:

- repository scaffolding
- API wrappers
- PDF parsing implementation
- retrieval implementation
- structured data models
- test skeletons
- debugging
- refactoring
- edge-case identification
- code review

Human review remains responsible for:

- scientific scope
- architecture
- agent responsibility decomposition
- evaluation methodology
- gold-label validation
- scientific comparability
- interpretation of results
- acceptance of major architectural changes

Codex should not silently change frozen design decisions.

Any proposed architectural deviation should be reviewed before implementation.

---

## Log template

For each meaningful development session, append an entry using this structure:

### YYYY-MM-DD — Short description

**Codex / AI-assisted work**
- What Codex implemented, debugged, reviewed, or proposed.

**Human decisions / review**
- What was manually checked or decided.

**Validation**
- Tests or commands that were run.
- Relevant outputs or observed behavior.

**Known issues / next steps**
- Remaining problems, limitations, or planned work.

---

## Initial repository setup

Date: 2026-08-16

**Human actions**
- Created a dedicated local project directory.
- Initialized a Git repository on branch `main`.
- Created a dedicated Conda environment: `l3s_agent_311`.
- Created the initial project-design documents.
- Added `.gitignore` rules for secrets, Python cache files, downloaded papers, caches, and generated results.

**Codex / AI-assisted work**
- No runtime-system code has been implemented yet.
- ChatGPT was used to structure the frozen architecture, evaluation strategy, and Codex handoff documents.

**Validation**
- `git status` confirmed the repository was initialized correctly.
- Project-specification files were manually inspected after creation.

**Next steps**
- finalize repository metadata and environment files
- make the first Git commit
- open the repository as a Codex Desktop project
- ask Codex to read the frozen design documents before modifying code
