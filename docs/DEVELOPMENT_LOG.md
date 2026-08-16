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

---

### 2026-08-16 — Phase 1 contracts and package scaffolding

**Codex / AI-assisted work**
- Created the lightweight Python package structure without an agent framework.
- Added dependency-free typed configuration, scientific evidence/claim schemas,
  provider and tool protocols, and execution trace/failure contracts.
- Encoded approved invariants including one-based evidence pages, separate base
  and session evidence, RRF retrieval configuration, and two verifier calls total.
- Added minimal unit tests for configuration, schemas, interfaces, traces, and
  failure behavior.

**Human decisions / review**
- Approved a lightweight custom Python orchestration loop for a later phase.
- Froze a configurable single-provider abstraction and configurable local
  sentence-transformers embeddings without selecting concrete models.
- Required runtime literature expansion to remain isolated from the frozen base
  corpus and deferred ML dataset selection to a separate approval.
- Approved the Phase 1 scope and explicitly excluded API, PDF, embedding, LLM,
  orchestration, and ML implementations.

**Validation**
- Ran the Phase 1 suite with Python 3.11.15 and pytest 8.4.2 in the dedicated
  `l3s_agent_311` Conda environment: 17 tests passed.
- These unit-test results validate software contracts only and are not
  scientific or system-evaluation metrics.

**Known issues / next steps**
- Concrete LLM and embedding models remain intentionally unset.
- The solar-generation dataset remains unapproved and unimplemented.
- All external tools and the two-agent orchestration loop remain future work.

---

### 2026-08-17 — Phase 1 trace and verifier contract corrections

**Codex / AI-assisted work**
- Made session evidence opt-in and added validation requiring a concrete session ID.
- Extended execution traces to retain tool results, produced evidence IDs, and failures.
- Extended verifier traces to retain complete structured verification results.
- Added simple validation for inconsistent PASS results and unknown finding claim IDs.
- Clarified multimodal provider/tool layering and provisional execution-budget values.

**Human decisions / review**
- Selected the six corrections after a read-only Phase 1 review.
- Kept the existing configuration structure, ML approval placeholder, generic mappings,
  provisional search/tool limits, and dependency set unchanged.

**Validation**
- The corrected suite contains 23 passing unit tests in `l3s_agent_311`.
- Python compilation, dependency consistency, and Git whitespace checks pass.

**Known issues / next steps**
- Retrieval implementations must use the shared session-scope validator.
- Literature, PDF, retrieval, provider, orchestration, and ML implementations remain
  outside Phase 1.
