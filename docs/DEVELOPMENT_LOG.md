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

---

### 2026-08-17 — Phase 2 automated literature corpus construction

**Codex / AI-assisted work**
- Implemented a bounded OpenAlex Works client with required-key handling, retries,
  response caching/checksums, metadata normalization, and offline mock testing.
- Implemented deterministic multi-source relevance filtering/ranking across titles,
  reconstructed abstracts, OpenAlex topics/keywords, and matched-query provenance.
- Added DOI, normalized-title, and conservative fuzzy author/year deduplication with
  recorded provenance and merged repeated-query matches.
- Added OA PDF downloading, byte-level PDF validation, SHA-256 calculation,
  deterministic download backfill, and per-run PDF directories.
- Added stable manifest serialization with candidate decisions, score breakdowns,
  failures, exact request parameters, overwrite protection, and completeness status.
- Added the Phase 2 CLI and offline test fixtures without implementing PDF parsing,
  chunking, embeddings, RAG, vision, agent orchestration, or ML.
- Corrected OpenAlex search ordering to use the documented default relevance order
  and moved API-key authentication from the URL to a bearer header.
- Added credential-safe OpenAlex exceptions that suppress credential-bearing HTTP
  exception chains and regression coverage for strings, reprs, tracebacks, cached
  metadata, error records, and manifests.

**Human decisions / review**
- Froze the scientific topic, solar/wind modalities, work types, English-language
  policy, no-hard-year-cutoff policy, OA PDF requirement, 8/10/12 corpus bounds,
  35/35/20/5/5 ranking weights, and conservative fuzzy threshold.
- Required rule-driven selection without modality, conflict, regional, or evaluation
  quotas and required frozen base/runtime-session separation.
- Approved `httpx>=0.27,<1` as the sole new runtime dependency.

**Validation**
- The complete offline suite contains 43 passing tests in `l3s_agent_311`.
- Python compilation, dependency consistency, and Git whitespace checks pass.
- The live OpenAlex corpus build was not run because `OPENALEX_API_KEY` was not
  configured; no papers, citations, or scientific results were fabricated.

**Known issues / next steps**
- Configure `OPENALEX_API_KEY` in the environment before the first live build.
- Manually inspect the automatically generated corpus only after collection; do not
  replace its papers manually when creating evaluation labels.
- PDF parsing and indexing remain a later phase.

---

### 2026-08-17 — Sanitized OpenAlex transport diagnostics

**Codex / AI-assisted work**
- Retained only the final HTTPX transport exception class name and completed attempt
  count when bounded OpenAlex retries are exhausted.
- Added offline regression coverage for repeated timeouts, credential-safe exception
  rendering and records, and post-200 cache failures that must not trigger retries.

**Human decisions / review**
- Approved the minimal diagnostic change without altering request parameters, HTTP
  client construction, timeouts, backoff, redirects, caching, or corpus logic.

**Validation**
- The complete offline suite contains 47 passing tests in the dedicated Python 3.11
  environment; compilation, dependency consistency, and Git whitespace checks pass.

---

### 2026-08-17 — Phase 2 PDF acquisition and relevance precision corrections

**Codex / AI-assisted work**
- Added OpenAlex `has_content`/`content_urls` metadata and credential-safe content-PDF
  acquisition ahead of best and alternate OA-location URLs.
- Preserved PDF byte validation, SHA-256 generation, OA eligibility, deterministic
  backfill, ranking weights, and accessibility scoring.
- Added deterministic three-axis eligibility requiring renewable context, a physical
  weather/climate factor, a renewable-energy outcome, and a relationship linking them.
- Prevented matched queries, climate-mitigation language, resource-only forecasts, and
  generic efficiency/performance language from independently satisfying eligibility.
- Added a saved live-metadata regression fixture for the approved 24-retained/8-rejected
  result and offline tests for priority, fallback, secret safety, and score stability.

**Human decisions / review**
- Approved the acquisition priority and scientific-precision rule without embeddings,
  LLM classification, relevance-weight changes, or a lower corpus minimum.

**Validation**
- The complete offline suite contains 62 passing tests in the dedicated Python 3.11
  environment.
- No live OpenAlex or PDF requests were made during implementation.

---

### 2026-08-17 — Bounded Phase 2 discovery expansion

**Codex / AI-assisted work**
- Added deterministic focused-query expansion when the initial pool yields fewer
  than eight validated PDFs, in ten-candidate increments up to 90 unique works.
- Preserved first-admission scientific scores while retaining later duplicate-query
  retrieval provenance and recomputing only the deterministic final ordering.
- Recorded expansion triggers, query pages, response checksums, admission IDs,
  duplicate hits, acquisition outcomes, cumulative counts, and stop reasons in the
  reproducible corpus manifest.
- Added offline regression coverage for the 8/9-PDF no-expansion boundary, target
  completion, unique-budget and search-space exhaustion, deduplication, provenance,
  acquisition backfill, and unchanged scoring behavior.

**Human decisions / review**
- Froze the initial/increment/maximum unique-candidate budgets at 50/10/90, retained
  the 8/10 corpus thresholds, and approved three ordered focused fallback queries.
- Required scores to remain independent of later PDF-acquisition failures and kept
  all scientific eligibility, OA, ranking, PDF-validation, and authentication rules
  unchanged.

**Validation**
- No live OpenAlex or PDF requests were made during implementation.

---

### 2026-08-17 — Phase 2 base corpus frozen

**Codex / AI-assisted work**
- Promoted the approved v3 candidate manifest byte-for-byte to the official
  Git-trackable `data/manifests/base_corpus.json`; no discovery, ranking,
  selection, download, checksum, or provenance field was changed.
- Added a narrow ignore rule for candidate manifests while leaving the official
  base manifest trackable.
- Recorded the final corpus composition and integrity audit in the repository
  documentation.

**Human decisions / review**
- Approved the automatically constructed ten-paper corpus as the frozen base:
  four solar, four wind, and two cross-modality/general papers.
- Classified five papers as forecasting-focused and five as climate-impact,
  variability, performance, or reliability-focused.
- Recorded `W3126094341` as marginal because its implemented forecasting
  experiment relies mainly on historical PV output; it must not serve as
  primary gold evidence for meteorological-effect claims.
- Kept OpenAlex-hosted content as opportunistic after HTTP 401 responses during
  development; publisher/repository OA fallback remained necessary.

**Corpus construction record**
- The initial pool yielded six validated PDFs.
- Three focused expansion rounds reached 80 unique candidates and ten validated
  PDFs, stopping with `target_reached`.
- All ten selected local PDFs existed, were readable, and matched their manifest
  SHA-256 values.
- Candidate manifests remain local development artifacts; the official manifest
  is `data/manifests/base_corpus.json`.
