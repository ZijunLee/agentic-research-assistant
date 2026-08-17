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

---

### 2026-08-17 — Phase 3 page-aware PDF ingestion implementation

**Codex / AI-assisted work**
- Added an offline PyMuPDF ingestion package that validates every frozen PDF
  against the official manifest before page extraction or rendering.
- Added deterministic physical-page numbering, 144-DPI RGB PNG rendering,
  page-local chunking, conservative section propagation, evidence identifiers,
  structured warnings/failures, and all-or-nothing artifact creation.
- Added checksummed `pages.jsonl`, `evidence.jsonl`, page-image, and ingestion
  metadata artifacts for later retrieval and multimodal page inspection.
- Added synthetic-PDF offline tests for deterministic output, 1-based pages,
  empty pages, section propagation, integrity failures, encryption, invalid and
  zero-page PDFs, output overwrite refusal, and source-manifest immutability.

**Human decisions / review**
- Froze OpenAlex work IDs as `source_id`, physical `page_index + 1` numbering,
  `ceil(character_count / 4)` token estimation, and page-local overlap.
- Froze PNG/RGB/144-DPI rendering with visible annotations and no OCR or image
  post-processing.
- Required deterministic section persistence, location-plus-content-hash
  evidence IDs, and explicit output paths for rebuilds.

**Scope boundary**
- No BM25, embeddings, RRF, retrieval ranking, LLM, vision reasoning, agent,
  verifier, ML, or evaluation-question implementation was added.
- The Phase 3 command was not run on the real frozen corpus during implementation.

**Validation**
- The complete offline suite contains 82 passing tests. PyMuPDF emits five
  upstream SWIG deprecation warnings under Python 3.11; no test fails.
- Compilation, dependency consistency, and Git whitespace checks pass.

---

### 2026-08-17 — Phase 3 chunk-quality correction

**Codex / AI-assisted work**
- Decoupled deterministic section metadata from page-local chunk boundaries;
  chunks now follow page, paragraph, sentence, and word boundaries only.
- Tightened heading recognition to reject manuscript IDs, running journal
  headers, table/equation fragments, isolated values, and heading-plus-body lines.
- Added backward merging for avoidable short final remainders when the merged
  chunk remains within the 800-token approximation limit.
- Added offline regressions for false headings, metadata-only sections, short
  pages, final merges, page isolation, maximum size, and deterministic output.

**Scope boundary**
- PDF validation, page rendering, provenance, evidence IDs, checksums, and
  overwrite protection remain unchanged.
- The corrected ingestion pipeline was not run on the real frozen corpus.

---

### 2026-08-17 — Phase 3 frozen-corpus finalization

**Codex / AI-assisted work**
- Rebuilt the corrected ingestion at the canonical ignored cache path
  `data/cache/base_index/` using only the frozen manifest and existing local
  PDFs; no discovery, downloading, or network access was performed.
- Preserved the old v1 cache during comparison, verified the canonical rebuild
  against the approved v2 artifact, then removed both temporary derived cache
  copies.
- Recorded the corrected `deterministic_conservative_regex_v2` section
  strategy as final. Section labels remain descriptive, optional metadata and
  are not intended as a strong retrieval-ranking feature.

**Final real-ingestion result**
- 10 papers, 207 physical PDF pages, 207 rendered page images, and 345
  page-local text chunks.
- Approximate chunk-token statistics: minimum 131, median 654, mean 581.07,
  maximum 800; zero chunks below 100, 230 chunks from 500 through 800, and
  zero chunks above 800.
- One nonfatal warning remains for the known empty-text physical page 2 in
  `paper_W1984703120`; its page record and rendered image are retained.
- All JSONL, per-image, and image-set checksums passed. No page zero or
  cross-page chunk was found.
- The frozen `base_corpus.json` SHA-256 remained
  `51ea0a37783949f3d6da9eefb86a6fb23de7a2d100f1c807a52bbdf4715663c1`.

---

### 2026-08-17 — Phase 4 hybrid retrieval and page-level evaluation

**Codex / AI-assisted work**
- Added a transparent local BM25 Okapi implementation, configurable dense
  embedding provider, normalized NumPy search, and deterministic rank-only RRF.
- Added separate, checksummed index build/load operations that bind indexes to
  the finalized Phase 3 evidence checksum without duplicating Evidence content.
- Added rich retrieval diagnostics and a Phase 1 `RetrievalTool` adapter that
  explicitly rejects session evidence during this base-corpus-only phase.
- Added an offline evaluator for Hit@3/5, Page Recall@3/5, MRR, and first
  relevant physical-page rank across BM25, dense, and hybrid modes.
- Added fully offline tests using synthetic Evidence and a deterministic fake
  embedding provider; no production model was initialized or downloaded.

**Human decisions / review**
- Froze content-only BM25 (`k1=1.5`, `b=0.75`) and content-only dense retrieval;
  title and section metadata do not influence ranking.
- Froze RRF at `k=60`, component depth 50, with deterministic evidence-ID tie
  breaking and no tuning against the small gold set.
- Selected `Alibaba-NLP/gte-modernbert-base` as the default 8192-token local
  embedding model without `trust_remote_code`; its immutable revision remains
  intentionally unset pending a separately approved model-download step.
- Required document/query encoding semantics, float32 normalized vectors, and
  explicit failure rather than silent truncation beyond model context.

**Retrieval gold set**
- Manually verified six natural questions against the frozen physical PDFs,
  covering solar irradiance/cloud forecasting, solar reliability, wind NWP,
  wind atmospheric variability/wakes, climate impacts, and cross-modality
  complementarity.
- Gold relevance consists only of 1-based `(paper_id, page)` labels, allows
  multiple valid pages, and is checksum-bound to the finalized Phase 3
  `evidence.jsonl`. It includes no gold answers or manufactured conflicts.

**Scope boundary**
- No real embedding model initialization/download, real retrieval-index build,
  metric claim, LLM, agent, verifier, vision reasoning, ML, or literature
  expansion was performed.

**Validation**
- The complete offline repository suite contains 127 passing tests. Compilation,
  installed-package consistency, and Git whitespace checks pass.

---

### 2026-08-17 — Phase 4 production embedding-model setup

**Codex / AI-assisted work**
- Narrowed the approved NumPy dependency to `>=1.26,<2` after the macOS x86_64
  Torch 2.2.2 wheel reported that it was compiled against the NumPy 1.x ABI.
- Installed NumPy 1.26.4 and verified Torch-to-NumPy and NumPy-to-Torch
  conversion before model setup.
- Downloaded `Alibaba-NLP/gte-modernbert-base` with remote repository code
  disabled and safetensors required, then resolved and recorded immutable
  revision `e7f32e3c00f91d699e8c43b53106206bcc72bb22`.
- Reloaded that exact revision with `local_files_only=true` and forced offline
  Hugging Face/Transformers modes. One document and one query encoded as
  normalized float32 `(1, 768)` arrays.

**Verified runtime metadata**
- Python 3.11.15; NumPy 1.26.4; Torch 2.2.2; Transformers 4.57.6; Sentence
  Transformers 5.7.0.
- Model maximum sequence length 8192; embedding dimension 768.

**Scope boundary**
- No retrieval algorithm, gold annotation, chunk, corpus content, or embedding
  model choice changed. No real retrieval index was built and no retrieval
  evaluation was run.

---

### 2026-08-17 — Phase 5A bounded two-role runtime

**Codex / AI-assisted work**
- Added a lightweight custom Research Agent state machine and a separate,
  tool-free Evidence Verifier wrapper without introducing an agent framework or
  third runtime role.
- Reused `ResearchDraft`, `Claim`, `VerifierInput`, `VerificationResult`, all
  Phase 1 tool Protocols, and the complete `ExecutionTrace` tool/verifier
  contracts.
- Added typed Research Agent actions and arguments, compact base/session state,
  deterministic tool dispatch, structural draft validation, evidence caps, and
  explicit unresolved final outcomes.
- Extended execution traces to retain accepted/rejected actions and safe
  rationales while excluding provider messages, credentials, raw exceptions,
  and hidden reasoning.
- Added deterministic scripted-provider tests for retrieval, two-stage
  verification, all frozen verifier statuses, failures, budgets, routing,
  provenance separation, corpus immutability, and the real Phase 4 adapter over
  a synthetic offline index.

**Human decisions / review**
- Froze the exact flow `GATHERING -> DRAFT -> VERIFY #1 -> optional FOLLOW_UP ->
  VERIFY #2 -> COMPLETE`, with no path to a third verifier call.
- Reserved verifier calls for structurally valid drafts; unknown Evidence IDs
  are rejected before scientific verification.
- Froze Phase 5A defaults of six tool calls, ten Research Agent decisions, one
  literature search, two page inspections, one Python call, three follow-up
  tool calls, retrieval `k=5`, and base/session caps of 40/20.
- Kept `PaperRecord` and `AnalysisResult` distinct from citable Evidence and
  required page-inspection Evidence to retain physical page provenance in the
  active session scope.

**Scope boundary**
- No production LLM, multimodal model, runtime literature ingestion, real
  Python/ML analysis, retrieval change, chunking change, or network access was
  introduced.

---

### 2026-08-17 — Phase 5B production LLM and retrieval wiring

**Codex / AI-assisted work**
- Added an OpenAI Responses API adapter behind the existing `LLMProvider`
  Protocol, with separate stateless calls for typed action selection, structured
  drafting, and independent verification.
- Added strict provider-local Pydantic schemas for every `AgentAction` variant,
  `ResearchDraft`, claims, and complete `VerificationResult` findings, followed
  by conversion into the frozen Phase 5A dataclasses.
- Added explicit handling for refusals, incomplete or missing parsed output,
  schema failures, unsupported response types, unavailable page inspection, and
  credential-safe SDK failures with no retained exception chain.
- Added concise prompts that treat all paper Evidence as untrusted data, forbid
  invented provenance and causal overreach, and isolate verifier context from
  Research Agent state, rationales, traces, and provider history.
- Added production assembly for the existing checksummed Phase 4 index, frozen
  local GTE revision, MPS query embeddings, hybrid retrieval, and a registry in
  which retrieval is the only available tool.
- Added a minimal CLI whose default output is limited to answer, claim
  provenance, verifier status, and terminal status; the sanitized execution
  trace is opt-in.
- Added network-free mocked tests covering schemas, provider failures, secret
  safety, injection boundaries, role/model separation, context and budget
  exposure, provenance propagation, and bounded end-to-end orchestration.

**Human decisions / review**
- Froze `gpt-5.6-terra` for Research Agent calls and
  `gpt-4.1-2025-04-14` for the separate Evidence Verifier call.
- Froze zero SDK retries, a 60-second timeout, 600-character action Evidence
  previews, and a 200,000-character complete provider-context bound.
- Kept the Phase 5A state, action, draft, verification, tracing, budget, and
  two-verifier termination contracts authoritative and unchanged.
- Confirmed that runtime verifier PASS is a reliability signal rather than the
  final system-evaluation judgment.

**Scope boundary**
- No real OpenAI API call or scientific production question was run.
- No retrieval index build/tuning, corpus/chunk change, page vision, runtime
  literature ingestion, session indexing, Python/ML analysis, third role, UI,
  or final evaluation harness was introduced.

**Validation**
- The complete network-free suite contains 175 passing tests. The five warnings
  are the existing upstream PyMuPDF SWIG deprecation warnings under Python 3.11.
- Python compilation, installed-package consistency, and Git whitespace checks
  pass. No real OpenAI smoke test was run.

---

### 2026-08-17 — Phase 5B OpenAI action-schema compatibility correction

**Codex / AI-assisted work**
- Diagnosed the first production action call as an OpenAI API-side
  `invalid_json_schema` rejection of `properties.action.oneOf` in the
  provider-local discriminated-union schema.
- Replaced only the provider action wire format with one flat strict object;
  the frozen Phase 5A `AgentAction` union and all runtime semantics remain
  unchanged.
- Added deterministic action-specific conversion that requires applicable
  fields, rejects incompatible populated fields, parses Python request JSON
  only as an object, and performs no repair or enum guessing.
- Extended credential-safe provider errors with allowlisted OpenAI status,
  type, code, parameter, request ID, and sanitized short-message metadata while
  continuing to discard raw SDK exception chains.
- Added offline schema inspection and conversion/security regressions.

**Validation**
- The OpenAI strict action schema is a root object with all 13 properties
  required, `additionalProperties=false`, and no `oneOf` at any depth.
- The existing draft and verification schemas also contain no `oneOf` or
  arbitrary `additionalProperties`; no redesign was needed.
- The complete network-free suite contains 176 passing tests with the five
  existing upstream PyMuPDF SWIG deprecation warnings.

**Scope boundary**
- No production API call or smoke-test rerun was made after the correction.
- No prompt, model, domain contract, orchestrator, budget, retrieval, corpus,
  gold annotation, or verifier-architecture change was introduced.

---

### 2026-08-17 — Phase 5B production retrieval capability clarification

**Codex / AI-assisted work**
- Clarified the Research Agent tool contract that Phase 5B retrieval searches
  only the frozen base corpus, session Evidence retrieval is unavailable, and
  `include_session_evidence` must be false.
- Added a generic action-selection instruction to follow tool capability
  descriptions and argument restrictions without adding question-specific
  routing logic.
- Added offline regressions confirming the capability text reaches production
  action context and that a scripted true flag remains unchanged and is
  explicitly rejected by the existing Phase 4 adapter rather than coerced.

**Scope boundary**
- The frozen action field, domain models, adapter behavior, retrieval settings,
  models, budgets, verifier architecture, corpus, index, and gold annotations
  remain unchanged.
- No production API call was made after this correction.

**Validation**
- The complete network-free suite contains 177 passing tests with the five
  existing upstream PyMuPDF SWIG deprecation warnings.

---

### 2026-08-17 — Phase 5B incremental safe runtime observability

**Codex / AI-assisted work**
- Added an opt-in best-effort event callback and `--live-safe-trace` CLI flag
  that flush safe provider, action, tool, draft, and verifier milestones as
  soon as each operation completes.
- Restricted live events to operation/model/request/usage metadata, validated
  action fields, Evidence IDs with paper/page provenance, admission/duplicate
  counts, remaining tool budget, and aggregate verifier status.
- Added offline regressions for duplicate Evidence admission, partial-run
  preservation, callback isolation, provider-failure safety, and immediate CLI
  flushing.

**Scope boundary**
- `ExecutionTrace` remains the authoritative deterministic trace. No prompt,
  contract, state transition, budget, retrieval behavior, model, corpus,
  index, verifier behavior, or session-Evidence behavior changed.
- No OpenAI API or other network request was made.

---

### 2026-08-17 — Phase 5B production Test 1 smoke result

**Observed production result**
- Ran the approved numerical-weather-prediction and wind-power-forecasting
  question once through the frozen Phase 5B production runtime with live safe
  tracing enabled.
- The observed route was `retrieve (5 new) -> retrieve (3 new, 2 duplicate)
  -> retrieve (0 new, 4 duplicate) -> draft -> verifier PASS`.
- The runtime admitted 8 unique Evidence records, produced 6 grounded claims,
  and passed structural validation for every cited Evidence ID.
- The independent Evidence Verifier returned `PASS`; no unavailable tool was
  selected and no second verifier call was required.
- The run made 6 OpenAI API calls, used 22,167 total tokens, and completed in
  48.13 seconds of wall-clock time.

**Interpretation boundary**
- This is one production smoke-test observation, not a benchmark or an
  efficiency result.

---

### 2026-08-17 — Phase 5C offline canonical page inspection

**Codex / AI-assisted work**
- Added a checksummed resolver over the completed Phase 3 `pages.jsonl` map;
  page inspection accepts only a known paper ID and 1-based physical page and
  rejects missing, escaping, altered, or unknown image records before provider
  use.
- Added a bounded typed `PageInspectionResult` and one-image Structured Outputs
  path in the existing stateless OpenAI provider. The configured multimodal
  model is used when present, otherwise the configured Research model is used;
  there is no hard-coded additional model.
- Added `CanonicalPageInspectionTool`, which conservatively normalizes the
  question, validates provider provenance, and converts one result into exactly
  one deterministic session-scoped figure/table Evidence record. Frozen base
  Evidence and retrieval artifacts remain unchanged.
- Registered page inspection in the production factory and documented its
  canonical-page, no-path, no-OCR, and no-digitization capability bounds.
- Extended safe tool-result metadata with Evidence corpus scope and session
  admission counts while excluding paths, images, data URLs, prompts, and
  complete inspection content.

**Verifier boundary**
- Existing citation selection can pass cited derived visual Evidence text to
  the independent verifier. The verifier does not receive or independently
  inspect the raw page image, provider history, inspection prompt, or full
  Research state.

**Validation and scope**
- Offline tests use only fake Responses clients and synthetic tiny PNGs and
  cover canonical resolution, checksums, path containment, schema/result
  bounds, deterministic identity, production registration, budgets, duplicate
  admission, verifier isolation, and safe tracing.
- No API/network call or real multimodal smoke test was made. The frozen later
  primary page remains `paper_W2153263933`, physical page 4.

---

### 2026-08-17 — Phase 5C real multimodal smoke observations

**Provider boundary**
- The approved provider-only call resolved `paper_W2153263933`, physical page
  4, through the canonical checksummed page map and successfully completed one
  image-input Structured Output call with the configured Research model.
- The returned page interpretation passed the frozen paper/page/question and
  result-bound validation. This is a provider smoke observation, not a model
  benchmark.

**Distinct routing observations**
- Text-sufficient case: the NWP-processing question followed retrieval-only
  tool use, then drafting and verifier `PASS`; `inspect_page` was not selected
  because retrieved text directly described the GP correction pipeline.
- Visually dependent case: the Romania/Dubai clearness-index comparison followed
  `retrieve -> retrieve -> inspect_page(paper_W3003290234, physical page 3) ->
  draft -> verifier PASS`. The runtime admitted 10 unique base Evidence records
  and one session visual Evidence record, and the draft used mixed base-text and
  visual citations.
- Claim `c2` directly cited the session visual Evidence describing the relative
  distribution shapes, and the independent verifier returned `PASS` for that
  claim. No unavailable tool was selected and no provider or tool failure
  occurred.
- Checks of the frozen base-corpus manifest, Phase 3 ingestion manifest and
  Evidence artifact, and Phase 4 retrieval-index manifest were unchanged across
  the visually dependent run.

**Interpretation boundary**
- These are smoke/evaluation observations, not accuracy, performance, cost, or
  efficiency benchmarks. The verifier checked the derived visual Evidence text;
  it did not independently re-read the raw page image.

---

### 2026-08-17 — Phase 6 bounded Berlin scientific-analysis tool

**Codex / AI-assisted work**
- Inspected the ignored local Berlin CSV before implementation and confirmed
  36,296 rows, 24 columns, the exact `X50Hertz..MW.` target, chronological
  2018–2019 timestamps, no missing values, and no duplicate timestamps.
- Added one bounded `RUN_PYTHON` operation accepting only
  `{"analysis":"berlin_weather_solar_v1"}`. It accepts no Evidence, arbitrary
  code, paths, expressions, imports, shell commands, model parameters, tuning
  parameters, or extra request fields.
- Added strict schema, row-count, and SHA-256 validation; deterministic
  chronological splitting; train-only imputation/encoding/scaling; cyclic wind
  direction; and exactly a training-mean baseline, Ridge, and fixed histogram
  gradient boosting model.
- Added held-out MAE, RMSE, R², and raw-feature permutation importance through
  the complete fitted nonlinear pipeline. Computed values remain a non-Evidence
  `AnalysisResult` and preserve the tool call through `ResearchDraft.tool_trace`.
- Added synthetic, network-free tests for temporal leakage controls, exact
  features/models, bounded requests, reproducibility, provenance, safe missing
  data behavior, production registration, and capability text.

**Data availability**
- No tracked source, download, license, or preparation provenance was found for
  `Berlin_solar_regression.csv`; no public acquisition path was invented.
- The raw dataset remains ignored. A local run requires
  `data/ml/berlin/Berlin_solar_regression.csv` with SHA-256
  `eda6fccb75d8e76d9ae56e806e20fcb12f017041e02d463d60a94817ee5656d8`.

**Frozen real local analysis**
- Ran the approved analysis exactly once after the offline suite passed. Train
  was all 18,108 observations from 2018; test was all 18,188 observations from
  2019, with no shuffle, cross-validation, HPO, or post-result tuning.
- Held-out baseline: MAE 661.328 MW, RMSE 757.954 MW, R² -0.0251.
- Held-out Ridge: MAE 211.282 MW, RMSE 288.315 MW, R² 0.8517.
- Held-out histogram gradient boosting: MAE 190.356 MW, RMSE 278.180 MW,
  R² 0.8619.
- The target mean changed from 974.655 MW in training to 856.106 MW in test:
  -118.549 MW, or -12.163% relative to the training mean. No per-year target
  normalization was applied.
- Held-out nonlinear permutation importance using negative MAE was dominated by
  clear-sky GHI (378.844 MW), followed by observed GHI (36.961 MW), wind
  direction (20.757 MW), precipitable water (9.908 MW), and surface albedo
  (9.872 MW). Values are predictive for this table, not causal effects, and
  correlated irradiance/solar-geometry variables can redistribute importance.
- The analysis completed in 29.60 seconds with NumPy 1.26.4,
  scikit-learn 1.9.0, random seed 42, and five permutation repeats.

**Scientific interpretation boundary**
- This is contemporaneous prediction of regional 50Hertz solar generation from
  Berlin-area weather, observed irradiance, clear-sky/solar-geometry, and
  surface conditions. It is not future forecasting, a causal weather-impact
  estimate, or site-level Berlin PV prediction.
- The table is daylight-only, includes upstream-filled meteorological values,
  lacks locally documented timezone/alignment provenance, pairs a local weather
  representation with a regional target, and exhibits a material interannual
  target shift.

**Scope boundary**
- No OpenAI API call, retrieval/index change, provider-schema change, verifier
  change, page-inspection change, corpus/gold change, runtime-budget change, or
  processed dataset copy was introduced.

---

### 2026-08-17 — Phase 6B computed-result provenance integration

**Codex / AI-assisted work**
- Kept literature/page `Evidence` and computed `AnalysisResult` as distinct
  scientific-support classes. Claims can cite either class or both; computed
  results are not converted into synthetic Evidence.
- Added producer-owned deterministic IDs in the form
  `analysis:berlin_weather_solar_v1:<sha256>`. The hash covers the versioned
  analysis specification, dataset checksum and target, feature definitions and
  transforms, chronological split and row counts, preprocessing, exact model
  parameters, random seed, and permutation-importance specification. It
  excludes outputs, summaries, runtime, timestamps, request IDs, and software
  versions.
- Changed session-local computed-result admission to an ID-keyed mapping.
  Operational metadata changes are idempotent for an existing ID, while a
  conflicting scientific result under the same ID fails explicitly.
- Supplied bounded typed computed results and their producing call IDs to draft
  generation. Structural validation independently resolves Evidence and
  AnalysisResult references and requires the producing Python call in
  `tool_trace` whenever computed support is cited.
- Extended the verifier input with only claim-referenced AnalysisResults. The
  verifier checks faithful metric, split, ranking, predictive/causal, and
  limitation statements; this is claim/result consistency checking, not model
  retraining or computational reproduction.
- Extended the strict claim wire schema and safe observability with computed
  result IDs. CLI output distinguishes literature citations from computed
  result citations and does not project complete AnalysisResult values into
  the user-visible trace.

**Validation boundary**
- Phase 6B was implemented and tested offline only. No OpenAI API call or real
  Berlin analysis rerun was made, and the revised end-to-end production path
  is not yet claimed as smoke-tested.

---

### 2026-08-17 — Phase 6 production smoke observation

**Observed route**
- For the approved Berlin weather/solar question, the Research Agent naturally
  selected `RUN_PYTHON`, produced the deterministic
  `analysis:berlin_weather_solar_v1:291f89330918be0febc7596e46975bb1d1823f24aa257ecab392a6811cb61efc`
  result, and then selected `DRAFT_ANSWER`.
- The draft contained four affirmative computed claims. Each cited the
  AnalysisResult ID, cited no fake literature Evidence, and included the
  producing Python call in `tool_trace`.
- The independent verifier received only the claim-referenced bounded
  AnalysisResult and returned `PASS` for all four claims. It checked exact
  metric and ranking transcription plus the predictive/noncausal limitations;
  it did not rerun Python, retrain models, or independently reproduce results.

**Observed scientific result**
- Training-mean baseline: MAE 661.328 MW, RMSE 757.954 MW, R² -0.0251.
- Ridge: MAE 211.282 MW, RMSE 288.315 MW, R² 0.8517.
- HistGradientBoosting: MAE 190.356 MW, RMSE 278.180 MW, R² 0.8619.
- The target mean shifted from 974.655 MW in training to 856.106 MW in testing,
  a relative change of -12.163%.
- The leading held-out permutation-importance variables were `Clearsky.GHI`,
  `GHI`, `Wind.Direction`, `Precipitable.Water`, and `Surface.Albedo`. These are
  predictive/noncausal rankings, and correlated irradiance variables may
  redistribute importance.

**Integrity and interpretation boundary**
- The raw dataset checksum was unchanged, one-call Python budget was respected,
  and no unavailable tool, retrieval, page inspection, provider failure, or
  tool failure occurred.
- This is one production smoke/evaluation observation, not a benchmark or an
  independent reproduction of the scientific computation.
