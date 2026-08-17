# Evaluation Plan

This document defines the frozen Gate 4A system-evaluation design. Evaluation
asks three separate questions:

1. Did the Research Agent choose a scientifically useful action?
2. Did retrieval find manually verified paper/page evidence?
3. Did each final claim resolve to the Evidence or AnalysisResult it cites?

The suite is deliberately small and diagnostic. It is not a benchmark, and its
execution classes must not be pooled into one success rate.

## Corpus and annotation boundary

The cases were defined only after inspecting the automatically discovered
10-paper frozen base corpus. Gold literature relevance is primarily annotated
as paper_id plus 1-based physical PDF page. More than one page may be valid.
The Berlin computed case instead uses the frozen deterministic AnalysisResult
ID.

Manual annotation is used for evaluation, not corpus selection. The suite does
not manufacture scientific disagreement, balance, or answer coverage. In
particular, contextual heterogeneity is not automatically conflicting evidence.

The versioned source of truth is evaluation/system_cases.json. It records the
source Evidence artifact checksum and contains scientific expectations, not
generated answers, request IDs, provider payloads, or credentials.

## Frozen Gate 4A cases

| ID | Execution class | Scientific purpose | Expected useful tools |
|---|---|---|---|
| T01 | offline | Solar irradiance/cloud forecasting retrieval | retrieve_evidence |
| T02 | offline | Wind atmospheric variability, turbulence, and wakes retrieval | retrieve_evidence |
| T03 | offline | Wind-solar complementarity retrieval | retrieve_evidence |
| M01 | historical | Text-sufficient NWP processing control | retrieve_evidence |
| M02 | historical | Visually dependent Romania/Dubai comparison | retrieve_evidence, inspect_page |
| A01 | historical | Berlin computed analysis grounding | run_python |
| S01 | new_live | Broad multi-paper reliability synthesis | retrieve_evidence |
| O01 | new_live | Out-of-corpus insufficiency | direct cautious response or retrieve_evidence |
| I01 | new_live | Unsupported universal-method claim | retrieve_evidence |
| X01 | new_live | Mixed literature and Berlin computation | retrieve_evidence, run_python |

This 10-case design replaces the earlier provisional 12-question taxonomy.
There is no mandatory literature-expansion case because live expansion is
unavailable in the frozen production runtime. There is no conflict quota:
real conflicts must not be fabricated merely to satisfy a category count.

### Important case semantics

- O01 permits either a direct cautious zero-claim answer or retrieval followed
  by a cautious insufficiency answer. Unrelated renewable Evidence must not be
  used to make coral-science claims, and corpus silence is not negative
  scientific evidence.
- I01 evaluates insufficiency for a universal forecasting-method claim.
  Variation across datasets, horizons, locations, seasons, and metrics is not
  itself CONFLICTING_EVIDENCE.
- M02 requires retrieval before page inspection so that the page is selected
  through corpus evidence rather than supplied as an answer hint.
- X01 is the highest-value adaptive case: it should keep published Evidence and
  the local Berlin AnalysisResult distinct while comparing them cautiously.

## Execution classes and denominators

### Offline: T01–T03

These cases exercise deterministic Phase 4 retrieval only. They do not invoke
an LLM, Research Agent, verifier, vision model, or Berlin analysis and do not
make live answer-quality claims.

Report page-level metrics for these three cases as their own denominator:

- Hit@3
- Hit@5
- Page Recall@3
- Page Recall@5
- reciprocal rank of the first relevant page (MRR when aggregated)

### Historical: M01, M02, A01

These cases preserve already documented production observations:

- M01: retrieval-only text-sufficient route, verifier PASS.
- M02: retrieval then page inspection, mixed base and visual support,
  verifier PASS.
- A01: bounded Berlin analysis, first-class computed support, verifier PASS.

Historical records may contain only facts already documented at the time of
the original run. Unrecorded call, token, time, or retrieval metrics remain
null; they must not be reconstructed from memory or inferred from present
code. These three cases form a separate historical denominator and are not
rerun by Gate 4A implementation work.

### New live: S01, O01, I01, X01

Each new live case must be explicitly selected by case_id. There is no implicit
run-all operation. Running one can use the frozen production runtime and
consume API calls, so live execution requires separate approval.

The harness must never silently execute all new-live cases, rerun historical
cases, or run the Berlin analysis as part of validation.

## Layer 1: routing evaluation

Compare attempted tool categories with each case's expected, optional, and
unavailable tools. Report one deterministic label:

- appropriate
- missed_useful_tool
- unnecessary_tool_use
- unavailable_tool_attempt

Unavailable-tool attempts take precedence. Missing expected tools and violated
prerequisites are next, followed by unnecessary tools. Repeated calls to an
otherwise allowed tool are not automatically wrong; duplicate Evidence and
bounded tool-use metadata should be inspected separately.

For O01, both no tool call and a retrieval call are appropriate. For M02,
inspect_page before retrieve_evidence is a missed-useful-tool/prerequisite
failure even if both tools eventually appear.

## Layer 2: retrieval evaluation

Gold relevance is page-level: paper_id plus 1-based physical PDF page.
Multiple chunks from one page count as one relevant page. Primary retrieval
metrics are:

- Hit@3 and Hit@5: whether any gold page occurs within the cutoff;
- Page Recall@3 and Page Recall@5: unique retrieved gold pages divided by all
  annotated gold pages;
- MRR: reciprocal rank of the first relevant page.

T01–T03 are evaluated with the frozen hybrid retrieval settings. No answer is
generated and no verifier status is imputed. The approved final aggregation
regenerated the six-query Phase 4 BM25/dense/hybrid measurements once and ran
T01–T03 deterministically without a Research Agent. The safe measured summary
is tracked in `evaluation/system_results_summary.json`; retrieval parameters,
gold labels, and the canonical index were unchanged.

## Layer 3: structural provenance and scientific reliability

For a completed live outcome, validate at least:

- every affirmative claim has an Evidence ID, AnalysisResult ID, or both;
- every cited Evidence ID resolves to admitted Evidence;
- every cited AnalysisResult ID resolves to an admitted AnalysisResult;
- cited Evidence retains paper, title, source, and 1-based page provenance;
- cited visual Evidence is session-scoped and distinct from base Evidence;
- AnalysisResult objects remain distinct from literature/page Evidence;
- every cited computed result has a producing Python ToolCall;
- the producing call appears in ResearchDraft.tool_trace;
- every draft tool-trace ID and stored ToolResult call ID resolves;
- no unavailable tool was dispatched;
- no support ID was invented.

These checks establish structural grounding. They do not by themselves prove
that a scientific interpretation is correct.

The independent verifier status remains one of PASS, NEED_MORE_EVIDENCE,
UNSUPPORTED_CLAIM, or CONFLICTING_EVIDENCE. Verifier behavior and manual
scientific review are not interchangeable. Verifier PASS must not be described
as model reproduction, raw-image reinspection, or a human reliability label.

## Manual reliability review

After a new live case has a safe persisted record, a human reviewer may assign
exactly one of:

- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED
- CONFLICTING
- INSUFFICIENT_EVIDENCE

The field defaults to null. It is never automatically derived from verifier
status, routing, citations, or retrieval metrics.

Review should assess:

- whether cited material actually supports the associated claim;
- whether qualifications, scope, and uncertainty are preserved;
- whether predictive associations are incorrectly presented as causal;
- whether exact values or cross-study equivalence are overstated;
- whether an insufficiency response is appropriate rather than evasive.

## Multimodal evaluation

M01 and M02 form a paired routing observation:

- M01 tests that page inspection is not used when text is sufficient.
- M02 tests natural selection of page inspection for a visual comparison.

For visual support, record the canonical paper and physical page, derived
session Evidence ID, modality, and session scope. The verifier checks the
derived visual Evidence text; it does not independently re-read the raw image.
Do not claim that one successful page inspection proves vision is generally
necessary or accurate.

## Computed and mixed evaluation

A01 checks that the Berlin task is contemporaneous prediction with a
chronological 2018/2019 split, bounded fixed models, predictive/noncausal
interpretation, and first-class computed-result provenance.

X01 requires literature retrieval and the frozen Berlin analysis. Literature
claims must cite Evidence; local computed claims must cite the AnalysisResult
and its producing tool call. Similarity between the literature and local
feature rankings must be described cautiously, without claiming causal
confirmation, future forecasting, or exact equivalence.

## Safe result storage

Generated Gate 4A records may be written only beneath
data/cache/system_evaluation/. That directory is an ignored derived cache. The
writer refuses silent overwrite and rejects output paths outside the cache
root.

Safe persisted records may include:

- case ID, question, and execution class;
- typed action category, user-safe reason, and sanitized arguments;
- tool names and call IDs;
- final answer, claim text, and cited support IDs;
- cited Evidence provenance without Evidence content;
- AnalysisResult IDs without full result blobs;
- verifier status and bounded findings;
- structural provenance booleans;
- model identifiers, safe token counts, call counts, and wall time.

They must exclude:

- API keys and Authorization headers;
- prompts and raw request/response bodies;
- request IDs;
- complete Evidence content;
- complete AnalysisResult values;
- image bytes or data URLs;
- hidden reasoning;
- unsafe exception chains.

## Reporting rules

Always report offline, historical, and new-live outcomes separately. Never
combine them into a single accuracy denominator.

Report only measured or directly documented values. Missing historical metrics
remain null; unexecuted live cases have no outcome. Do not interpret the small
suite as a benchmark, optimize prompts or retrieval against it, or edit gold
pages after observing outcomes.

## Limitations

- The frozen corpus contains only 10 automatically selected papers.
- Gold pages and live-answer reliability require manual judgment.
- Three offline retrieval cases cannot estimate broad-domain retrieval quality.
- Historical observations were not captured by the Gate 4A harness and have
  intentionally incomplete metadata.
- Four new-live cases are stochastic and potentially costly.
- The suite contains an insufficiency case but no manufactured conflict case.
- Visual verification operates on derived page-inspection text.
- Computed-result verification checks consistency, not reproduction.

The goal is a transparent diagnostic of action choice, evidence recovery, and
claim grounding—not maximization of one aggregate score.

## Final Gate 4A observations

### Frozen six-query Phase 4 retrieval

| Mode | Hit@3 | Hit@5 | Page Recall@3 | Page Recall@5 | MRR | First relevant ranks (q01–q06) |
|---|---:|---:|---:|---:|---:|---|
| BM25 | 0.833 | 0.833 | 0.264 | 0.375 | 0.611 | —, 3, 1, 1, 3, 1 |
| Dense | 0.500 | 0.500 | 0.222 | 0.222 | 0.417 | —, 2, 1, —, 1, — |
| Hybrid RRF | 0.667 | 0.667 | 0.208 | 0.264 | 0.583 | —, 2, 1, —, 1, 1 |

BM25 outperformed hybrid on this small frozen gold set. This is reported as a
diagnostic result; no retrieval setting was tuned after measurement.

### Offline Gate 4A retrieval cases

| Case | Gold pages | Top-five retrieved physical pages | Hit@3 / Hit@5 | Recall@3 / Recall@5 | First relevant rank | Provenance |
|---|---|---|---|---|---:|---|
| T01 | W3089230449:4,5,6 | W3089230449:3,1; W3126094341:7; W3089230449:3; W3126094341:5 | 0 / 0 | 0 / 0 | — | valid |
| T02 | W2974473399:16,19,25 | W2974473399:31,45,53,44,54 | 0 / 0 | 0 / 0 | — | valid |
| T03 | W2933024154:3,13,14,18 | W2933024154:13,17,30,15,19 | 1 / 1 | 0.25 / 0.25 | 1 | valid |

T01 and T02 retrieved the correct papers but missed the annotated pages within
the top five. T03 retrieved a gold page at rank 1. These are retrieval-layer
diagnostics, not live answer evaluations.

### Seven real end-to-end observations

| Case | Type | Expected → actual tools | Support | Verifier | Manual reliability | Outcome |
|---|---|---|---|---|---|---|
| M01 | historical | retrieve → retrieve | text | PASS | — | text-sufficient routing observed; terminal unrecorded |
| M02 | historical | retrieve + inspect → retrieve + inspect | text + visual | PASS | — | visual routing observed; terminal unrecorded |
| A01 | historical | Python → Python | computed | PASS | — | terminal pass |
| X01 | new live | retrieve + Python → retrieve + Python | text + computed | PASS | SUPPORTED | scoped mixed-provenance answer |
| O01 | new live | cautious/retrieve → retrieve | none cited | PASS | INSUFFICIENT_EVIDENCE | safe refusal after six retrievals |
| S01 | new live | retrieve → retrieve | text | PASS | PARTIALLY_SUPPORTED | grounded claims, incomplete wind coverage |
| I01 | new live | retrieve → retrieve | text | PASS | SUPPORTED | rejected a universal best method |

Documented verifier outcomes were PASS in 7/7 observations. Terminal outcome
was recorded for five cases and was `pass` in 5/5; M01 and M02 are excluded
from that denominator because their historical terminal status was not
recorded. The four new-live cases were 4/4 routing-appropriate and 4/4 passed
automatic structural-provenance checks; none required verifier #2 or attempted
an unavailable tool. Historical manual labels remain null because verifier
PASS alone is not a defensible human annotation.

The new-live manual-label distribution is 2 SUPPORTED, 1 PARTIALLY_SUPPORTED,
and 1 INSUFFICIENT_EVIDENCE. O01 is a successful robustness observation: the
system refused to turn unrelated renewable-energy evidence into a coral-science
answer, although it used six retrieval calls before stopping. S01 preserves the
distinction between verifier PASS and partial human reliability: its claims
were grounded, but wind-specific coverage was weak, only two papers supported
the synthesis, and retrieval calls 3–5 did not add final cited support. X01
kept literature Evidence and the local AnalysisResult distinct and retained
regional, contemporaneous, and noncausal scope; it did not explicitly enumerate
every underlying AnalysisResult limitation. I01 rejected a universal winner
and treated its within-study result without cross-context overgeneralization.

Across X01, O01, S01, and I01, the single-run operational totals were 30 API
calls, 18 tool calls, 135,263 input tokens, 10,018 output tokens, 145,281 total
tokens, and 168.152 seconds of runtime wall time. These are observations from
one run per case, not latency, cost, or efficiency benchmarks.
