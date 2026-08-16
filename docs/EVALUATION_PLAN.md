# Evaluation Plan

This document defines how the system will be evaluated.

The evaluation design is intentionally separated from implementation so that retrieval quality, tool selection, and scientific reliability can be tested independently.

The exact evaluation questions and gold evidence must be finalized only after the automatically collected paper corpus has been inspected.

---

## Evaluation goals

The evaluation should answer three core questions:

1. Did the agent choose the right research action?
2. Did the system retrieve the right scientific evidence?
3. Did the final claim actually follow from that evidence?

These correspond to:

- agentic behavior
- retrieval / RAG quality
- scientific reliability

---

## Evaluation question taxonomy

The evaluation harness should contain approximately 12 questions.

The taxonomy is frozen as follows:

- 4 text-RAG questions
- 1 literature-expansion question
- 2 multimodal figure/table questions
- 1 ML/data-analysis question
- 1 mixed literature + ML question
- 1 missing-evidence question
- 2 conflicting-evidence questions

Total:

12 questions

The exact wording may change after corpus inspection, but the category distribution should remain approximately stable.

---

## Gold-label creation

Gold labels must be created after the automatic literature collection step.

Do not manually curate the paper corpus first and then claim that collection was automated.

Recommended process:

1. run automated literature search
2. automatically collect approximately 8-12 papers
3. inspect the resulting corpus manually
4. identify scientifically meaningful test questions
5. identify the correct supporting paper(s)
6. identify supporting page(s)
7. identify figure/table references where relevant
8. write concise expected-answer notes
9. define expected tool behavior
10. save gold labels in machine-readable form

The purpose of manual inspection is evaluation annotation, not replacement of automated discovery.

---

## Avoid circular evaluation

Do not rely on one LLM to:

- generate all questions
- generate the gold answers
- answer the questions
- judge its own answers

LLMs may assist with candidate question generation, but gold evidence must be manually verified against the source papers.

LLM-as-judge may be used only as a supplementary evaluation signal.

Primary evidence correctness should remain grounded in manually verified paper/page labels.

---

## Suggested question schema

Suggested file:

`evaluation/questions.json`

Each question may contain fields such as:

```json
{
  "id": "q01",
  "question": "How does cloud cover affect photovoltaic forecasting accuracy?",
  "category": "text_rag",
  "gold_papers": ["paper_03"],
  "gold_pages": [5, 6],
  "gold_sources": ["Section 4.2"],
  "expected_tools": ["retrieve_evidence"],
  "expected_answer_notes": "The paper reports lower forecasting accuracy under higher cloud variability.",
  "requires_uncertainty": false
}
```

For multimodal questions:

```json
{
  "id": "q06",
  "question": "Which weather condition shows the largest forecasting error in the reported figure?",
  "category": "multimodal",
  "gold_papers": ["paper_07"],
  "gold_pages": [8],
  "gold_sources": ["Figure 4"],
  "expected_tools": ["retrieve_evidence", "inspect_page"],
  "expected_answer_notes": "Answer should identify the condition shown as highest in Figure 4.",
  "requires_uncertainty": false
}
```

For missing-evidence questions:

```json
{
  "id": "q10",
  "question": "Does the literature prove that cloud cover always reduces solar generation by at least 30%?",
  "category": "missing_evidence",
  "gold_papers": [],
  "gold_pages": [],
  "gold_sources": [],
  "expected_tools": ["retrieve_evidence"],
  "expected_answer_notes": "The corpus does not justify this universal quantitative claim.",
  "requires_uncertainty": true
}
```

---

# Layer 1: Agent behavior evaluation

## Goal

Evaluate whether the Research Agent chooses appropriate tools rather than following a fixed sequence.

## What to record

For every evaluation question, record:

- question ID
- expected tool categories
- actual tool calls
- tool-call order
- number of calls
- whether unnecessary tools were used
- whether required tools were omitted

Suggested execution trace:

```json
{
  "question_id": "q06",
  "tool_trace": [
    "retrieve_evidence",
    "inspect_page"
  ]
}
```

## Example routing expectations

### Text-RAG question

Expected:

`retrieve_evidence`

Usually not expected:

- `inspect_page`
- `run_ml_analysis`
- new literature search

### Literature-expansion question

Expected:

- initial retrieval or evidence assessment
- `search_literature`
- retrieval from newly available evidence

### Multimodal question

Expected:

- retrieval / page localization
- `inspect_page`

### ML question

Expected:

`run_ml_analysis`

### Mixed question

Expected:

- literature retrieval
- ML analysis
- synthesis

### Missing evidence

Expected:

- retrieval
- possibly search
- explicit insufficiency

### Conflicting evidence

Expected:

- retrieval from multiple papers
- verifier conflict detection
- possibly additional search

---

## Routing metrics

A simple MVP metric can use expected tool categories versus actual tool categories.

Possible metrics:

- required-tool recall
- unnecessary-tool rate
- exact routing match

These do not need to be overly sophisticated.

The main goal is to demonstrate that tool choice is measurable.

---

# Layer 2: Evidence retrieval evaluation

## Goal

Test whether relevant evidence is retrieved.

## Primary metrics

Use:

- Recall@3
- Recall@5

For each question with gold textual evidence:

Recall@k = 1 if at least one gold evidence item appears in the top-k results, otherwise 0.

Average across eligible questions.

Depending on annotation quality, evaluation may be done at:

- paper level
- page level
- chunk level

Page-level evaluation is preferable where practical because the system preserves page provenance.

---

## Citation validity

For every citation used in a final answer, check:

- paper ID exists
- page exists
- evidence object exists
- cited content is relevant to the associated claim

Citation formatting alone does not count as citation correctness.

---

## Optional retrieval ablation

If time permits, compare:

1. dense-only retrieval
2. hybrid BM25 + dense retrieval

Use the same gold questions.

Report actual measured differences only.

Do not assume hybrid retrieval is better before running the evaluation.

---

## Retrieval error analysis

For failed questions, classify the likely cause where possible:

- relevant paper was never collected
- relevant content was not extracted
- chunk boundary problem
- lexical mismatch
- semantic retrieval ranking error
- figure/table evidence not represented textually
- metadata/provenance mismatch

A small error-analysis table is more informative than a single aggregate metric.

---

# Layer 3: Scientific reliability evaluation

## Goal

Determine whether final claims are justified by the evidence.

This is the main role of the Evidence Verifier.

## Reliability dimensions

Evaluate:

- claim support
- citation support
- missing-evidence handling
- conflicting-evidence handling
- uncertainty expression
- overclaim prevention
- unsupported numerical precision

---

## Suggested claim-level annotation

For each final answer, important claims may be marked as:

- SUPPORTED
- PARTIALLY_SUPPORTED
- UNSUPPORTED
- CONFLICTING
- NOT_ENOUGH_EVIDENCE

The verifier runtime statuses remain:

- PASS
- NEED_MORE_EVIDENCE
- UNSUPPORTED_CLAIM
- CONFLICTING_EVIDENCE

---

## Missing-evidence evaluation

Include at least 1 question where the requested conclusion cannot be justified from the corpus.

Success criteria:

- no fabricated paper
- no fabricated citation
- no unsupported exact number
- explicit statement that evidence is insufficient
- appropriate verifier status

Failure examples:

- confidently answering anyway
- inventing universal claims from limited studies
- extrapolating beyond the available evidence

---

## Conflicting-evidence evaluation

Include 2 questions where papers disagree if suitable cases can be found in the automatically collected corpus.

Success criteria:

- identify that studies disagree
- preserve separate evidence sources
- avoid collapsing disagreement into one definitive conclusion
- mention methodological/context differences only when evidence supports them
- express residual uncertainty

Possible conflict sources:

- different geographic regions
- different forecast horizons
- different datasets
- different model classes
- different weather regimes
- different evaluation metrics

Do not manufacture a conflict if the corpus does not contain a real one.

If necessary, adapt the exact questions while preserving the conflict category.

---

# Multimodal evaluation

## Goal

Evaluate both visual evidence extraction and correct visual-tool routing.

Include approximately 2 multimodal questions.

Possible types:

- interpret a figure trend
- identify the highest/lowest value in a table
- compare conditions shown visually
- read a qualitative result that is not fully stated in surrounding text

## Gold labels

For each multimodal question, record:

- paper
- page
- figure/table identifier
- expected qualitative conclusion
- expected numerical conclusion only if clearly readable

## Success criteria

- correct page selected
- `inspect_page` called when needed
- answer consistent with visual evidence
- no invented precision when the image is unclear

Also test some text-only questions to confirm that vision is not always invoked.

---

# ML / scientific analysis evaluation

## Goal

Evaluate whether the system can perform a small scientifically valid ML analysis.

## Required checks

Verify:

- correct feature/target definition
- chronological split
- no test-set leakage
- preprocessing fitted only on training data
- baseline included
- linear model included
- nonlinear model included
- MAE reported
- RMSE reported
- R2 reported
- permutation feature importance reported
- interpretation avoids causal overstatement

## Result interpretation

A successful answer should use wording such as:

"In this dataset..."

or:

"The model suggests predictive importance..."

Avoid:

"This proves that variable X causes..."

The ML component is empirical and dataset-specific.

---

# Mixed literature + ML evaluation

Include at least 1 question that requires both literature evidence and empirical ML analysis.

Example pattern:

"Which weather variables are emphasized in the literature, and does the small empirical model show a similar predictive pattern?"

Expected behavior:

1. retrieve literature evidence
2. run ML analysis
3. compare findings
4. distinguish literature-wide evidence from dataset-specific empirical evidence
5. avoid claiming agreement if results differ

This question is especially useful for demonstrating adaptive multi-tool reasoning.

---

# Verifier evaluation

## Goal

Demonstrate that the second agent performs an independent reasoning responsibility.

Include at least one example where:

1. Research Agent creates a draft
2. Verifier rejects or requests more evidence
3. Research Agent gathers additional evidence
4. Verifier checks again
5. final answer is revised

Maximum verifier rounds:

2

The verifier should inspect claim-evidence alignment rather than simply produce another answer.

---

# Tool failure evaluation

Include at least one simple failure scenario if practical.

Possible cases:

- PDF unavailable
- malformed PDF
- unreadable visual
- missing dataset field
- weak search results

Success means:

- system does not crash
- failure is logged
- fallback is attempted where reasonable
- final answer exposes limitations when unresolved

---

# Execution metrics

For each example, optionally record:

- number of LLM calls
- number of tool calls
- number of retrieval calls
- number of vision calls
- verifier rounds
- latency

These are secondary metrics.

They are useful to show that adaptive reasoning does not result in uncontrolled tool usage.

---

# Evaluation result storage

Suggested output:

`results/evaluation.json`

Possible structure:

```json
{
  "retrieval": {
    "recall_at_3": 0.0,
    "recall_at_5": 0.0
  },
  "routing": {
    "required_tool_recall": 0.0,
    "unnecessary_tool_rate": 0.0
  },
  "reliability": {
    "supported_claim_rate": 0.0,
    "missing_evidence_success": false,
    "conflict_detection_success": false
  }
}
```

These values are placeholders only.

Do not commit fabricated final evaluation numbers.

Populate metrics only after actual evaluation runs.

---

# Human review

Before reporting final results:

1. inspect all 12 questions
2. verify gold paper/page labels
3. inspect multimodal gold evidence
4. review missing-evidence cases
5. review conflicting-evidence cases
6. inspect suspicious metric results
7. manually review example final answers

The final presentation should report only results that were actually measured.

---

# Presentation-oriented evaluation summary

The full evaluation can be summarized in the interview through three questions:

1. Did the agent choose the right tool?
2. Did it retrieve the right evidence?
3. Did its conclusion follow from that evidence?

Example presentation metrics may include:

- tool-routing correctness
- Recall@3 / Recall@5
- citation validity
- multimodal question success
- missing-evidence behavior
- conflicting-evidence behavior

Use actual measured values only.

---

# Evaluation limitations

The evaluation itself has limitations.

Examples:

- only approximately 12 questions
- manually annotated gold evidence
- small automatically collected corpus
- domain-limited questions
- LLM stochasticity
- multimodal interpretation uncertainty
- incomplete coverage of systematic-review behavior

These limitations should be stated explicitly rather than hidden.

---

# Final evaluation principle

The goal is not to maximize one benchmark score.

The goal is to demonstrate that the system behaves like a trustworthy scientific research assistant:

- it chooses useful research actions
- it retrieves traceable evidence
- it checks whether claims are supported
- it identifies missing or conflicting evidence
- it exposes uncertainty instead of fabricating certainty
