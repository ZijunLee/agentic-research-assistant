# Requirements -> Architecture -> Evaluation Mapping

This document maps the take-home requirements and interview evaluation criteria to concrete system components and tests.

The purpose is to ensure that every implemented component has a clear reason to exist and a clear way to demonstrate that it works.

---

## 1. Automated literature discovery and collection

### Requirement

Find and collect relevant scientific papers on the topic.

This step should be automated as much as possible.

### Architecture

The Research Agent has access to a literature-search tool.

The corpus must not be manually curated.

### Implementation

Suggested flow:

scientific topic
-> query generation / expansion
-> OpenAlex search
-> approximately 30-50 candidate papers
-> relevance filtering / ranking
-> DOI or normalized-title deduplication
-> open-access filtering
-> automatic collection of approximately 8-12 papers
-> PDF download
-> corpus metadata persistence

Suggested tool:

`search_literature(query)`

### Evaluation / demonstration

Record:

- number of candidate papers returned
- number selected
- number removed as duplicates
- number with usable full text
- download failures
- selected-paper metadata

Manually inspect a small sample for scientific relevance.

The evaluation corpus should be the automatically collected corpus, not a manually selected replacement corpus.

---

## 2. Process and index papers

### Requirement

Process and index the papers, including text and, where useful, figures and tables.

### Architecture

Create a multimodal evidence store linked by paper/page provenance.

### Implementation

For each PDF:

PDF
-> page text
-> section/page-aware chunks
-> metadata
-> rendered page images
-> retrieval index

Suggested chunk size:

approximately 500-800 tokens with overlap.

Each evidence item should preserve:

- paper_id
- title
- page
- section if available
- modality
- source identifier
- content

Supported modalities:

- text
- figure
- table

### Evaluation / demonstration

Check sampled evidence objects manually.

Verify that:

- paper metadata is correct
- page numbers are correct
- retrieved chunks can be traced to the original PDF
- rendered page images correspond to the same page metadata

Use retrieval gold labels later to test whether relevant evidence is indexed correctly.

---

## 3. Evidence-grounded scientific question answering

### Requirement

Build an agent that can answer questions using evidence from the collected papers.

### Architecture

The Research Agent owns evidence acquisition and synthesis.

It should produce structured claims and evidence rather than only free-form prose.

### Implementation

Suggested Research Agent output:

- draft_answer
- claims
- evidence_ids
- uncertainty
- tool_trace

Each major factual claim should be linked to one or more evidence objects.

### Evaluation / demonstration

For manually labelled evaluation questions:

- inspect whether the correct paper/page was retrieved
- inspect whether claims are supported by cited evidence
- inspect whether uncertainty is appropriate

Do not evaluate only surface answer similarity.

---

## 4. Literature-search tool

### Requirement

Give the agent useful tools, including literature search.

### Architecture

Literature search is a tool used by the Research Agent.

It is not a separate agent in the MVP.

### Implementation

Suggested function:

`search_literature(query)`

Possible backend:

OpenAlex

Possible behavior:

- search by query
- return scientific metadata
- prefer relevant and open-access papers
- expose enough metadata for downstream ranking and download

### Evaluation / demonstration

Include at least one evaluation question where the current indexed corpus is insufficient and additional literature search is expected.

Check whether the Research Agent invokes the search tool appropriately.

---

## 5. Paper retrieval / RAG

### Requirement

Give the agent a paper retrieval / RAG tool.

### Architecture

Use hybrid retrieval exposed as a Research Agent tool.

### Implementation

Suggested tool:

`retrieve_evidence(query, k)`

Preferred retrieval:

- BM25 or lexical retrieval
- dense semantic embeddings
- simple rank fusion or weighted combination

Return evidence objects with provenance metadata.

### Evaluation / demonstration

Primary metrics:

- Recall@3
- Recall@5

Optional small ablation:

- dense-only retrieval
- hybrid retrieval

The purpose of the ablation is to test retrieval design rather than assume that hybrid retrieval is better.

---

## 6. Figure / image / table inspection

### Requirement

Give the agent a figure or image inspection capability and process figures/tables where useful.

### Architecture

Treat figures and tables as first-class scientific evidence.

Use page-level multimodal inspection rather than a complex figure-detection pipeline.

### Implementation

Preferred flow:

text retrieval
-> identify relevant paper/page
-> detect that textual evidence is incomplete
-> render page as image
-> inspect page with multimodal model
-> return structured visual evidence

Suggested tool:

`inspect_page(paper_id, page, question)`

Principle:

Text retrieves the page; vision interprets the page.

### Evaluation / demonstration

Include at least 2 multimodal evaluation questions.

For those questions, evaluate:

- whether the Research Agent chose the visual inspection tool
- whether it identified the correct page
- whether the visual answer matches the figure/table
- whether it avoids reporting unreadable exact numbers

Also include text-only questions to test whether the agent avoids unnecessary vision calls.

---

## 7. Python / scientific analysis

### Requirement

Give the agent a Python / data-analysis capability.

### Architecture

Python analysis remains a tool under the Research Agent.

### Implementation

Suggested use cases:

- compare numerical values
- compute summary statistics
- generate a small plot
- compare model metrics
- analyze the scientific ML dataset

Suggested tool:

`run_python_analysis(...)`

Cross-paper numerical comparison should first check comparability.

Important comparability dimensions include:

- units
- dataset
- forecast horizon
- normalization
- metric definition
- evaluation protocol

### Evaluation / demonstration

Include at least one quantitative question.

Preferably include:

- one valid comparison case
- one case where direct aggregation would be scientifically invalid

A correct refusal to compare incompatible results counts as a successful scientific behavior.

---

## 8. Small machine-learning component

### Requirement

Include at least one small machine-learning or scientific data-analysis component.

The interview email also explicitly states interest in machine-learning fundamentals.

### Architecture

Implement a real supervised regression experiment rather than relying only on embeddings.

### Scientific case study

Weather variables -> solar generation prediction

### Implementation

Minimum experiment:

1. define features and target
2. perform data cleaning
3. use chronological train/test split
4. prevent temporal leakage
5. implement a simple baseline
6. train Ridge regression
7. train Random Forest or Gradient Boosting
8. evaluate on held-out data
9. compute MAE
10. compute RMSE
11. compute R2
12. compute permutation feature importance

Any learned preprocessing must be fitted only on training data.

Feature importance must be described as predictive, not causal.

### Evaluation / demonstration

Report:

- baseline performance
- linear model performance
- nonlinear model performance
- train/test performance where useful
- held-out MAE
- held-out RMSE
- held-out R2
- feature importance

Demonstrate knowledge of:

- regression formulation
- data leakage
- train/test splitting
- baselines
- generalization
- overfitting
- metric choice
- interpretation limits

---

## 9. Simple evaluation harness

### Requirement

Build a simple evaluation harness to test whether the system retrieves the right evidence and gives reliable answers.

### Architecture

Evaluation is a separate reproducible subsystem.

Suggested files:

- `evaluation/questions.json`
- `evaluation/evaluate_retrieval.py`
- `evaluation/evaluate_routing.py`
- `evaluation/evaluate_reliability.py`

### Evaluation philosophy

Use three layers:

1. Did the agent choose the right research action?
2. Did it retrieve the right scientific evidence?
3. Did the final claim actually follow from that evidence?

### Evaluation / demonstration

Save machine-readable results, for example:

`results/evaluation.json`

Do not fabricate metrics.

Gold labels should be manually confirmed after the automatically collected corpus has been inspected.

---

## 10. Adaptive tool selection rather than a fixed pipeline

### Requirement

The agent should decide which tools it needs rather than following one fixed pipeline.

### Architecture

The Research Agent follows an adaptive loop:

reason about evidence need
-> choose tool
-> observe result
-> reassess evidence
-> choose another tool or stop

### Implementation

Possible trajectories:

Text question:

retrieve
-> answer

Figure question:

retrieve
-> inspect_page
-> answer

Data question:

run_ml_analysis
-> answer

Mixed literature/data question:

retrieve
-> run_ml_analysis
-> synthesize

Insufficient-evidence question:

retrieve
-> search_literature
-> retrieve
-> verifier

Conflict question:

retrieve multiple papers
-> compare
-> verifier
-> possibly search more evidence
-> uncertainty-aware answer

### Evaluation / demonstration

For each evaluation question, define an expected tool set or expected tool category.

Compare:

expected tools
vs.
actual tool trace

This directly evaluates agentic behavior.

---

## 11. Two-agent system structure

### Evaluation focus

How the candidate structures an agentic system.

### Architecture

Use two agents:

1. Research Agent
2. Evidence Verifier

### Why two agents

There are two distinct reasoning responsibilities:

- evidence acquisition and synthesis
- independent claim-evidence verification

Search, retrieval, vision, Python, and ML remain tools because they are capabilities supporting the Research Agent's central objective.

### Evaluation / demonstration

Show at least one trace with:

Research Agent
-> draft answer
-> Evidence Verifier
-> request more evidence
-> Research Agent
-> revised evidence
-> Evidence Verifier
-> final status

---

## 12. Evidence verification

### Evaluation focus

Robustness, scientific reliability, and multi-agent design.

### Architecture

The Evidence Verifier independently checks claim support.

### Implementation

Suggested statuses:

- PASS
- NEED_MORE_EVIDENCE
- UNSUPPORTED_CLAIM
- CONFLICTING_EVIDENCE

Suggested structured fields:

- status
- claim
- reason
- requested_evidence

The verifier should focus on claim-evidence alignment rather than regenerating the answer from scratch.

### Evaluation / demonstration

Measure whether the verifier correctly identifies:

- supported claims
- unsupported claims
- insufficient evidence
- conflicting evidence
- overconfident wording

---

## 13. Robustness to missing evidence

### Evaluation focus

The interview explicitly asks for robustness to missing scientific evidence.

### Architecture

The verifier must be able to reject unsupported conclusions.

### Implementation

Example behavior:

User asks for a universal conclusion.

Retrieved literature supports only limited settings.

Expected result:

- do not generalize universally
- mark evidence as insufficient
- optionally search for more evidence
- return explicit uncertainty if still unresolved

### Evaluation / demonstration

Include at least 1 deliberately unanswerable or overbroad evaluation question.

Success means:

- no fabricated answer
- no invented citation
- explicit uncertainty
- appropriate verifier status

---

## 14. Robustness to conflicting evidence

### Evaluation focus

The interview explicitly asks for robustness to conflicting scientific evidence.

### Architecture

The verifier checks whether papers agree rather than automatically averaging conclusions.

### Implementation

When studies disagree:

- identify conflicting findings
- preserve separate citations
- inspect methodology/context where possible
- surface plausible methodological differences without overstating causality
- return uncertainty when appropriate

### Evaluation / demonstration

Include 2 conflicting-evidence evaluation questions where possible.

Success means that disagreement is surfaced rather than collapsed into one overconfident conclusion.

---

## 15. Citation and provenance reliability

### Evaluation focus

Scientific transparency and reproducibility.

### Architecture

Carry provenance through the entire pipeline.

### Implementation

Every evidence object should retain:

- paper
- page
- section
- source type
- modality
- content

Final claims should reference evidence IDs that resolve to real paper/page evidence.

### Evaluation / demonstration

Check:

- cited paper exists
- cited page exists
- evidence content is relevant to the claim
- figure/table citation points to the inspected page

Do not treat citation formatting alone as evidence validity.

---

## 16. Structured communication between agents

### Architecture

Avoid free-form agent-to-agent conversation where a schema is practical.

### Implementation

Research Agent output:

- draft_answer
- claims
- evidence_ids
- uncertainty
- tool_trace

Verifier output:

- status
- claim
- reason
- requested_evidence

### Evaluation / demonstration

Add basic schema or unit tests.

Structured interfaces should make failures easier to debug and demonstrate.

---

## 17. Bounded autonomous behavior

### Evaluation focus

Engineering robustness.

### Architecture

Agent loops must have explicit stopping conditions.

### Implementation

Possible controls:

- maximum literature-search rounds
- maximum tool calls per question
- maximum 2 verification rounds
- stop when evidence is sufficient
- stop when no meaningful new evidence is found
- stop when execution budget is exhausted

### Evaluation / demonstration

Test a difficult question and verify that the system terminates with uncertainty instead of looping indefinitely.

---

## 18. Chunking quality

### Evaluation focus

Retrieval / RAG quality.

### Architecture

Preserve meaningful scientific context.

### Implementation

Prefer:

- section-aware chunking
- page-aware chunking
- approximately 500-800 tokens
- modest overlap
- full provenance metadata

Avoid arbitrary character slicing where possible.

### Evaluation / demonstration

Inspect retrieval failures and document whether they are caused by:

- chunk size
- section boundaries
- retrieval ranking
- missing PDF content

---

## 19. Tool failure and fallback behavior

### Evaluation focus

Robust engineering.

### Architecture

Tools should fail gracefully.

### Implementation

Examples:

PDF unavailable:
-> try another source or paper

PDF malformed:
-> log failure and continue

Figure unreadable:
-> avoid fabricated precision

Python analysis missing required fields:
-> refuse unsupported calculation

Search returns weak evidence:
-> return uncertainty or reformulate query

### Evaluation / demonstration

Test at least one controlled failure case.

Record fallback behavior in the execution trace.

---

## 20. Reproducibility

### Evaluation focus

The job description emphasizes reproducibility and scientific software development.

### Architecture

Persist important artifacts and environment information.

### Implementation

Expected repository artifacts:

- `environment.yml`
- `.env.example`
- saved corpus metadata
- cached papers where appropriate
- saved embeddings where appropriate
- random seeds
- saved evaluation results
- development log
- execution traces

Secrets must not be committed.

### Evaluation / demonstration

README should contain commands sufficient to reproduce the demo from the saved corpus/environment.

---

## 21. Coding / AI agents used during development

### Evaluation focus

The take-home explicitly asks how coding / AI agents were used while building the system.

### Architecture

Development-agent usage is separate from runtime agent architecture.

### Implementation

Use Codex for suitable tasks such as:

- repository scaffolding
- API wrappers
- test skeletons
- debugging
- refactoring
- code review
- edge-case identification

Human-controlled decisions include:

- scientific scope
- architecture
- agent responsibility decomposition
- evaluation methodology
- scientific validity criteria
- interpretation of results

Record meaningful activity in:

`docs/DEVELOPMENT_LOG.md`

### Evaluation / demonstration

README or presentation should briefly explain:

- what Codex accelerated
- what decisions were intentionally kept under human control
- how AI-generated code was reviewed and tested

---

## 22. No polished UI required

### Requirement / constraint

The interview does not expect a polished product or UI.

### Architecture

Prioritize core scientific and agentic functionality.

### Implementation

A CLI or simple Python entry point is sufficient.

Possible usage:

`python -m src.main --question "..."`

Exact CLI design may change during implementation.

### Evaluation / demonstration

Demonstrate:

- question
- tool trace
- retrieved evidence
- verifier result
- final answer

Do not spend implementation time on frontend polish until core requirements are complete.

---

## 23. Environmental scientist as target user

### Architecture

Keep system-facing AI complexity hidden from the scientific user.

### Implementation

Example questions should use environmental-science language.

Avoid requiring users to specify:

- retrieval parameters
- embedding models
- vector stores
- tool names
- agent names

### Evaluation / demonstration

Use example questions that resemble realistic research questions from environmental / energy scientists.

---

## 24. Scientific limitations

### Architecture

Limitations should be explicit rather than hidden.

### Required limitations to communicate

- a small automatically collected corpus is not a systematic review
- multimodal extraction may fail on complex or low-resolution figures
- cross-paper metrics are not always directly comparable
- the ML experiment is dataset-specific
- predictive feature importance is not causal evidence
- automatically retrieved literature may contain incomplete or conflicting findings

### Evaluation / demonstration

The verifier and final answer format should support uncertainty-aware wording.

The README and final presentation should include a limitations section.

---

# Evaluation Question Taxonomy

The evaluation harness should contain approximately 12 questions.

The taxonomy is frozen, while the exact questions and gold evidence should be created only after automatic corpus collection and manual inspection.

Suggested distribution:

- 4 text-RAG questions
- 1 literature-expansion question
- 2 multimodal figure/table questions
- 1 ML/data-analysis question
- 1 mixed literature + ML question
- 1 missing-evidence question
- 2 conflicting-evidence questions

Total:

12 questions

---

# Core Evaluation Story

The final evaluation should be explainable through three questions.

## Layer 1: Agent behavior

Did the agent choose the right research action?

Examples:

- use RAG when text is sufficient
- use literature search when corpus evidence is insufficient
- use vision for figure/table questions
- use ML analysis when empirical data is required
- avoid unnecessary tools

## Layer 2: Evidence retrieval

Did the system retrieve the right scientific evidence?

Primary measures:

- Recall@3
- Recall@5
- citation validity

Optional:

dense-only vs hybrid retrieval ablation.

## Layer 3: Scientific reliability

Did the final claim actually follow from the evidence?

Evaluate:

- claim support
- missing evidence handling
- conflict detection
- uncertainty
- overclaim prevention

---

# Final Design Principle

Every implementation choice should answer two questions:

1. Which take-home requirement or interview evaluation criterion does this component serve?
2. How will we demonstrate or measure that it works?

Do not add architectural complexity unless both questions have a clear answer.
