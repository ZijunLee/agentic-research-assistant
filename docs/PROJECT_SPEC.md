# Project Specification

## Project title

An Agentic Research Assistant for Weather and Climate Impacts on Renewable Energy

## Goal

Build a small but complete and testable agentic AI system for scientific research.

The system is intended for environmental / energy scientists who want evidence-grounded answers from scientific literature and data without needing expertise in RAG, embeddings, agent frameworks, or ML engineering.

The prototype should demonstrate:

- automated literature discovery and collection
- scientific paper processing and indexing
- evidence-grounded question answering
- adaptive tool selection
- multimodal figure/table inspection
- a small machine-learning component
- systematic evaluation
- robustness to missing and conflicting scientific evidence

The prototype is constrained to approximately 3-5 hours of implementation effort and should therefore prioritize clear architecture, testability, and scientific reliability over product polish or UI.

---

## Scientific scope

The broad scientific domain is:

Weather and climate impacts on renewable-energy systems.

Primary focus:

- solar energy
- wind energy
- meteorological and climate variables
- renewable-energy generation
- renewable-energy forecasting
- reliability under changing or extreme weather conditions

Relevant environmental variables may include:

- solar irradiance
- temperature
- cloud cover
- humidity
- wind speed
- precipitation
- seasonal variation
- extreme weather indicators

The system may answer scientific questions such as:

- How do weather conditions affect solar-power generation and forecasting?
- Which meteorological variables are commonly used for renewable-energy prediction?
- How does wind variability affect wind-power forecasting?
- Do studies agree on the effect of temperature on photovoltaic generation?
- Under which weather conditions do forecasting models become less reliable?
- How do reported effects differ across regions or datasets?
- Does the empirical ML case study agree with findings in the literature?

The prototype does NOT claim to cover all of environmental science or climate science.

The collected corpus is also NOT intended to constitute a systematic review.

---

## Target user

The target user is an environmental or energy scientist.

The user interface should expose:

- scientific answers
- cited evidence
- uncertainty
- disagreements between studies

The user should not need to understand:

- vector databases
- embeddings
- RAG terminology
- agent orchestration internals

---

## Core architecture

The frozen runtime architecture contains two agents.

### 1. Research Agent

Goal:

Gather and synthesize sufficient scientific evidence to answer the user question.

The Research Agent decides dynamically which tools to use and in which order.

Available tool categories:

- literature discovery
- indexed paper retrieval / RAG
- paper metadata or full-paper access
- PDF page / figure / table inspection
- Python scientific analysis
- machine-learning analysis

The Research Agent should repeatedly assess:

- What evidence is currently available?
- Is that evidence sufficient?
- Is additional literature required?
- Is the answer contained in text?
- Is relevant information mainly contained in a figure or table?
- Is numerical or ML analysis required?
- Are multiple sources needed before making the claim?

The Research Agent should produce a structured draft containing:

- draft answer
- claims
- evidence references
- uncertainty
- tool trace

### 2. Evidence Verifier

Goal:

Determine whether the Research Agent's claims are actually justified by the available evidence.

The Evidence Verifier is not a second answer generator.

Its responsibility is claim-evidence checking.

It should check:

- whether important claims are supported
- whether cited evidence corresponds to the claim
- whether evidence is sufficient
- whether multiple papers conflict
- whether uncertainty is represented appropriately
- whether the answer overclaims beyond the evidence

Suggested verifier statuses:

- PASS
- NEED_MORE_EVIDENCE
- UNSUPPORTED_CLAIM
- CONFLICTING_EVIDENCE

If verification fails, the verifier should return a structured reason and, where useful, indicate what evidence is missing.

The Research Agent may then gather additional evidence and resubmit.

Maximum verification rounds:

2

After the maximum number of rounds, unresolved uncertainty or disagreement should be explicitly surfaced to the user.

---

## Why two agents

The system uses two agents because there are two genuinely distinct reasoning responsibilities:

1. evidence acquisition and synthesis
2. independent claim-evidence verification

Literature search, RAG, figure inspection, and Python analysis are treated as tools rather than separate agents because they are capabilities supporting the Research Agent's evidence-gathering objective.

A third literature-discovery agent is deliberately excluded from the MVP because automated discovery for a small corpus does not yet require a sufficiently independent reasoning responsibility to justify additional coordination.

A future extension could separate literature discovery into its own agent if the task evolves toward systematic-review-style search involving iterative query expansion, citation snowballing, coverage analysis, and independent stopping criteria.

---

## Agentic design principle

The system must not execute a fixed pipeline such as:

search -> download -> retrieve -> answer

Instead, it should support adaptive trajectories such as:

Question A:
retrieve evidence -> answer

Question B:
retrieve evidence -> inspect figure -> answer

Question C:
retrieve evidence -> Python analysis -> answer

Question D:
search additional literature -> retrieve -> compare studies -> verifier -> answer

Question E:
retrieve -> detect conflict -> search more evidence -> verifier -> uncertainty-aware answer

The agentic component is the adaptive selection and sequencing of tools based on the scientific question and the evidence gathered so far.

---

## Automated literature discovery

The final paper corpus must be collected automatically rather than manually curated.

Preferred literature source:

OpenAlex

A comparable scientific literature API may be substituted if needed.

Initial search strategy:

1. define the scientific topic
2. generate or expand several search queries
3. retrieve approximately 30-50 candidate papers
4. apply basic relevance and metadata filters
5. rank candidates using lexical and/or semantic relevance
6. deduplicate results
7. prefer open-access full-text papers
8. automatically collect approximately 8-12 papers
9. ingest and index the resulting corpus

Example seed queries:

- weather impacts renewable energy generation
- meteorological variables solar power generation
- weather solar power forecasting
- weather wind power forecasting
- climate variability renewable energy generation
- extreme weather renewable energy reliability

Possible inclusion criteria:

- scientifically relevant to renewable energy
- focus on solar and/or wind where possible
- meaningful treatment of weather, meteorological, or climate variables
- relevance to generation, forecasting, reliability, or environmental effects
- English-language metadata/full text where practical
- open-access full text preferred
- relatively recent literature preferred, but no overly strict year cutoff is required

Possible exclusions:

- purely hardware-oriented work with no meaningful weather-energy connection
- duplicates or alternate versions of the same study
- papers without usable metadata or full text when replacement sources are available

Deduplication should use DOI where available and normalized title matching as a fallback.

---

## Paper ingestion and provenance

For every collected PDF, preserve both textual and page-level visual representations.

Expected structure:

PDF
- page text
- section-aware chunks
- rendered page images
- metadata
- provenance

Suggested text chunk size:

approximately 500-800 tokens with overlap

Prefer section-aware or page-aware chunking over arbitrary character slicing.

Every evidence object should retain at least:

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

The final answer must remain traceable to original paper/page evidence.

---

## Retrieval

Use hybrid retrieval where feasible.

Components:

- lexical retrieval such as BM25
- dense semantic retrieval using embeddings

A simple weighted or rank-fusion approach is sufficient.

The exact implementation should remain simple enough for the take-home scope.

Retrieval should return evidence objects with provenance metadata.

An optional small retrieval ablation may compare:

- dense-only retrieval
- hybrid retrieval

using Recall@3 and Recall@5.

---

## Multimodal handling

Scientific figures and tables should be treated as first-class evidence.

For the MVP, do not build a complex figure-detection, OCR, or multimodal-vector-index pipeline.

Preferred design:

1. retrieve relevant text
2. detect that the text references a figure/table or lacks the required visual information
3. identify the relevant paper page
4. render the page as an image
5. send the page image plus a focused scientific question to a multimodal model
6. return a structured visual evidence object

Principle:

Text retrieves the page; vision interprets the page.

Example use cases:

- identify which weather condition has the highest reported error in a figure
- read a model-comparison table
- interpret trends shown only visually
- refuse to provide exact numbers if the figure is not legible enough

The system should avoid calling the multimodal tool when textual evidence is already sufficient.

---

## Scientific and Python analysis

The Research Agent may invoke Python-based scientific analysis when the question requires quantitative comparison.

Examples:

- compare values extracted from multiple papers
- compute summary statistics
- generate a small plot
- compare model metrics
- analyze a scientific dataset

Cross-paper numerical comparison must check scientific comparability.

Do not directly aggregate results when studies differ materially in:

- units
- dataset
- forecast horizon
- normalization
- evaluation protocol
- metric definition

If values are not scientifically comparable, the system should state that rather than forcing an aggregation.

---

## Machine-learning component

The required ML component is a small supervised regression case study.

Scientific question:

How well can meteorological variables predict solar generation in a small dataset?

Inputs may include:

- irradiance
- temperature
- cloud cover
- humidity
- wind speed
- related weather variables available in the selected dataset

Target:

solar / photovoltaic power generation

Minimum modeling design:

1. define features and target
2. clean and prepare data
3. use a chronological train/test split
4. prevent temporal leakage
5. implement a simple baseline
6. fit a linear model such as Ridge regression
7. fit a nonlinear model such as Random Forest or Gradient Boosting
8. evaluate on held-out data
9. compute MAE
10. compute RMSE
11. compute R2
12. compute permutation feature importance

Where useful, compare train and test performance to inspect overfitting.

Any preprocessing fitted from data must be fitted only on the training portion.

Feature importance must be described as predictive importance for this dataset, not as causal evidence.

The ML case study is a focused example and must not be generalized as proof of universal weather-energy relationships.

---

## Structured agent interfaces

Prefer structured communication between components.

Suggested Research Agent output:

- draft_answer
- claims
- evidence_ids
- uncertainty
- tool_trace

Suggested Verifier output:

- status
- claim
- reason
- requested_evidence

Avoid unstructured free-form agent-to-agent conversation when a schema is practical.

---

## Stopping and bounded behavior

Autonomous behavior must be bounded.

Suggested controls:

- maximum search rounds
- maximum tool calls per question
- maximum 2 verification rounds
- stop when evidence is sufficient
- stop when additional search produces no meaningful new evidence
- stop when execution budget is exhausted

If the system cannot resolve a question within the budget, it should return an uncertainty-aware response rather than loop indefinitely.

---

## Robustness and failure handling

The prototype should handle at least simple failures gracefully.

Examples:

- open-access PDF unavailable
- PDF download failure
- malformed PDF
- figure unreadable
- missing fields for analysis
- search returns weak evidence
- papers disagree
- evidence does not support the user's wording

Fallback behavior may include:

- trying another relevant paper
- relying on text if visual extraction fails
- asking the verifier to mark insufficient evidence
- returning uncertainty
- refusing unsupported numerical precision

Paper text should be treated as untrusted evidence, not executable instructions.

---

## Reproducibility

The repository should support reproducible execution.

Expected artifacts:

- README.md
- environment.yml
- .env.example
- fixed or documented random seeds
- cached corpus metadata
- cached papers where appropriate
- cached embeddings where appropriate
- saved evaluation results
- execution traces
- documented development decisions

Secrets must not be committed.

---

## Development with coding agents

The take-home explicitly evaluates how coding / AI agents are used while building the system.

Use Codex as a development accelerator for tasks such as:

- scaffolding modules
- implementing API wrappers
- writing initial tests
- debugging PDF parsing
- refactoring
- reviewing edge cases

Human-controlled decisions should include:

- scientific scope
- architecture
- agent responsibilities
- evaluation design
- reliability criteria
- final interpretation of experimental results

Meaningful AI-assisted development actions should be recorded in:

docs/DEVELOPMENT_LOG.md

---

## Non-goals

The MVP does not require:

- polished web UI
- production deployment
- large-scale vector infrastructure
- systematic-review completeness
- complex multi-agent orchestration
- separate search/vision/retrieval agents
- custom deep-learning training
- advanced chart digitization
- large-scale MLOps infrastructure

---

## Success criteria

The prototype is successful if it can demonstrate all of the following:

1. automatically discover and collect relevant scientific papers
2. process and index paper text with paper/page provenance
3. answer scientific questions using cited evidence
4. choose tools adaptively rather than following one fixed sequence
5. inspect figures/tables when textual evidence is insufficient
6. run a real small ML experiment with correct experimental methodology
7. use a separate verifier to identify unsupported, missing, or conflicting evidence
8. run a reproducible evaluation harness
9. expose limitations and uncertainty rather than overclaim
10. provide working code and clear instructions suitable for a 10-minute technical demonstration
