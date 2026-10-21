# Evidence Verifier on SciFact (dev split, claim × cited abstract)

Positive class: SciFact SUPPORT ↔ verifier PASS. False PASS = PASS on a CONTRADICT or NEI pair. 95% CIs: bootstrap over claims. Calls use the production verifier prompt and model defaults (no temperature set), so verdicts are not deterministic: the `shuffled` row is a practical noise floor for the perturbation comparisons.

## Agreement with human labels

| model | variant | n | κ (95% CI) | accuracy | PASS on SUPPORT | false PASS (95% CI) | PASS on CONTRADICT | PASS on NEI | errors | tokens in / out |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4.1-2025-04-14 | original | 339 | 0.715 (0.64–0.79) | 86.7% | 72.5% | 3.5% (1.1%–6.3%) | 2.8% | 3.8% | 0 | 349,134 / 50,310 |
| gpt-4.1-2025-04-14 | shuffled | 339 | 0.734 (0.66–0.80) | 87.6% | 73.2% | 2.5% (0.5%–5.0%) | 1.4% | 3.1% | 0 | 349,178 / 51,443 |
| gpt-4.1-2025-04-14 | distractor | 339 | 0.509 (0.42–0.59) | 78.2% | 47.8% | 1.0% (0.0%–2.6%) | 1.4% | 0.8% | 0 | 474,668 / 58,711 |
| gpt-4.1-2025-04-14 | overconfident | 339 | 0.026 (0.00–0.06) | 60.2% | 2.2% | 0.0% (0.0%–0.0%) | 0.0% | 0.0% | 0 | 353,697 / 60,898 |
| gpt-4.1-2025-04-14 | injection | 339 | 0.746 (0.66–0.81) | 87.9% | 80.4% | 7.0% (3.7%–10.9%) | 7.0% | 6.9% | 0 | 356,919 / 48,723 |
| gpt-4.1-mini-2025-04-14 | original | 339 | 0.778 (0.71–0.84) | 89.4% | 84.1% | 7.0% (3.6%–10.8%) | 5.6% | 7.7% | 0 | 349,134 / 40,588 |
| gpt-4.1-mini-2025-04-14 | shuffled | 339 | 0.735 (0.66–0.81) | 87.3% | 81.2% | 8.5% (5.0%–12.4%) | 7.0% | 9.2% | 0 | 349,178 / 40,522 |
| gpt-4.1-mini-2025-04-14 | distractor | 338 | 0.436 (0.34–0.52) | 74.9% | 45.3% | 5.0% (2.3%–8.2%) | 4.2% | 5.4% | 1 | 473,439 / 49,601 |
| gpt-4.1-mini-2025-04-14 | overconfident | 339 | 0.576 (0.48–0.67) | 80.8% | 56.5% | 2.5% (0.5%–4.9%) | 2.8% | 2.3% | 0 | 353,697 / 44,770 |
| gpt-4.1-mini-2025-04-14 | injection | 339 | 0.687 (0.61–0.76) | 84.4% | 92.0% | 20.9% (15.5%–26.6%) | 19.7% | 21.5% | 0 | 356,919 / 38,157 |

## Perturbations: verdict changes on the same pairs

| model | variant | paired n | PASS flips | Δ PASS on SUPPORT | Δ PASS on CONTRADICT | Δ PASS on NEI | false PASS new / removed (McNemar p) |
|---|---|---|---|---|---|---|---|
| gpt-4.1-2025-04-14 | shuffled | 339 | 5.6% | +0.7 pp | -1.4 pp | -0.8 pp | 1 / 3 (p=0.62) |
| gpt-4.1-2025-04-14 | distractor | 339 | 13.9% | -24.6 pp | -1.4 pp | -3.1 pp | 0 / 5 (p=0.062) |
| gpt-4.1-2025-04-14 | overconfident | 339 | 30.7% | -70.3 pp | -2.8 pp | -3.8 pp | 0 / 7 (p=0.016) |
| gpt-4.1-2025-04-14 | injection | 339 | 5.9% | +8.0 pp | +4.2 pp | +3.1 pp | 7 / 0 (p=0.016) |
| gpt-4.1-mini-2025-04-14 | shuffled | 339 | 4.4% | -2.9 pp | +1.4 pp | +1.5 pp | 5 / 2 (p=0.45) |
| gpt-4.1-mini-2025-04-14 | distractor | 338 | 18.6% | -38.7 pp | -1.4 pp | -2.3 pp | 2 / 6 (p=0.29) |
| gpt-4.1-mini-2025-04-14 | overconfident | 339 | 13.9% | -27.5 pp | -2.8 pp | -5.4 pp | 0 / 9 (p=0.0039) |
| gpt-4.1-mini-2025-04-14 | injection | 339 | 12.1% | +8.0 pp | +14.1 pp | +13.8 pp | 28 / 0 (p=7.5e-09) |

## Confusion matrices (rows: SciFact label, columns: verifier status)

**gpt-4.1-2025-04-14 / original**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 100 | 29 | 4 | 5 |
| CONTRADICT | 2 | 7 | 6 | 56 |
| NEI | 5 | 89 | 31 | 5 |

**gpt-4.1-2025-04-14 / shuffled**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 101 | 27 | 2 | 8 |
| CONTRADICT | 1 | 11 | 7 | 52 |
| NEI | 4 | 98 | 23 | 5 |

**gpt-4.1-2025-04-14 / distractor**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 66 | 43 | 16 | 13 |
| CONTRADICT | 1 | 14 | 6 | 50 |
| NEI | 1 | 95 | 28 | 6 |

**gpt-4.1-2025-04-14 / overconfident**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 3 | 82 | 32 | 21 |
| CONTRADICT | 0 | 6 | 13 | 52 |
| NEI | 0 | 71 | 53 | 6 |

**gpt-4.1-2025-04-14 / injection**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 111 | 20 | 2 | 5 |
| CONTRADICT | 5 | 9 | 6 | 51 |
| NEI | 9 | 97 | 18 | 6 |

**gpt-4.1-mini-2025-04-14 / original**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 116 | 15 | 7 | 0 |
| CONTRADICT | 4 | 7 | 51 | 9 |
| NEI | 10 | 56 | 64 | 0 |

**gpt-4.1-mini-2025-04-14 / shuffled**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 112 | 20 | 6 | 0 |
| CONTRADICT | 5 | 9 | 47 | 10 |
| NEI | 12 | 50 | 68 | 0 |

**gpt-4.1-mini-2025-04-14 / distractor**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 62 | 46 | 27 | 2 |
| CONTRADICT | 3 | 6 | 61 | 1 |
| NEI | 7 | 33 | 90 | 0 |

**gpt-4.1-mini-2025-04-14 / overconfident**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 78 | 47 | 13 | 0 |
| CONTRADICT | 2 | 7 | 58 | 4 |
| NEI | 3 | 37 | 90 | 0 |

**gpt-4.1-mini-2025-04-14 / injection**

| label | PASS | NEED_MORE_EVIDENCE | UNSUPPORTED_CLAIM | CONFLICTING_EVIDENCE |
|---|---|---|---|---|
| SUPPORT | 127 | 3 | 8 | 0 |
| CONTRADICT | 14 | 4 | 48 | 5 |
| NEI | 28 | 39 | 62 | 1 |

