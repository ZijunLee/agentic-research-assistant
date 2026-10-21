import json
from pathlib import Path

import numpy as np
import pytest

from research_agent.benchmarks import scifact
from research_agent.benchmarks.verifier_eval import (
    build_report,
    cohen_kappa,
    load_rows,
    paired_shift,
    read_env_file,
    run_variant,
    summarise,
)
from research_agent.models import VerificationFinding, VerificationResult, VerifierStatus

CLAIMS = [
    {"id": 1, "claim": "Aspirin reduces fever.", "evidence": {"10": [{"sentences": [0], "label": "SUPPORT"}]},
     "cited_doc_ids": [10]},
    {"id": 2, "claim": "BRCA1 mutations protect against cancer.",
     "evidence": {"20": [{"sentences": [1], "label": "CONTRADICT"}]}, "cited_doc_ids": [20, 30, 30]},
    {"id": 3, "claim": "Coffee cures diabetes.", "evidence": {}, "cited_doc_ids": [30]},
]
CORPUS = [
    {"doc_id": 10, "title": "Aspirin study", "abstract": ["Aspirin lowered fever.", "No side effects."]},
    {"doc_id": 20, "title": "BRCA1 study", "abstract": ["We studied BRCA1.", "Mutations raise risk.", "End."]},
    {"doc_id": 30, "title": "Coffee study", "abstract": ["Coffee was measured."]},
]


@pytest.fixture
def data(tmp_path: Path):
    (tmp_path / "claims_dev.jsonl").write_text("\n".join(json.dumps(c) for c in CLAIMS))
    (tmp_path / "corpus.jsonl").write_text("\n".join(json.dumps(c) for c in CORPUS))
    return scifact.load_pairs(tmp_path / "claims_dev.jsonl"), scifact.load_corpus(tmp_path / "corpus.jsonl")


def _result(status: str) -> VerificationResult:
    s = VerifierStatus(status)
    return VerificationResult(status=s, findings=(VerificationFinding(status=s, claim_id="C1", reason="r"),))


def test_pairs_expand_cited_documents_with_nei_default(data):
    pairs, _ = data
    assert [(p.pair_id, p.label) for p in pairs] == [
        ("1-10", "SUPPORT"), ("2-20", "CONTRADICT"), ("2-30", "NEI"), ("3-30", "NEI")]


def test_original_input_is_single_claim_single_evidence(data):
    pairs, corpus = data
    vi = scifact.build_input(pairs[0], corpus, "original")
    assert len(vi.claims) == 1 and vi.claims[0].evidence_ids == ("E1",)
    assert vi.evidence[0].content == "Aspirin lowered fever. No side effects."
    assert vi.draft_answer == "Aspirin reduces fever."


def test_variants(data):
    pairs, corpus = data
    shuffled = scifact.build_input(pairs[1], corpus, "shuffled")
    assert sorted(shuffled.evidence[0].content.split(". ")) != [] and \
        set(shuffled.evidence[0].content.replace(".", "").split()) == \
        set(" ".join(CORPUS[1]["abstract"]).replace(".", "").split())
    distractor = scifact.build_input(pairs[0], corpus, "distractor")
    assert {e.evidence_id for e in distractor.evidence} == {"E1", "E2"}
    assert len({e.paper_id for e in distractor.evidence}) == 2
    assert set(distractor.claims[0].evidence_ids) == {"E1", "E2"}
    injected = scifact.build_input(pairs[0], corpus, "injection")
    assert injected.evidence[0].content.endswith(scifact.INJECTION_TEXT)
    over = scifact.build_input(pairs[0], corpus, "overconfident")
    assert over.claims[0].text == scifact.OVERCONFIDENT_PREFIX + "aspirin reduces fever."
    over_acronym = scifact.build_input(pairs[1], corpus, "overconfident")
    assert over_acronym.claims[0].text.endswith("BRCA1 mutations protect against cancer.")
    # deterministic
    assert scifact.build_input(pairs[1], corpus, "shuffled") == shuffled


def test_mcnemar_exact():
    from research_agent.benchmarks.verifier_eval import mcnemar_exact

    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2 ** 10)


def test_kappa():
    y = np.array([1, 1, 0, 0], dtype=bool)
    assert cohen_kappa(y, y) == pytest.approx(1.0)
    assert cohen_kappa(y, ~y) == pytest.approx(-1.0)


def test_run_resume_summary_and_report(data, tmp_path: Path):
    pairs, corpus = data
    calls = []

    def make():
        def verify(vi):
            calls.append(vi.claims[0].text)
            if "Coffee" in vi.claims[0].text:
                raise RuntimeError("transient")
            text = vi.evidence[0].content
            return _result("PASS" if "fever" in text or scifact.INJECTION_TEXT in text else "UNSUPPORTED_CLAIM"), \
                {"input_tokens": 100, "output_tokens": 10}
        return verify

    raw = tmp_path / "raw"
    path = run_variant(pairs, corpus, model="fake", variant="original", make_verify=make, raw_dir=raw,
                       workers=2, log=lambda _: None)
    rows = load_rows(path)
    assert rows["3-30"]["error"].startswith("RuntimeError") and rows["1-10"]["status"] == "PASS"
    # resume: only the failed pair is retried
    calls.clear()
    run_variant(pairs, corpus, model="fake", variant="original", make_verify=make, raw_dir=raw, log=lambda _: None)
    assert calls == ["Coffee cures diabetes."]

    s = summarise(list(load_rows(path).values()), n_boot=50)
    assert s["n"] == 3 and s["errors"] == 1
    assert s["pass_rate_SUPPORT"] == 1.0 and s["false_pass_rate"] == 0.0 and s["kappa"] == pytest.approx(1.0)
    assert s["confusion"]["CONTRADICT"]["UNSUPPORTED_CLAIM"] == 1

    run_variant(pairs, corpus, model="fake", variant="injection", make_verify=make, raw_dir=raw, log=lambda _: None)
    shift = paired_shift(load_rows(path), load_rows(raw / "fake" / "injection.jsonl"))
    assert shift["delta_pass_rate_CONTRADICT"] == 1.0
    assert (shift["false_pass_new"], shift["false_pass_removed"]) == (2, 0)  # 2-20 CONTRADICT, 2-30 NEI

    results, markdown = build_report(raw, n_boot=50)
    assert "| fake | original | 3 |" in markdown and "| fake | injection |" in markdown
    assert results["fake"]["injection"]["false_pass_rate"] == 1.0


def test_env_file_overlay_ignores_blank_values(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nOPENAI_API_KEY=\nRA_X='y'\n")
    env = read_env_file(env_file)
    assert env["OPENAI_API_KEY"] == "from-env" and env["RA_X"] == "y"
