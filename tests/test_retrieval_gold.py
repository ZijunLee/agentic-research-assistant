import json
from pathlib import Path

from l3s_agent.retrieval.evaluation import load_gold


GOLD_PATH = Path(__file__).parents[1] / "evaluation" / "retrieval_gold.json"


def test_tracked_gold_set_has_six_page_level_questions_without_answers() -> None:
    raw = json.loads(GOLD_PATH.read_text())
    gold = load_gold(GOLD_PATH)
    assert len(gold) == 6
    assert {item.category for item in gold} == {
        "solar_irradiance_cloud_forecasting",
        "solar_intermittency_reliability",
        "wind_nwp_forecasting",
        "wind_atmospheric_variability_wakes",
        "climate_impacts_renewable_generation",
        "cross_modality_complementarity",
    }
    assert all(item.gold_pages for item in gold)
    assert all(page >= 1 for item in gold for _, page in item.gold_pages)
    assert all("answer" not in key.lower() for item in raw["queries"] for key in item)
    assert raw["source_evidence_sha256"] == (
        "bafc20e33a93712a1c6f4a309fa2cc8773b846884ef4cbb21ae76d17524b549a"
    )
