from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import l3s_agent.analysis.berlin_solar as berlin_module
from l3s_agent.analysis.berlin_solar import (
    ANALYSIS_NAME,
    EXCLUDED_COLUMNS,
    FEATURE_GROUPS,
    RAW_FEATURES,
    BerlinSolarAnalysisError,
    BerlinWeatherSolarAnalysisTool,
    analysis_result_id_from_provenance,
    berlin_analysis_identity_provenance,
    build_model_pipelines,
    load_berlin_dataset,
    run_berlin_weather_solar_analysis,
    split_dataset,
)
from l3s_agent.config import MLDatasetConfig


FIXTURE = Path(__file__).parent / "fixtures" / "ml" / "berlin_small.csv"


def _identity_provenance():
    return berlin_analysis_identity_provenance(
        dataset_sha256="a" * 64,
        train_rows=12,
        test_rows=12,
    )


def _copy_rows(source: Path, destination: Path, mutate=None) -> None:
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if mutate is not None:
        mutate(rows)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)


def test_schema_timestamp_split_and_frozen_features() -> None:
    dataset = load_berlin_dataset(FIXTURE)
    x_train, x_test, y_train, y_test = split_dataset(dataset)

    assert dataset.rows == 24
    assert dataset.columns == 24
    assert len(x_train) == len(y_train) == 12
    assert len(x_test) == len(y_test) == 12
    assert dataset.timestamps[11].year == 2018
    assert dataset.timestamps[12].year == 2019
    assert max(dataset.timestamps[:12]) < min(dataset.timestamps[12:])
    assert RAW_FEATURES == (
        "Temperature", "Clearsky.DHI", "Clearsky.DNI", "Clearsky.GHI",
        "Cloud.Type", "Dew.Point", "DHI", "DNI", "GHI",
        "Relative.Humidity", "Solar.Zenith.Angle", "Surface.Albedo",
        "Pressure", "Precipitable.Water", "Wind.Direction", "Wind.Speed",
    )
    assert set(EXCLUDED_COLUMNS) == {
        "Year", "Month", "Day", "Hour", "Minute", "Fill.Flag", "Ozone",
        "X50Hertz..MW.",
    }
    assert set().union(*map(set, FEATURE_GROUPS.values())) == set(RAW_FEATURES)


def test_unordered_rows_are_sorted_and_duplicate_timestamps_are_rejected(tmp_path) -> None:
    unordered = tmp_path / "unordered.csv"
    _copy_rows(FIXTURE, unordered, lambda rows: rows.__setitem__(slice(1, 3), rows[1:3][::-1]))
    dataset = load_berlin_dataset(unordered)
    assert list(dataset.timestamps) == sorted(dataset.timestamps)

    duplicate = tmp_path / "duplicate.csv"
    _copy_rows(FIXTURE, duplicate, lambda rows: rows.append(rows[1]))
    with pytest.raises(BerlinSolarAnalysisError, match="duplicate timestamps"):
        load_berlin_dataset(duplicate)


def test_invalid_timestamp_and_schema_are_rejected(tmp_path) -> None:
    invalid = tmp_path / "invalid.csv"
    _copy_rows(FIXTURE, invalid, lambda rows: rows[1].__setitem__(1, "13"))
    with pytest.raises(BerlinSolarAnalysisError, match="invalid value"):
        load_berlin_dataset(invalid)

    wrong_schema = tmp_path / "wrong-schema.csv"
    _copy_rows(FIXTURE, wrong_schema, lambda rows: rows[0].__setitem__(5, "Wrong"))
    with pytest.raises(BerlinSolarAnalysisError, match="schema"):
        load_berlin_dataset(wrong_schema)


def test_preprocessing_uses_training_data_and_dense_cyclic_encoding() -> None:
    dataset = load_berlin_dataset(FIXTURE)
    x_train, x_test, y_train, _ = split_dataset(dataset)
    pipelines = build_model_pipelines()
    assert set(pipelines) == {"baseline_mean", "ridge", "hist_gradient_boosting"}

    ridge = pipelines["ridge"].fit(x_train, y_train)
    transformed = ridge.named_steps["preprocess"].transform(x_test)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 12
    assert not np.isnan(transformed).any()
    numeric = ridge.named_steps["preprocess"].named_transformers_["numeric"]
    numeric_indices = ridge.named_steps["preprocess"].transformers_[0][2]
    expected_train_means = np.asarray(x_train[:, numeric_indices], dtype=float).mean(axis=0)
    assert np.allclose(numeric.named_steps["scaler"].mean_, expected_train_means)

    wind = np.asarray([[0.0], [90.0], [180.0], [270.0]])
    components = berlin_module._wind_direction_components(wind)
    assert np.allclose(components[:, 0], [0, 1, 0, -1], atol=1e-12)
    assert np.allclose(components[:, 1], [1, 0, -1, 0], atol=1e-12)


def test_baseline_uses_only_train_target_and_models_have_frozen_parameters() -> None:
    dataset = load_berlin_dataset(FIXTURE)
    x_train, x_test, y_train, _ = split_dataset(dataset)
    pipelines = build_model_pipelines()
    baseline = pipelines["baseline_mean"].fit(x_train, y_train)
    assert np.allclose(baseline.predict(x_test), np.mean(y_train))
    assert pipelines["ridge"].named_steps["model"].alpha == 1.0
    nonlinear = pipelines["hist_gradient_boosting"].named_steps["model"]
    assert nonlinear.max_iter == 100
    assert nonlinear.learning_rate == 0.1
    assert nonlinear.max_leaf_nodes == 31
    assert nonlinear.early_stopping is False
    assert nonlinear.random_state == 42


def test_result_is_finite_reproducible_and_contains_required_provenance() -> None:
    first = run_berlin_weather_solar_analysis(FIXTURE, enforce_frozen_source=False)
    second = run_berlin_weather_solar_analysis(FIXTURE, enforce_frozen_source=False)
    assert first.evidence_ids == second.evidence_ids == ()
    assert first.analysis_result_id == second.analysis_result_id
    assert first.analysis_result_id.startswith(f"analysis:{ANALYSIS_NAME}:")
    assert first.values["analysis"] == ANALYSIS_NAME
    assert first.values["dataset"]["rows"] == 24
    assert first.values["split"]["shuffle"] is False
    assert first.values["test_metrics"] == second.values["test_metrics"]
    assert first.values["permutation_importance"] == second.values["permutation_importance"]
    assert len(first.values["permutation_importance"]) == len(RAW_FEATURES)
    assert all(
        np.isfinite(value)
        for metrics in first.values["test_metrics"].values()
        for value in metrics.values()
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item["dataset"].__setitem__("sha256", "b" * 64),
        lambda item: item["features"]["raw"].append("Extra.Feature"),
        lambda item: item["split"].__setitem__("boundary", "2020-01-01T00:00:00"),
        lambda item: item["models"]["ridge"].__setitem__("alpha", 2.0),
        lambda item: item.__setitem__("random_state", 7),
        lambda item: item["preprocessing"].__setitem__("numeric_imputer", "mean"),
    ],
)
def test_identity_changes_with_frozen_scientific_provenance(mutation) -> None:
    original = _identity_provenance()
    changed = deepcopy(original)
    mutation(changed)
    assert analysis_result_id_from_provenance(original) != analysis_result_id_from_provenance(
        changed
    )


def test_identity_ignores_summary_runtime_and_software_execution_provenance() -> None:
    identity = analysis_result_id_from_provenance(_identity_provenance())
    first = berlin_module.AnalysisResult(
        identity,
        "First summary",
        values={
            "analysis": ANALYSIS_NAME,
            "reproducibility": {
                "runtime_seconds": 1.0,
                "numpy_version": "1.26.4",
                "scikit_learn_version": "1.9.0",
            },
        },
    )
    second = berlin_module.AnalysisResult(
        identity,
        "Different wording",
        values={
            "analysis": ANALYSIS_NAME,
            "reproducibility": {
                "runtime_seconds": 99.0,
                "numpy_version": "different",
                "scikit_learn_version": "different",
            },
        },
    )
    assert first.analysis_result_id == second.analysis_result_id == identity


def test_permutation_importance_receives_only_held_out_rows(monkeypatch) -> None:
    captured = {}

    def fake_importance(estimator, x, y, **kwargs):
        captured.update(x=x.copy(), y=y.copy(), kwargs=kwargs)
        return SimpleNamespace(
            importances_mean=np.zeros(len(RAW_FEATURES)),
            importances_std=np.zeros(len(RAW_FEATURES)),
        )

    monkeypatch.setattr(berlin_module, "permutation_importance", fake_importance)
    dataset = load_berlin_dataset(FIXTURE)
    _, x_test, _, y_test = split_dataset(dataset)
    run_berlin_weather_solar_analysis(FIXTURE, enforce_frozen_source=False)
    assert np.array_equal(captured["x"], x_test)
    assert np.array_equal(captured["y"], y_test)
    assert captured["kwargs"] == {
        "scoring": "neg_mean_absolute_error",
        "n_repeats": 5,
        "random_state": 42,
    }


@pytest.mark.parametrize(
    "request_payload",
    [
        {},
        {"analysis": "unknown"},
        {"analysis": ANALYSIS_NAME, "alpha": 2},
        {"analysis": ANALYSIS_NAME, "code": "print('no')"},
        {"analysis": ANALYSIS_NAME, "path": "/tmp/data.csv"},
    ],
)
def test_bounded_tool_rejects_every_request_except_exact_contract(request_payload) -> None:
    tool = BerlinWeatherSolarAnalysisTool(
        MLDatasetConfig(True, ANALYSIS_NAME, FIXTURE)
    )
    with pytest.raises(BerlinSolarAnalysisError, match="unsupported"):
        tool.analyze(request=request_payload, evidence=())


def test_bounded_tool_accepts_exact_request_and_rejects_evidence(monkeypatch) -> None:
    expected = object()
    monkeypatch.setattr(berlin_module, "run_berlin_weather_solar_analysis", lambda path: expected)
    tool = BerlinWeatherSolarAnalysisTool(MLDatasetConfig(True, ANALYSIS_NAME, FIXTURE))
    assert tool.analyze(request={"analysis": ANALYSIS_NAME}, evidence=()) is expected
    with pytest.raises(BerlinSolarAnalysisError, match="does not accept"):
        tool.analyze(request={"analysis": ANALYSIS_NAME}, evidence=(object(),))


def test_missing_dataset_and_frozen_source_mismatch_fail_safely(tmp_path) -> None:
    with pytest.raises(BerlinSolarAnalysisError, match="unavailable"):
        load_berlin_dataset(tmp_path / "missing.csv")
    with pytest.raises(BerlinSolarAnalysisError, match="SHA-256"):
        load_berlin_dataset(FIXTURE, expected_sha256="0" * 64)
