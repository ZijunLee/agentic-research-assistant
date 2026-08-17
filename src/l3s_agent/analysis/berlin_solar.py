"""Frozen Berlin-weather to regional-solar-generation regression case study."""

from __future__ import annotations

import csv
import copy
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ..config import MLDatasetConfig
from ..models import AnalysisResult, Evidence


ANALYSIS_NAME = "berlin_weather_solar_v1"
EXPECTED_RELATIVE_PATH = Path("data/ml/berlin/Berlin_solar_regression.csv")
EXPECTED_SHA256 = "eda6fccb75d8e76d9ae56e806e20fcb12f017041e02d463d60a94817ee5656d8"
EXPECTED_ROWS = 36_296
EXPECTED_COLUMNS = 24
TARGET = "X50Hertz..MW."
SPLIT_BOUNDARY = datetime(2019, 1, 1)
RANDOM_STATE = 42
PERMUTATION_SCORING = "neg_mean_absolute_error"
PERMUTATION_REPEATS = 5

CALENDAR_COLUMNS = ("Year", "Month", "Day", "Hour", "Minute")
RAW_FEATURES = (
    "Temperature",
    "Clearsky.DHI",
    "Clearsky.DNI",
    "Clearsky.GHI",
    "Cloud.Type",
    "Dew.Point",
    "DHI",
    "DNI",
    "GHI",
    "Relative.Humidity",
    "Solar.Zenith.Angle",
    "Surface.Albedo",
    "Pressure",
    "Precipitable.Water",
    "Wind.Direction",
    "Wind.Speed",
)
EXCLUDED_COLUMNS = (*CALENDAR_COLUMNS, "Fill.Flag", "Ozone", TARGET)
EXPECTED_SCHEMA = (
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Temperature",
    "Clearsky.DHI",
    "Clearsky.DNI",
    "Clearsky.GHI",
    "Cloud.Type",
    "Dew.Point",
    "DHI",
    "DNI",
    "Fill.Flag",
    "GHI",
    "Ozone",
    "Relative.Humidity",
    "Solar.Zenith.Angle",
    "Surface.Albedo",
    "Pressure",
    "Precipitable.Water",
    "Wind.Direction",
    "Wind.Speed",
    TARGET,
)
FEATURE_GROUPS: Mapping[str, tuple[str, ...]] = {
    "weather": (
        "Temperature",
        "Cloud.Type",
        "Dew.Point",
        "Relative.Humidity",
        "Pressure",
        "Precipitable.Water",
        "Wind.Direction",
        "Wind.Speed",
    ),
    "observed_irradiance": ("DHI", "DNI", "GHI"),
    "solar_geometry_clear_sky": (
        "Clearsky.DHI",
        "Clearsky.DNI",
        "Clearsky.GHI",
        "Solar.Zenith.Angle",
    ),
    "surface": ("Surface.Albedo",),
}
FEATURE_TO_GROUP = {
    feature: group for group, features in FEATURE_GROUPS.items() for feature in features
}
FEATURE_TRANSFORMS: Mapping[str, Any] = {
    "Wind.Direction": "sin_cos_degrees",
    "Cloud.Type": {
        "encoding": "one_hot",
        "handle_unknown": "ignore",
        "sparse_output": False,
    },
}
PREPROCESSING_SPEC: Mapping[str, Any] = {
    "numeric_imputer": "median",
    "cloud_type_imputer": "most_frequent",
    "wind_direction_imputer": "median",
    "ridge_numeric_scaler": "standard",
    "tree_numeric_scaler": None,
    "target_scaled": False,
    "fit_scope": "train_only",
}
MODEL_SPEC: Mapping[str, Mapping[str, Any]] = {
    "baseline_mean": {"class": "DummyRegressor", "strategy": "mean"},
    "ridge": {"class": "Ridge", "alpha": 1.0},
    "hist_gradient_boosting": {
        "class": "HistGradientBoostingRegressor",
        "max_iter": 100,
        "learning_rate": 0.1,
        "max_leaf_nodes": 31,
        "early_stopping": False,
        "random_state": RANDOM_STATE,
    },
}


class BerlinSolarAnalysisError(RuntimeError):
    """Safe failure raised for invalid data or unsupported analysis requests."""


@dataclass(frozen=True)
class BerlinDataset:
    timestamps: tuple[datetime, ...]
    features: np.ndarray
    target: np.ndarray
    sha256: str
    rows: int
    columns: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analysis_result_id_from_provenance(provenance: Mapping[str, Any]) -> str:
    """Create the stable ID for a versioned deterministic analysis specification."""

    encoded = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"analysis:{ANALYSIS_NAME}:{hashlib.sha256(encoded).hexdigest()}"


def berlin_analysis_identity_provenance(
    *,
    dataset_sha256: str,
    train_rows: int,
    test_rows: int,
    raw_features: Sequence[str] = RAW_FEATURES,
    feature_groups: Mapping[str, Sequence[str]] = FEATURE_GROUPS,
    feature_transforms: Mapping[str, Any] = FEATURE_TRANSFORMS,
    split_boundary: str = SPLIT_BOUNDARY.isoformat(),
    preprocessing: Mapping[str, Any] = PREPROCESSING_SPEC,
    models: Mapping[str, Mapping[str, Any]] = MODEL_SPEC,
    random_state: int = RANDOM_STATE,
    permutation_scoring: str = PERMUTATION_SCORING,
    permutation_repeats: int = PERMUTATION_REPEATS,
) -> Mapping[str, Any]:
    """Return only deterministic scientific/computation identity fields."""

    return {
        "schema_version": 1,
        "analysis": ANALYSIS_NAME,
        "dataset": {"sha256": dataset_sha256, "target": TARGET},
        "features": {
            "raw": list(raw_features),
            "groups": {key: list(value) for key, value in feature_groups.items()},
            "transforms": copy.deepcopy(dict(feature_transforms)),
        },
        "split": {
            "boundary": split_boundary,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "shuffle": False,
        },
        "preprocessing": copy.deepcopy(dict(preprocessing)),
        "models": copy.deepcopy(dict(models)),
        "random_state": random_state,
        "permutation": {
            "scoring": permutation_scoring,
            "repeats": permutation_repeats,
            "random_state": random_state,
        },
    }


def load_berlin_dataset(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_rows: int | None = None,
) -> BerlinDataset:
    """Load, validate, and chronologically order the frozen tabular schema."""

    if not path.is_file():
        raise BerlinSolarAnalysisError("Berlin dataset is unavailable at the configured local path")
    actual_sha256 = _sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise BerlinSolarAnalysisError("Berlin dataset SHA-256 does not match the frozen source")

    records: list[tuple[datetime, list[float], float]] = []
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != EXPECTED_SCHEMA:
                raise BerlinSolarAnalysisError("Berlin dataset schema does not match the frozen schema")
            for row_number, row in enumerate(reader, start=2):
                try:
                    timestamp = datetime(*(int(row[item]) for item in CALENDAR_COLUMNS))
                    feature_values = [float(row[item]) for item in RAW_FEATURES]
                    target = float(row[TARGET])
                except (TypeError, ValueError) as exc:
                    raise BerlinSolarAnalysisError(
                        f"Berlin dataset contains an invalid value at row {row_number}"
                    ) from None
                if not all(math.isfinite(item) for item in (*feature_values, target)):
                    raise BerlinSolarAnalysisError(
                        f"Berlin dataset contains a non-finite value at row {row_number}"
                    )
                records.append((timestamp, feature_values, target))
    except OSError:
        raise BerlinSolarAnalysisError("Berlin dataset could not be read") from None

    if expected_rows is not None and len(records) != expected_rows:
        raise BerlinSolarAnalysisError("Berlin dataset row count does not match the frozen source")
    if not records:
        raise BerlinSolarAnalysisError("Berlin dataset contains no observations")
    records.sort(key=lambda item: item[0])
    timestamps = tuple(item[0] for item in records)
    if len(set(timestamps)) != len(timestamps):
        raise BerlinSolarAnalysisError("Berlin dataset contains duplicate timestamps")
    return BerlinDataset(
        timestamps=timestamps,
        features=np.asarray([item[1] for item in records], dtype=object),
        target=np.asarray([item[2] for item in records], dtype=np.float64),
        sha256=actual_sha256,
        rows=len(records),
        columns=len(EXPECTED_SCHEMA),
    )


def split_dataset(dataset: BerlinDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.asarray([item < SPLIT_BOUNDARY for item in dataset.timestamps])
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise BerlinSolarAnalysisError("Berlin dataset does not span the frozen train/test boundary")
    return (
        dataset.features[train_mask],
        dataset.features[test_mask],
        dataset.target[train_mask],
        dataset.target[test_mask],
    )


def _wind_direction_components(values: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(values, dtype=np.float64).reshape(-1))
    return np.column_stack((np.sin(radians), np.cos(radians)))


def _preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    cloud_index = RAW_FEATURES.index("Cloud.Type")
    wind_index = RAW_FEATURES.index("Wind.Direction")
    numeric_indices = [
        index for index in range(len(RAW_FEATURES)) if index not in {cloud_index, wind_index}
    ]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=(
            ("numeric", Pipeline(numeric_steps), numeric_indices),
            (
                "cloud_type",
                Pipeline(
                    (
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    )
                ),
                [cloud_index],
            ),
            (
                "wind_direction",
                Pipeline(
                    (
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "cyclic",
                            FunctionTransformer(
                                _wind_direction_components,
                                validate=False,
                                feature_names_out=lambda _self, _names: np.asarray(
                                    ["Wind.Direction.sin", "Wind.Direction.cos"],
                                    dtype=object,
                                ),
                            ),
                        ),
                    )
                ),
                [wind_index],
            ),
        ),
        sparse_threshold=0.0,
    )


def build_model_pipelines() -> Mapping[str, Pipeline]:
    return {
        "baseline_mean": Pipeline(
            (("preprocess", _preprocessor(scale_numeric=False)),
             ("model", DummyRegressor(strategy="mean")))
        ),
        "ridge": Pipeline(
            (("preprocess", _preprocessor(scale_numeric=True)),
             ("model", Ridge(alpha=1.0)))
        ),
        "hist_gradient_boosting": Pipeline(
            (
                ("preprocess", _preprocessor(scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=100,
                        learning_rate=0.1,
                        max_leaf_nodes=31,
                        early_stopping=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            )
        ),
    }


def _metrics(target: np.ndarray, prediction: np.ndarray) -> Mapping[str, float]:
    return {
        "mae_mw": float(mean_absolute_error(target, prediction)),
        "rmse_mw": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
    }


def run_berlin_weather_solar_analysis(
    path: Path,
    *,
    enforce_frozen_source: bool = True,
) -> AnalysisResult:
    """Run the fixed experiment once and return structured, non-Evidence results."""

    started = time.perf_counter()
    dataset = load_berlin_dataset(
        path,
        expected_sha256=EXPECTED_SHA256 if enforce_frozen_source else None,
        expected_rows=EXPECTED_ROWS if enforce_frozen_source else None,
    )
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    if enforce_frozen_source and (len(y_train), len(y_test)) != (18_108, 18_188):
        raise BerlinSolarAnalysisError("Berlin train/test counts do not match the frozen split")

    pipelines = build_model_pipelines()
    test_metrics: dict[str, Mapping[str, float]] = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        test_metrics[name] = _metrics(y_test, pipeline.predict(x_test))

    importance = permutation_importance(
        pipelines["hist_gradient_boosting"],
        x_test,
        y_test,
        scoring=PERMUTATION_SCORING,
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
    )
    importance_rows = [
        {
            "feature": feature,
            "group": FEATURE_TO_GROUP[feature],
            "mean": float(importance.importances_mean[index]),
            "std": float(importance.importances_std[index]),
        }
        for index, feature in enumerate(RAW_FEATURES)
    ]
    importance_rows.sort(key=lambda item: (-item["mean"], item["feature"]))

    train_mean = float(np.mean(y_train))
    test_mean = float(np.mean(y_test))
    relative_difference = (test_mean - train_mean) / train_mean
    timestamp_strings = [item.isoformat() for item in dataset.timestamps]
    values: dict[str, Any] = {
        "schema_version": 1,
        "analysis": ANALYSIS_NAME,
        "dataset": {
            "relative_path": EXPECTED_RELATIVE_PATH.as_posix(),
            "sha256": dataset.sha256,
            "rows": dataset.rows,
            "columns": dataset.columns,
        },
        "task": {
            "framing": (
                "Prediction of observed 50Hertz regional solar generation from "
                "contemporaneously aligned Berlin-area weather and solar-irradiance conditions"
            ),
            "target": TARGET,
            "not_future_forecasting": True,
            "not_causal": True,
            "not_site_level": True,
        },
        "split": {
            "boundary": SPLIT_BOUNDARY.isoformat(),
            "train_date_range": [timestamp_strings[0], timestamp_strings[len(y_train) - 1]],
            "test_date_range": [timestamp_strings[len(y_train)], timestamp_strings[-1]],
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "shuffle": False,
        },
        "features": {
            "raw": list(RAW_FEATURES),
            "excluded": list(EXCLUDED_COLUMNS),
            "groups": {key: list(value) for key, value in FEATURE_GROUPS.items()},
            "transforms": copy.deepcopy(dict(FEATURE_TRANSFORMS)),
            "transformed_feature_notes": [
                "Wind.Direction is replaced by sine and cosine components.",
                "Cloud.Type is one-hot encoded with unknown categories ignored.",
                "All learned preprocessing is fitted on the training partition only.",
            ],
        },
        "preprocessing": copy.deepcopy(dict(PREPROCESSING_SPEC)),
        "models": copy.deepcopy(dict(MODEL_SPEC)),
        "test_metrics": test_metrics,
        "target_shift": {
            "train_mean_mw": train_mean,
            "test_mean_mw": test_mean,
            "absolute_mean_difference_mw": abs(test_mean - train_mean),
            "relative_mean_difference": relative_difference,
        },
        "permutation_importance": importance_rows,
        "limitations": [
            "The table contains daylight-only sampling rather than a complete 24-hour series.",
            "Fill.Flag indicates upstream filling for some meteorological observations.",
            "The timestamp timezone and cross-source alignment are undocumented locally.",
            "Berlin-area conditions are paired with regional 50Hertz generation, not site-level PV.",
            "The train and test years have a material target-mean shift.",
            "Correlated irradiance predictors can redistribute permutation importance.",
            "Predictive performance and importance are not causal weather-effect estimates.",
        ],
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "permutation_scoring": PERMUTATION_SCORING,
            "permutation_repeats": PERMUTATION_REPEATS,
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    }
    identity_provenance = berlin_analysis_identity_provenance(
        dataset_sha256=dataset.sha256,
        train_rows=len(y_train),
        test_rows=len(y_test),
    )
    return AnalysisResult(
        analysis_result_id=analysis_result_id_from_provenance(identity_provenance),
        summary=(
            "Completed the frozen contemporaneous Berlin-weather to regional 50Hertz "
            "solar-generation regression experiment."
        ),
        values=values,
        evidence_ids=(),
    )


class BerlinWeatherSolarAnalysisTool:
    """Research-Agent-facing wrapper for exactly one predefined local analysis."""

    def __init__(self, dataset: MLDatasetConfig) -> None:
        if not dataset.approved or dataset.adapter != ANALYSIS_NAME or dataset.path is None:
            raise BerlinSolarAnalysisError("Berlin analysis requires the approved dataset config")
        self._path = dataset.path

    def analyze(
        self,
        *,
        request: Mapping[str, Any],
        evidence: Sequence[Evidence],
    ) -> AnalysisResult:
        if evidence:
            raise BerlinSolarAnalysisError("Berlin analysis does not accept paper Evidence inputs")
        if dict(request) != {"analysis": ANALYSIS_NAME}:
            raise BerlinSolarAnalysisError("unsupported bounded scientific-analysis request")
        return run_berlin_weather_solar_analysis(self._path)
