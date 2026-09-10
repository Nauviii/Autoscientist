"""Model construction. Config only: agents never write model code.

Every family here is pinned to its deterministic settings. Left at defaults, all
three produce results that shift in the last digits when the thread count changes,
because floating point reduction order changes with it. That would quietly break
the fingerprint contract, where an equal fingerprint must mean an equal score.

LightGBM's deterministic mode costs roughly 10-20% throughput. At the dataset sizes
this system targets that is a few seconds per experiment, which is a trade worth
making without hesitation.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.contracts import ModelConfig, Task

Family = Literal["lightgbm", "xgboost", "catboost"]

# Environment variables that must be set before numpy or any BLAS-backed library is
# imported, otherwise reduction order stays outside our control.
DETERMINISM_ENV: Mapping[str, str] = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
}


def apply_determinism_env(n_threads: int = 4) -> None:
    """Pin thread counts. Call before importing numeric libraries, not after."""
    for key, value in DETERMINISM_ENV.items():
        os.environ.setdefault(key, str(n_threads) if key.endswith("NUM_THREADS") else value)


def _lightgbm_flags(seed: int, n_threads: int) -> dict[str, Any]:
    return {
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": n_threads,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "verbose": -1,
    }


def _xgboost_flags(seed: int, n_threads: int) -> dict[str, Any]:
    return {
        "tree_method": "hist",
        "nthread": n_threads,
        "random_state": seed,
        "verbosity": 0,
    }


def _catboost_flags(seed: int, n_threads: int) -> dict[str, Any]:
    return {
        "random_seed": seed,
        "thread_count": n_threads,
        "verbose": False,
        "allow_writing_files": False,
    }


DETERMINISM_FLAGS = {
    "lightgbm": _lightgbm_flags,
    "xgboost": _xgboost_flags,
    "catboost": _catboost_flags,
}


def default_params(family: Family, task: Task) -> dict[str, Any]:
    """Untuned starting points, deliberately modest so the reference stays honest."""
    shared = {"n_estimators": 300, "learning_rate": 0.05}
    if family == "lightgbm":
        return {**shared, "num_leaves": 31, "min_child_samples": 20}
    if family == "xgboost":
        return {**shared, "max_depth": 6, "min_child_weight": 1}
    return {"iterations": 300, "learning_rate": 0.05, "depth": 6}


def build_model(config: ModelConfig, task: Task, seed: int, n_classes: int = 2) -> Any:
    """Instantiate an estimator with determinism flags overriding any conflict."""
    params = {**config.params, **DETERMINISM_FLAGS[config.family](seed, config.n_threads)}

    if config.family == "lightgbm":
        import lightgbm as lgb

        if task == "regression":
            return lgb.LGBMRegressor(**params)
        params.setdefault("objective", "binary" if task == "binary" else "multiclass")
        if task == "multiclass":
            params.setdefault("num_class", n_classes)
        return lgb.LGBMClassifier(**params)

    if config.family == "xgboost":
        import xgboost as xgb

        params.setdefault("enable_categorical", True)
        if task == "regression":
            return xgb.XGBRegressor(**params)
        return xgb.XGBClassifier(**params)

    from catboost import CatBoostClassifier, CatBoostRegressor

    if task == "regression":
        return CatBoostRegressor(**params)
    return CatBoostClassifier(**params)


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    """Columns the boosters can consume natively once cast to the category dtype."""
    return [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]


def align_categories(train: pd.DataFrame, validate: pd.DataFrame) -> pd.DataFrame:
    """Give the validation fold the training fold's category levels.

    Unseen levels become missing rather than shifting every code by one, which
    would otherwise scramble the mapping the model learned.
    """
    out = validate.copy()
    for column in categorical_columns(train):
        if column in out.columns:
            out[column] = out[column].astype(
                pd.CategoricalDtype(categories=train[column].cat.categories)
            )
    return out


def fit_predict(
    config: ModelConfig,
    task: Task,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    seed: int,
    classes: Sequence[Any] | None = None,
) -> np.ndarray:
    """Fit on one fold and return scores in the shape the metrics expect.

    Binary returns the positive-class probability, multiclass the full matrix,
    regression the point prediction.
    """
    n_classes = len(classes) if classes is not None else 2
    model = build_model(config, task, seed, n_classes)
    valid_x = align_categories(train_x, valid_x)

    model.fit(train_x, train_y)
    if task == "regression":
        return np.asarray(model.predict(valid_x))

    proba = np.asarray(model.predict_proba(valid_x))
    return proba[:, 1] if task == "binary" else proba