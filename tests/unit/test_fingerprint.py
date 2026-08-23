"""Fingerprints must collapse cosmetic rewrites and separate real differences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsar.core.contracts import ModelConfig, ValidationSpec
from dsar.core.fingerprint import (
    canonical_source,
    cell_fingerprint,
    environment_fingerprint,
    experiment_fingerprint,
    model_fingerprint,
    output_signature,
    pipeline_fingerprint,
    pipeline_prefixes,
    prompt_fingerprint,
    quantise_params,
    shared_prefix_length,
    validation_fingerprint,
)

BASE = '''
class FeatureStep:
    """Fill the gap with the training median."""

    def fit(self, X, y):
        self.median_ = X["num_1"].median()
        return self

    def transform(self, X):
        out = X.copy()
        out["num_1"] = out["num_1"].fillna(self.median_)
        return out
'''

RENAMED = '''
class FeatureStep:
    """A different docstring entirely."""

    def fit(self, X, y):
        self.fill_value = X["num_1"].median()
        return self

    def transform(self, X):
        frame = X.copy()
        frame["num_1"] = frame["num_1"].fillna(self.fill_value)
        return frame
'''

ANNOTATED = '''
import pandas as pd


class FeatureStep:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureStep":
        self.median_: float = X["num_1"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out: pd.DataFrame = X.copy()
        out["num_1"] = out["num_1"].fillna(self.median_)
        return out
'''

DIFFERENT_STATISTIC = BASE.replace(".median()", ".mean()")
DIFFERENT_COLUMN = BASE.replace("num_1", "num_0")

SECOND_CELL = '''
class FeatureStep:
    def fit(self, X, y):
        self.q_ = X["num_0"].quantile(0.5)
        return self

    def transform(self, X):
        out = X.copy()
        out["above"] = (out["num_0"] > self.q_).astype(int)
        return out
'''


def test_local_renaming_and_docstrings_do_not_change_the_hash() -> None:
    assert cell_fingerprint(BASE) == cell_fingerprint(RENAMED)


def test_annotations_do_not_change_the_hash() -> None:
    """Only the import line differs after canonicalisation, so hashes must still split."""
    assert "pd.DataFrame" not in canonical_source(ANNOTATED)
    assert "->" not in canonical_source(ANNOTATED)


def test_canonical_form_drops_docstrings_and_local_names() -> None:
    canonical = canonical_source(BASE)
    assert "median_" not in canonical
    assert "Fill the gap" not in canonical
    assert "self" in canonical and "X" in canonical


def test_a_different_statistic_is_a_different_cell() -> None:
    assert cell_fingerprint(BASE) != cell_fingerprint(DIFFERENT_STATISTIC)


def test_a_different_column_is_a_different_cell() -> None:
    """String constants carry meaning and must survive canonicalisation."""
    assert cell_fingerprint(BASE) != cell_fingerprint(DIFFERENT_COLUMN)


def test_imported_names_survive_renaming() -> None:
    """Collapsing library names would make different libraries hash alike."""
    numpy_cell = "import numpy\n\n\nclass FeatureStep:\n    def fit(self, X, y):\n        self.v_ = numpy.median(X['num_0'])\n        return self\n\n    def transform(self, X):\n        return X\n"
    assert "numpy" in canonical_source(numpy_cell)


def test_pipeline_order_is_significant() -> None:
    assert pipeline_fingerprint([BASE, SECOND_CELL]) != pipeline_fingerprint([SECOND_CELL, BASE])


def test_prefixes_are_stable_when_the_tail_changes() -> None:
    """A shared head can then be reused from cache instead of refitted."""
    left = pipeline_prefixes([BASE, SECOND_CELL])
    right = pipeline_prefixes([BASE, DIFFERENT_STATISTIC])
    assert left[0] == right[0]
    assert left[1] != right[1]


def test_shared_prefix_length_counts_the_reusable_head() -> None:
    assert shared_prefix_length([BASE, SECOND_CELL], [BASE, DIFFERENT_STATISTIC]) == 1
    assert shared_prefix_length([BASE, SECOND_CELL], [BASE, SECOND_CELL]) == 2
    assert shared_prefix_length([BASE], [SECOND_CELL]) == 0


def test_float_noise_in_hyperparameters_is_quantised_away() -> None:
    assert quantise_params({"lr": 0.1}) == quantise_params({"lr": 0.10000000000000003})
    assert quantise_params({"lr": 0.1}) != quantise_params({"lr": 0.11})


def test_thread_count_is_part_of_model_identity() -> None:
    """Reduction order shifts the last digits, so a cached score would not reproduce."""
    four = ModelConfig(family="lightgbm", params={"num_leaves": 31}, n_threads=4)
    eight = ModelConfig(family="lightgbm", params={"num_leaves": 31}, n_threads=8)
    assert model_fingerprint(four) != model_fingerprint(eight)


def test_parameter_order_does_not_affect_model_identity() -> None:
    a = ModelConfig(family="lightgbm", params={"num_leaves": 31, "lr": 0.05})
    b = ModelConfig(family="lightgbm", params={"lr": 0.05, "num_leaves": 31})
    assert model_fingerprint(a) == model_fingerprint(b)


def test_validation_seed_changes_the_protocol_identity() -> None:
    a = ValidationSpec(kind="stratified_kfold", k=10, repeats=5, seed=42)
    b = ValidationSpec(kind="stratified_kfold", k=10, repeats=5, seed=43)
    assert validation_fingerprint(a) != validation_fingerprint(b)


def test_environment_fingerprint_tracks_the_lockfile() -> None:
    assert environment_fingerprint(b"lock-a", "3.12.3") != environment_fingerprint(
        b"lock-b", "3.12.3"
    )
    assert environment_fingerprint(b"lock-a", "3.12.3") != environment_fingerprint(
        b"lock-a", "3.13.0"
    )


def test_prompt_edits_are_visible_in_the_fingerprint() -> None:
    """Editing a template makes later experiments incomparable, so it must be tracked."""
    assert prompt_fingerprint({"propose": "v1"}) != prompt_fingerprint({"propose": "v2"})


def make_experiment(**overrides) -> str:
    defaults = dict(
        cells=[BASE],
        model=ModelConfig(family="lightgbm", params={"num_leaves": 31}),
        validation=ValidationSpec(kind="stratified_kfold", k=10, repeats=5, seed=42),
        data_hash="d0",
        environment="e0",
        seed=42,
    )
    return experiment_fingerprint(**{**defaults, **overrides})


def test_experiment_fingerprint_is_stable_under_cosmetic_rewrites() -> None:
    assert make_experiment() == make_experiment(cells=[RENAMED])


@pytest.mark.parametrize(
    "overrides",
    [
        {"cells": [DIFFERENT_STATISTIC]},
        {"model": ModelConfig(family="xgboost", params={"num_leaves": 31})},
        {"validation": ValidationSpec(kind="stratified_kfold", k=5, repeats=5, seed=42)},
        {"data_hash": "d1"},
        {"environment": "e1"},
        {"seed": 7},
    ],
)
def test_every_input_participates_in_the_experiment_fingerprint(overrides: dict) -> None:
    assert make_experiment(**overrides) != make_experiment()


def make_frame(seed: int = 0, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num_0": rng.normal(size=n),
            "num_1": rng.normal(size=n),
            "cat_0": rng.integers(0, 3, size=n).astype(str),
        }
    )


def test_output_signature_ignores_column_order() -> None:
    frame = make_frame()
    assert output_signature(frame) == output_signature(frame[["cat_0", "num_1", "num_0"]])


def test_output_signature_ignores_column_naming() -> None:
    """Different code producing identical features must collapse to one signature."""
    frame = make_frame()
    renamed = frame.rename(columns={"num_0": "feature_a", "num_1": "feature_b"})
    assert output_signature(frame) == output_signature(renamed)


def test_output_signature_reacts_to_content() -> None:
    frame = make_frame()
    changed = frame.copy()
    changed.loc[0, "num_0"] += 1.0
    assert output_signature(frame) != output_signature(changed)


def test_output_signature_reacts_to_missingness() -> None:
    frame = make_frame()
    holed = frame.copy()
    holed.loc[0, "num_1"] = np.nan
    assert output_signature(frame) != output_signature(holed)


def test_output_signature_reacts_to_an_added_feature() -> None:
    frame = make_frame()
    widened = frame.copy()
    widened["extra"] = 1.0
    assert output_signature(frame) != output_signature(widened)
