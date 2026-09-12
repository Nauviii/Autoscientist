"""EDA must adapt to the dataset, catch planted leakage, and never guess silently."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsar.core.contracts import Severity
from dsar.stages.eda import (
    _cramers_v,
    block_a_integrity,
    coerce_datetime,
    classify_column,
    coerce_numeric,
    choose_metric,
    dataset_hash,
    encode_target,
    exploration_index,
    infer_task,
    prepared_frame,
    run_eda,
)

N = 2500


def base_frame(seed: int = 0, n: int = N) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    frame = pd.DataFrame(x, columns=[f"num_{i}" for i in range(4)])
    frame["cat_0"] = rng.integers(0, 4, n).astype(str)
    signal = x @ np.array([1.0, -0.8, 0.6, 0.4])
    return frame, signal


def binary_frame(prevalence: float = 0.10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame, signal = base_frame(seed)
    label = (signal + rng.normal(size=len(frame)) > np.quantile(signal, 1 - prevalence)).astype(int)
    frame["Churn"] = np.where(label == 1, "Yes", "No")
    return frame


def test_task_inference_covers_the_three_shapes() -> None:
    assert infer_task(pd.Series([0, 1, 1, 0])) == "binary"
    assert infer_task(pd.Series(["a", "b", "c", "a"])) == "multiclass"
    assert infer_task(pd.Series(np.linspace(0, 100, 500))) == "regression"


def test_text_targets_are_encoded_by_meaning_not_by_order() -> None:
    """Yes is the positive class regardless of where it sorts."""
    encoded = encode_target(pd.Series(["Yes", "No", "No", "Yes"]), "binary")
    assert list(encoded) == [1, 0, 0, 1]


def test_the_dataset_hash_tracks_content() -> None:
    frame = binary_frame()
    assert dataset_hash(frame) == dataset_hash(frame.copy())
    changed = frame.copy()
    changed.loc[0, "num_0"] += 1.0
    assert dataset_hash(frame) != dataset_hash(changed)


def test_integrity_reports_structural_problems() -> None:
    frame = binary_frame()
    frame["constant"] = 1
    frame["empty"] = np.nan
    frame = pd.concat([frame, frame.head(3)], ignore_index=True)

    integrity = block_a_integrity(frame)
    assert integrity.duplicate_rows == 3
    assert "constant" in integrity.constant_columns
    assert "empty" in integrity.empty_columns


def test_the_exploration_slice_is_deterministic_and_stratified() -> None:
    """Profiling on a reserved slice keeps validation rows out of the agent's view."""
    frame = binary_frame(prevalence=0.05)
    target = encode_target(frame["Churn"], "binary")
    first = exploration_index(target, 0.15, seed=42)
    second = exploration_index(target, 0.15, seed=42)

    assert np.array_equal(first, second)
    assert target.iloc[first].nunique() == 2


def test_an_id_column_is_excluded_outright() -> None:
    frame = binary_frame()
    frame["record_key"] = [f"K{i:06d}" for i in range(len(frame))]
    artifact, contract = run_eda(frame=frame, target_column="Churn")

    assert "record_key" in contract.excluded_columns
    assert "record_key" not in contract.schema
    assert any(f.severity is Severity.BLOCK for f in artifact.blocking)


def test_continuous_floats_are_not_mistaken_for_keys() -> None:
    """Near-unique is normal for a float; dropping them would discard real features."""
    _, contract = run_eda(frame=binary_frame(), target_column="Churn")
    assert {"num_0", "num_1", "num_2", "num_3"} <= set(contract.schema)


def test_a_planted_post_outcome_column_is_flagged() -> None:
    frame = binary_frame()
    label = (frame["Churn"] == "Yes").to_numpy()
    rng = np.random.default_rng(1)
    frame["settled_after"] = np.where(label, rng.normal(5, 0.3, len(frame)), rng.normal(0, 0.3, len(frame)))

    artifact, _ = run_eda(frame=frame, target_column="Churn")
    flagged = {f.column: f for f in artifact.leakage if f.test == "single_column_power"}
    assert "settled_after" in flagged
    assert flagged["settled_after"].statistic > 0.95


def test_informative_missingness_is_flagged() -> None:
    """Rarely tested, and usually means the column was filled in after the outcome."""
    frame = binary_frame()
    label = (frame["Churn"] == "Yes").to_numpy()
    rng = np.random.default_rng(2)
    frame["filled_later"] = np.where(label, rng.normal(size=len(frame)), np.nan)

    artifact, _ = run_eda(frame=frame, target_column="Churn")
    tests = {(f.column, f.test) for f in artifact.leakage}
    assert ("filled_later", "informative_missingness") in tests


def test_duplicate_rows_are_flagged_as_a_fold_hazard() -> None:
    frame = binary_frame()
    frame = pd.concat([frame, frame.head(20)], ignore_index=True)
    artifact, _ = run_eda(frame=frame, target_column="Churn")
    assert any(f.test == "duplicate_rows" for f in artifact.leakage)


def test_a_clean_dataset_raises_no_warnings() -> None:
    """The screen must stay quiet when there is nothing to say."""
    artifact, _ = run_eda(frame=binary_frame(prevalence=0.35), target_column="Churn")
    assert artifact.to_confirm == ()


def test_a_monotonic_date_column_selects_temporal_validation() -> None:
    frame, signal = base_frame(seed=3)
    frame["event_time"] = pd.date_range("2024-01-01", periods=len(frame), freq="h")
    frame["y"] = (signal > 0).astype(int)

    artifact, contract = run_eda(frame=frame, target_column="y")
    assert contract.validation.kind == "time_series"
    assert contract.validation.time_column == "event_time"
    assert artifact.structure.ordered_by_time


def test_a_repeated_key_is_surfaced_for_confirmation_not_assumed() -> None:
    """A grouping key and a binned measurement look identical, so the human decides."""
    rng = np.random.default_rng(4)
    frame, signal = base_frame(seed=4)
    frame["entity"] = rng.integers(0, 200, len(frame)).astype(str)
    frame["y"] = (signal > 0).astype(int)

    artifact, contract = run_eda(frame=frame, target_column="y")
    assert artifact.structure.requires_confirmation
    assert "entity" in artifact.structure.group_candidates
    assert contract.validation.kind == "stratified_kfold"
    assert any("grouping key" in w for w in artifact.warnings)


def test_a_continuous_measurement_is_not_a_grouping_candidate() -> None:
    frame, signal = base_frame(seed=6)
    frame["y"] = (signal > 0).astype(int)
    artifact, _ = run_eda(frame=frame, target_column="y")
    assert artifact.structure.group_candidates == ()
    assert not artifact.structure.requires_confirmation


def test_regression_falls_back_to_plain_kfold() -> None:
    frame, _ = base_frame(seed=5)
    frame = frame.drop(columns=["cat_0"])
    frame["price"] = frame.to_numpy() @ np.array([2.0, 1.0, -1.0, 0.5])

    artifact, contract = run_eda(frame=frame, target_column="price")
    assert contract.task == "regression"
    assert contract.validation.kind == "kfold"
    assert contract.primary_metric == "r2"
    assert artifact.warnings == ()


@pytest.mark.parametrize(
    "prevalence,expected",
    [(0.08, "pr_auc"), (0.30, "pr_auc"), (0.50, "roc_auc")],
)
def test_metric_choice_follows_the_imbalance(prevalence: float, expected: str) -> None:
    _, contract = run_eda(frame=binary_frame(prevalence=prevalence), target_column="Churn")
    assert contract.primary_metric == expected


def test_multiclass_gets_a_macro_metric() -> None:
    from dsar.core.contracts import TargetDistribution

    distribution = TargetDistribution(task="multiclass", n_rows=100, class_counts={"a": 50, "b": 50})
    primary, family, _ = choose_metric("multiclass", distribution)
    assert primary == "macro_f1" and family == "accuracy"


def test_rare_positives_make_the_dataset_underpowered() -> None:
    """The warning has to appear before any experiment claims a small gain."""
    artifact, contract = run_eda(frame=binary_frame(prevalence=0.02), target_column="Churn")
    assert contract.power.underpowered
    assert any("resolves" in w for w in artifact.warnings)


def test_delta_min_is_floored_by_what_the_data_can_resolve() -> None:
    _, contract = run_eda(frame=binary_frame(), target_column="Churn", delta_practical=0.001)
    assert contract.delta_min == contract.power.mde > 0.001


def test_the_contract_fingerprint_is_stable() -> None:
    frame = binary_frame()
    _, first = run_eda(frame=frame, target_column="Churn")
    _, second = run_eda(frame=frame, target_column="Churn")
    assert first.fingerprint() == second.fingerprint()


def test_prepared_frame_applies_the_contract_exclusions() -> None:
    frame = binary_frame()
    frame["record_key"] = [f"K{i:06d}" for i in range(len(frame))]
    _, contract = run_eda(frame=frame, target_column="Churn")

    features, target = prepared_frame(frame, contract)
    assert "record_key" not in features.columns
    assert "Churn" not in features.columns
    assert set(target.unique()) == {0, 1}


def test_numbers_stored_as_text_are_read_as_numeric() -> None:
    """Left as categorical they gain thousands of levels and trip the leakage screen."""
    rng = np.random.default_rng(0)
    values = rng.gamma(2, 500, 800).round(2).astype(str)
    values[rng.choice(800, 6, replace=False)] = " "

    spec = classify_column("charges", pd.Series(values))
    assert spec.role == "numeric"
    assert spec.missing_rate == pytest.approx(6 / 800, abs=1e-6)
    assert spec.numeric_summary is not None


def test_genuinely_categorical_text_stays_categorical() -> None:
    spec = classify_column("plan", pd.Series(["basic", "premium", "basic"] * 100))
    assert spec.role == "categorical"
    assert coerce_numeric(pd.Series(["basic", "premium"])) is None


def test_association_is_bias_corrected_for_many_levels() -> None:
    """Uncorrected, a near-unique column looks like perfect leakage on any dataset."""
    rng = np.random.default_rng(1)
    target = pd.Series(rng.integers(0, 2, 800))
    near_unique = pd.Series([f"v{i}" for i in range(800)])

    assert _cramers_v(near_unique, target) < 0.10
    assert _cramers_v(target.astype(str), target) == pytest.approx(1.0, abs=1e-6)


def test_a_text_numeric_column_is_not_flagged_as_leakage() -> None:
    """The regression this pair of fixes was written for."""
    frame = binary_frame()
    rng = np.random.default_rng(2)
    charges = (frame["num_0"].abs() * 100 + rng.gamma(2, 50, len(frame))).round(2).astype(str)
    charges.iloc[:8] = " "
    frame["total_charges"] = charges

    artifact, contract = run_eda(frame=frame, target_column="Churn")
    assert contract.schema["total_charges"].role == "numeric"
    assert not any(
        f.column == "total_charges" and f.test == "near_perfect_association"
        for f in artifact.leakage
    )


def test_dates_stored_as_text_are_detected() -> None:
    """CSV readers hand back dates as strings, so the temporal path needs parsing."""
    stamps = pd.Series(pd.date_range("2024-01-01", periods=200, freq="D").astype(str))
    assert coerce_datetime(stamps) is not None
    assert classify_column("event_day", stamps).role == "datetime"


def test_a_bare_number_is_not_read_as_a_date() -> None:
    """Numeric parsing runs first, or a year column would become a timestamp."""
    assert classify_column("year", pd.Series([2011, 2012] * 100)).role == "numeric"
    assert coerce_datetime(pd.Series(["1.5", "2.5"] * 50)) is None


def test_a_text_date_column_drives_temporal_validation() -> None:
    rng = np.random.default_rng(8)
    frame, signal = base_frame(seed=8)
    frame["event_day"] = pd.date_range(
        "2024-01-01", periods=len(frame), freq="h"
    ).astype(str)
    frame["y"] = signal + rng.normal(size=len(frame))

    artifact, contract = run_eda(frame=frame, target_column="y")
    assert contract.validation.kind == "time_series"
    assert contract.validation.time_column == "event_day"
    assert "event_day" not in artifact.structure.group_candidates


def test_an_integer_row_counter_is_treated_as_an_identifier() -> None:
    """It parses as a number, so checking type before identity would let it through."""
    frame = binary_frame()
    frame.insert(0, "instant", range(1, len(frame) + 1))
    _, contract = run_eda(frame=frame, target_column="Churn")
    assert "instant" in contract.excluded_columns


def test_a_component_of_a_continuous_target_is_flagged() -> None:
    """A depth-2 stump cannot express this, which is why regression gets more depth."""
    rng = np.random.default_rng(9)
    frame, signal = base_frame(seed=9)
    frame = frame.drop(columns=["cat_0"])
    noise = rng.normal(scale=0.2, size=len(frame))
    frame["major_part"] = signal * 0.8
    frame["target"] = frame["major_part"] + signal * 0.2 + noise

    artifact, _ = run_eda(frame=frame, target_column="target")
    flagged = {f.column for f in artifact.leakage if f.test == "single_column_power"}
    assert "major_part" in flagged


def test_key_detection_reads_the_whole_frame_not_the_slice() -> None:
    """On a sixth of the rows a key with a dozen members looks like singletons.

    Computing the signal from the exploration slice would invert it, which is how
    this went unnoticed until a real dataset surfaced it.
    """
    from dsar.stages.eda import GROUP_SINGLETON_LIMIT, _singleton_ratio

    rng = np.random.default_rng(11)
    entity = pd.Series(rng.integers(0, 200, 2500).astype(str))
    slice_of_it = entity.iloc[rng.choice(2500, 375, replace=False)]

    assert _singleton_ratio(entity) < GROUP_SINGLETON_LIMIT
    assert _singleton_ratio(slice_of_it) > GROUP_SINGLETON_LIMIT


def test_area_measurements_are_not_mistaken_for_keys() -> None:
    """Square footage repeats and is integral, but leaves most values singletons."""
    from dsar.stages.eda import _looks_like_key

    rng = np.random.default_rng(12)
    n = 2930
    floor = n ** 0.6

    area = pd.Series(np.where(rng.random(n) < 0.6, 0, rng.integers(16, 1600, n)))
    finished = pd.Series(rng.integers(0, 1500, n))
    entity = pd.Series(rng.integers(0, n // 12, n))

    assert not _looks_like_key(area, floor)
    assert not _looks_like_key(finished, floor)
    assert _looks_like_key(entity, floor)


def test_a_settled_grouping_question_is_not_asked_again() -> None:
    """Answering in the config should stop the prompt reappearing on every run."""
    from dsar.core.dossier import Dossier

    rng = np.random.default_rng(13)
    frame, signal = base_frame(seed=13)
    frame["entity"] = rng.integers(0, 200, len(frame)).astype(str)
    frame["y"] = (signal > 0).astype(int)

    unanswered, _ = run_eda(frame=frame, target_column="y")
    assert any("grouping key" in w for w in unanswered.warnings)

    answered, _ = run_eda(
        frame=frame, target_column="y", dossier=Dossier(grouping_confirmed=True)
    )
    assert not any("grouping key" in w for w in answered.warnings)


def test_naming_a_group_column_switches_the_scheme() -> None:
    from dsar.core.dossier import Dossier

    rng = np.random.default_rng(14)
    frame, signal = base_frame(seed=14)
    frame["entity"] = rng.integers(0, 200, len(frame)).astype(str)
    frame["y"] = (signal > 0).astype(int)

    _, contract = run_eda(
        frame=frame, target_column="y", dossier=Dossier(group_column="entity")
    )
    assert contract.validation.kind == "group_kfold"
    assert contract.validation.group_column == "entity"