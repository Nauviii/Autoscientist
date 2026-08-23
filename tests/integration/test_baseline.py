"""The three anchors must be honest, reproducible, and correctly ordered."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from dsar.adapters.sandbox import ForkExecutor, run_gauntlet
from dsar.core.contracts import (
    DataContract,
    ExperimentStatus,
    HypothesisStatus,
    TargetDistribution,
    ValidationSpec,
)
from dsar.core.state import SessionBudget, build_state
from dsar.core.statistics import build_power_profile
from dsar.ports import FoldSpec, ResourceLimits
from dsar.stages.baseline import (
    MINIMAL_PREPARATION,
    REFERENCE_ID,
    TRIVIAL_ID,
    TUNED_ID,
    run_baselines,
    trivial_scores,
)
from dsar.modeling.cv import make_folds

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires fork and setrlimit")

LIMITS = ResourceLimits(memory_mb=2048, cpu_seconds=90, wall_seconds=90, max_features=100)
PREVALENCE = 0.28


def make_data(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Mixed types including a numeric column stored as text, plus blanks."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4))
    signal = x @ np.array([1.2, -0.9, 0.7, 0.5])

    frame = pd.DataFrame(x, columns=[f"num_{i}" for i in range(4)])
    category = rng.integers(0, 4, n)
    frame["cat_0"] = category.astype(str)
    signal = signal + 0.8 * category
    frame["num_as_text"] = [
        f"{v:.2f}" if rng.random() > 0.01 else " " for v in rng.normal(size=n)
    ]

    cutoff = np.quantile(signal, 1.0 - PREVALENCE)
    target = pd.Series((signal + rng.normal(size=n) > cutoff).astype(int))
    return frame, target


def make_contract(target: pd.Series, n: int, k: int = 5, repeats: int = 2) -> DataContract:
    validation = ValidationSpec(kind="stratified_kfold", k=k, repeats=repeats, seed=42)
    distribution = TargetDistribution(task="binary", n_rows=n, n_positive=int(target.sum()))
    power = build_power_profile(
        n_rows=n,
        n_pos=int(target.sum()),
        validation=validation,
        metric_family="pr_auc",
        expected_score=0.90,
        expected_auc=0.90,
        delta_practical=0.01,
    )
    return DataContract(
        dataset_hash="test",
        task="binary",
        target="y",
        target_distribution=distribution,
        schema={},
        validation=validation,
        primary_metric="pr_auc",
        metric_family="pr_auc",
        secondary_metrics=("brier",),
        secondary_gates={"brier": 0.02},
        delta_practical=0.01,
        power=power,
        leakage_flags=(),
        excluded_columns=(),
        seed=42,
    )


@pytest.fixture(scope="module")
def fitted() -> tuple:
    frame, target = make_data()
    contract = make_contract(target, len(frame))
    executor = ForkExecutor(frame, target)
    outcome = run_baselines(
        frame=frame,
        target=target,
        contract=contract,
        executor=executor,
        limits=LIMITS,
        environment="test-env",
    )
    return frame, target, contract, executor, outcome


def test_the_preparation_cell_passes_its_own_gauntlet(fitted) -> None:
    """Our own baseline code is held to the same standard as anything an agent writes."""
    frame, _, _, executor, _ = fitted
    fold = FoldSpec(0, 0, tuple(range(1000)), tuple(range(1000, 1200)))
    outcome = run_gauntlet([MINIMAL_PREPARATION], executor, fold, LIMITS)
    assert outcome.passed, outcome.failure


def test_the_trivial_floor_equals_the_prevalence(fitted) -> None:
    """A constant prediction scores exactly the base rate in PR-AUC, by construction."""
    _, target, contract, _, outcome = fitted
    trivial = outcome.by_id[TRIVIAL_ID]
    assert trivial.mean("pr_auc") == pytest.approx(float(target.mean()), abs=0.01)
    assert trivial.mean("roc_auc") == pytest.approx(0.5, abs=0.01)


def test_the_reference_clears_the_floor_decisively(fitted) -> None:
    _, _, _, _, outcome = fitted
    lift = next(e for e in outcome.evidences if e.evidence_id == "EV_reference")
    assert lift.status is HypothesisStatus.SUPPORTED
    assert lift.ci_low > 0


def test_tuning_is_measured_before_any_feature_work(fitted) -> None:
    """Otherwise later feature gains quietly absorb what tuning would have given."""
    _, _, _, _, outcome = fitted
    tuning = next(e for e in outcome.evidences if e.evidence_id == "EV_tuning")
    assert tuning.control_experiment_id == REFERENCE_ID
    assert outcome.chosen_params


def test_the_incumbent_advances_only_on_supported_evidence(fitted) -> None:
    _, _, _, _, outcome = fitted
    tuning = next(e for e in outcome.evidences if e.evidence_id == "EV_tuning")
    expected = TUNED_ID if tuning.status is HypothesisStatus.SUPPORTED else REFERENCE_ID
    assert outcome.incumbent_id == expected


def test_every_baseline_keeps_its_folds(fitted) -> None:
    """Averages cannot be paired later; the per-fold vector is what gets compared."""
    _, _, contract, _, outcome = fitted
    for experiment in outcome.experiments:
        assert len(experiment.fold_scores["pr_auc"]) == contract.validation.n_fits
        assert experiment.status is ExperimentStatus.COMPLETED


def test_baselines_are_distinguishable_by_fingerprint(fitted) -> None:
    _, _, _, _, outcome = fitted
    fingerprints = {e.fingerprint for e in outcome.experiments}
    assert len(fingerprints) == len(outcome.experiments)


def test_the_outcome_feeds_straight_into_the_research_state(fitted) -> None:
    """Baselines are ordinary experiments, so the state layer needs no special case."""
    _, _, contract, _, outcome = fitted
    state, trace = build_state(
        session_id="S1",
        contract=contract,
        experiments=outcome.experiments,
        evidences=outcome.evidences,
        slots=(),
        budget=SessionBudget(experiments=50, tokens=1_000_000, seconds=5400.0),
        reference_id=outcome.reference_id,
    )
    assert state.reference_experiment_id == REFERENCE_ID
    assert state.incumbent_experiment_id == outcome.incumbent_id
    assert trace.promotions == (1 if outcome.incumbent_id == TUNED_ID else 0)


def test_running_twice_gives_identical_scores(fitted) -> None:
    """Determinism has to survive the sandbox, not just the model."""
    frame, target, contract, executor, outcome = fitted
    replay = run_baselines(
        frame=frame,
        target=target,
        contract=contract,
        executor=executor,
        limits=LIMITS,
        environment="test-env",
        tune=False,
    )
    first = outcome.by_id[REFERENCE_ID].fold_scores["pr_auc"]
    second = replay.by_id[REFERENCE_ID].fold_scores["pr_auc"]
    assert first == second


def test_the_trivial_floor_is_computed_without_fitting_anything() -> None:
    frame, target = make_data(n=600, seed=3)
    contract = make_contract(target, len(frame), k=5, repeats=1)
    result = trivial_scores(target, make_folds(contract.validation, target), contract)
    assert result.fit_seconds == 0.0
    assert result.n_features == 0
    assert len(result.fold_scores["pr_auc"]) == 5
