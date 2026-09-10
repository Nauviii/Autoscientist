"""Run the full pipeline over every contract shape, not just the one we develop on.

The unit tests check that EDA reaches the right decision for each shape. This file
checks that the decision can actually be executed: folds generated, cells run in the
sandbox, a model fitted, metrics computed, evidence produced.

That gap is where the last three bugs lived. All of them were found by the real
dataset rather than by the synthetic cases, because the synthetic cases had only ever
exercised the reasoning and never the execution.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pytest

from dsar.adapters.sandbox import ForkExecutor
from dsar.core.contracts import ExperimentStatus
from dsar.core.state import SessionBudget, build_state
from dsar.ports import ResourceLimits
from dsar.stages.baseline import REFERENCE_ID, TRIVIAL_ID, run_baselines
from dsar.stages.eda import prepared_frame, run_eda
from tests.synthetic import SyntheticCase, contract_cases

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires fork and setrlimit")

# Small protocol: this file checks that each shape runs, not how well it scores.
K = 3
REPEATS = 1
LIMITS = ResourceLimits(memory_mb=2048, cpu_seconds=120, wall_seconds=120, max_features=600)

CASES = {case.name: case for case in contract_cases()}


@lru_cache(maxsize=None)
def execute(case_name: str):
    """Run EDA and the baselines end to end for one contract shape.

    Cached because every assertion below needs the same result and a rerun costs
    two model fits per fold.
    """
    case = CASES[case_name]
    artifact, contract = run_eda(
        frame=case.frame,
        target_column=case.target,
        k=K,
        repeats=REPEATS,
        seed=7,
    )
    features, target = prepared_frame(case.frame, contract)
    executor = ForkExecutor(features, target)
    outcome = run_baselines(
        frame=features,
        target=target,
        contract=contract,
        executor=executor,
        limits=LIMITS,
        environment="shapes",
        tune=False,
    )
    return artifact, contract, outcome


@pytest.mark.parametrize("name", sorted(CASES))
def test_every_contract_shape_runs_end_to_end(name: str) -> None:
    """No shape may reach a decision it cannot then execute."""
    artifact, contract, outcome = execute(name)

    assert len(outcome.experiments) == 2
    for experiment in outcome.experiments:
        assert experiment.status is ExperimentStatus.COMPLETED
        scores = experiment.fold_scores[contract.primary_metric]
        assert len(scores) == contract.validation.n_fits
        assert all(value == value for value in scores)


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_reference_clears_the_trivial_floor(name: str) -> None:
    """A booster on real signal must beat a constant prediction on every shape."""
    _, contract, outcome = execute(name)
    metric = contract.primary_metric
    trivial = outcome.by_id[TRIVIAL_ID].mean(metric)
    reference = outcome.by_id[REFERENCE_ID].mean(metric)

    if metric in {"brier", "logloss", "rmse", "mae"}:
        assert reference <= trivial
    else:
        assert reference >= trivial


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_declared_validation_scheme_is_the_one_used(name: str) -> None:
    case = CASES[name]
    _, contract, _ = execute(name)
    assert contract.validation.kind == case.expects_validation


@pytest.mark.parametrize("name", sorted(CASES))
def test_power_analysis_agrees_with_the_expected_verdict(name: str) -> None:
    """Checked at the standard protocol, not the fast one the other tests run on.

    Detectability is a property of the dataset paired with a protocol, so three
    fits and fifty fits give different and equally correct answers.
    """
    case = CASES[name]
    _, contract = run_eda(frame=case.frame, target_column=case.target, seed=7)
    assert contract.power.underpowered == case.expects_underpowered


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_outcome_folds_into_a_research_state(name: str) -> None:
    _, contract, outcome = execute(name)
    state, _ = build_state(
        session_id=f"shape-{name}",
        contract=contract,
        experiments=outcome.experiments,
        evidences=outcome.evidences,
        slots=(),
        budget=SessionBudget(experiments=20, tokens=100_000, seconds=600.0),
        reference_id=outcome.reference_id,
    )
    assert state.incumbent_experiment_id == REFERENCE_ID
    assert state.reference_score == outcome.by_id[REFERENCE_ID].mean(contract.primary_metric)


def test_a_wide_frame_does_not_swamp_the_projection() -> None:
    """Three hundred columns must not turn the prompt into a column listing."""
    from dsar.core.projection import render_schema

    _, contract, _ = execute("wide")
    rendered = render_schema(contract)
    assert "grouped by role" in rendered
    assert len(rendered) < 2000


def test_heavy_missingness_survives_the_preparation_cell() -> None:
    _, contract, outcome = execute("heavy_missing")
    reference = outcome.by_id[REFERENCE_ID]
    assert reference.status is ExperimentStatus.COMPLETED
    assert reference.complexity.n_features > 0


def test_a_confirmed_grouping_key_is_actually_runnable() -> None:
    """Confirming the key must switch the scheme and still execute.

    Without this the group path stays unreachable: the default never selects it,
    so nothing would ever generate a grouped fold.
    """
    from dataclasses import replace

    case = CASES["grouped"]
    _, contract = run_eda(frame=case.frame, target_column=case.target, k=K, repeats=REPEATS, seed=7)
    key = contract.schema and case.expects_group_candidate
    assert key is not None

    confirmed = replace(
        contract,
        validation=replace(contract.validation, kind="group_kfold", group_column=key),
    )
    features, target = prepared_frame(case.frame, confirmed)
    outcome = run_baselines(
        frame=features,
        target=target,
        contract=confirmed,
        executor=ForkExecutor(features, target),
        limits=LIMITS,
        environment="grouped",
        tune=False,
    )
    reference = outcome.by_id[REFERENCE_ID]
    assert reference.status is ExperimentStatus.COMPLETED
    assert len(reference.fold_scores[confirmed.primary_metric]) == confirmed.validation.n_fits


@pytest.mark.parametrize("name", ["temporal", "grouped", "regression_plain"])
def test_repeats_do_not_desynchronise_the_fold_count(name: str) -> None:
    """The fast protocol used above runs a single repeat, which hides a mismatch."""
    from dsar.modeling.cv import make_folds
    from dsar.stages.baseline import group_series

    case = CASES[name]
    _, contract = run_eda(frame=case.frame, target_column=case.target, k=5, repeats=3, seed=7)
    features, target = prepared_frame(case.frame, contract)
    folds = make_folds(contract.validation, target, groups=group_series(features, contract))
    assert len(folds) == contract.validation.n_fits