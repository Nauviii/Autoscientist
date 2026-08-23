"""The statistical layer decides what counts as evidence, so it is tested hardest."""

from __future__ import annotations

import random

import pytest

from dsar.core.contracts import (
    ComplexityVector,
    ControlRole,
    DataContract,
    HypothesisStatus,
    TargetDistribution,
    ValidationSpec,
)
from dsar.core.statistics import (
    aggregate_hypothesis_status,
    build_evidence,
    build_power_profile,
    calibrate_rho,
    check_gates,
    decide,
    empirical_fold_sd,
    equivalent_to_best,
    estimate_fold_sd,
    minimum_detectable_effect,
    nadeau_bengio_se,
    paired_delta,
    refresh_power_profile,
    select_recommended,
    selection_penalty,
    should_promote,
)

PROTOCOL = ValidationSpec(kind="stratified_kfold", k=10, repeats=5, seed=42)


def make_contract(delta_practical: float = 0.01, **overrides) -> DataContract:
    """Contract anchored on a mid-sized imbalanced binary task."""
    distribution = TargetDistribution(task="binary", n_rows=7043, n_positive=1869)
    power = build_power_profile(
        n_rows=distribution.n_rows,
        n_pos=distribution.n_positive,
        validation=PROTOCOL,
        metric_family="pr_auc",
        expected_score=0.84,
        expected_auc=0.84,
        delta_practical=delta_practical,
    )
    defaults = dict(
        dataset_hash="test",
        task="binary",
        target="y",
        target_distribution=distribution,
        schema={},
        validation=PROTOCOL,
        primary_metric="pr_auc",
        metric_family="pr_auc",
        secondary_metrics=("brier",),
        secondary_gates={"brier": 0.005},
        delta_practical=delta_practical,
        power=power,
        leakage_flags=(),
        excluded_columns=(),
    )
    return DataContract(**{**defaults, **overrides})


def make_scores(base: float, gain: float, seed: int, n: int = 50) -> list[float]:
    rng = random.Random(seed)
    return [base + gain + rng.gauss(0, 0.03) for _ in range(n)]


def paired_scores(control: list[float], gain: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    return [c + gain + rng.gauss(0, 0.008) for c in control]


def complexity(features: int, cells: int) -> ComplexityVector:
    return ComplexityVector(
        n_features=features,
        n_cells=cells,
        pipeline_depth=cells,
        fit_seconds=float(cells),
        ast_nodes=features * 5,
    )


def test_ten_fold_beats_five_fold_more_than_repeats_do() -> None:
    """The n_test/n_train term does not shrink with repeats, so k matters more."""
    five_repeated = nadeau_bengio_se(0.02, ValidationSpec("stratified_kfold", 5, 5, 42))
    ten_once = nadeau_bengio_se(0.02, ValidationSpec("stratified_kfold", 10, 1, 42))
    assert ten_once < five_repeated


def test_mde_shows_diminishing_returns_past_five_repeats() -> None:
    mdes = [
        minimum_detectable_effect(0.02, ValidationSpec("stratified_kfold", 10, r, 42))
        for r in (1, 3, 5, 10)
    ]
    assert mdes == sorted(mdes, reverse=True)
    assert mdes[2] - mdes[3] < 0.2 * (mdes[0] - mdes[1])


def test_fold_sd_estimate_rises_with_fold_count() -> None:
    """Each fold tests fewer rows as k grows, so per-fold scores swing wider."""
    assert estimate_fold_sd(0.01, k=10) > estimate_fold_sd(0.01, k=5)


def test_fold_sd_rejects_impossible_correlation() -> None:
    with pytest.raises(ValueError):
        estimate_fold_sd(0.01, k=10, rho=1.0)


def test_power_profile_flags_an_underpowered_dataset() -> None:
    tiny = build_power_profile(
        n_rows=400,
        n_pos=100,
        validation=PROTOCOL,
        metric_family="pr_auc",
        expected_score=0.80,
        expected_auc=0.80,
        delta_practical=0.01,
    )
    assert tiny.underpowered
    assert "INCONCLUSIVE" in tiny.note


def test_delta_min_is_floored_by_detectability() -> None:
    """A user may ask for less than the data can resolve; the floor wins."""
    contract = make_contract(delta_practical=0.001)
    assert contract.delta_min == contract.power.mde
    assert contract.delta_min > 0.001


def test_delta_min_respects_a_stricter_practical_threshold() -> None:
    contract = make_contract(delta_practical=0.10)
    assert contract.delta_min == 0.10


def test_paired_delta_requires_matching_folds() -> None:
    contract = make_contract()
    with pytest.raises(ValueError):
        paired_delta([0.1] * 50, [0.1] * 49, contract.validation, "pr_auc")
    with pytest.raises(ValueError):
        paired_delta([0.1] * 10, [0.1] * 10, contract.validation, "pr_auc")


def test_paired_interval_is_far_tighter_than_the_marginal_spread() -> None:
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=1)
    treatment = paired_scores(control, 0.02, seed=2)
    comparison = paired_delta(treatment, control, contract.validation, "pr_auc")
    assert comparison.raw_sd < 0.01
    assert comparison.ci_low < comparison.mean_delta < comparison.ci_high


@pytest.mark.parametrize(
    "gain,expected",
    [
        (0.050, HypothesisStatus.SUPPORTED),
        (0.005, HypothesisStatus.REJECTED),
        (-0.030, HypothesisStatus.REJECTED),
    ],
)
def test_decision_rule_maps_effect_size_to_status(gain: float, expected: HypothesisStatus) -> None:
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=3)
    treatment = paired_scores(control, gain, seed=4)
    verdict = decide({"pr_auc": treatment}, {"pr_auc": control}, contract)
    assert verdict.status is expected
    assert verdict.reason


def test_wide_intervals_yield_inconclusive_not_rejected() -> None:
    """Insufficient evidence must stay distinguishable from practical equivalence."""
    contract = make_contract()
    rng = random.Random(9)
    control = make_scores(0.652, 0.0, seed=5)
    treatment = [c + 0.016 + rng.gauss(0, 0.09) for c in control]
    verdict = decide({"pr_auc": treatment}, {"pr_auc": control}, contract)
    assert verdict.status is HypothesisStatus.INCONCLUSIVE
    assert verdict.underpowered


def test_secondary_gate_overrides_a_primary_gain() -> None:
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=6)
    treatment = paired_scores(control, 0.05, seed=7)
    verdict = decide(
        {"pr_auc": treatment, "brier": [0.20] * 50},
        {"pr_auc": control, "brier": [0.18] * 50},
        contract,
    )
    assert verdict.status is HypothesisStatus.REJECTED
    assert verdict.gate_violations


def test_gates_respect_metric_direction() -> None:
    """Lower-is-better and higher-is-better metrics must not be conflated."""
    assert not check_gates({"brier": [0.17] * 5}, {"brier": [0.18] * 5}, {"brier": 0.005})
    assert check_gates({"recall": [0.60] * 5}, {"recall": [0.70] * 5}, {"recall": 0.005})


def test_only_supported_evidence_promotes_the_incumbent() -> None:
    """The ladder rule is what stops the incumbent accumulating noise."""
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=8)
    strong = decide({"pr_auc": paired_scores(control, 0.05, 10)}, {"pr_auc": control}, contract)
    weak = decide({"pr_auc": paired_scores(control, 0.008, 11)}, {"pr_auc": control}, contract)
    assert should_promote(strong)
    assert not should_promote(weak)


def test_evidence_records_the_control_it_was_measured_against() -> None:
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=12)
    verdict = decide({"pr_auc": paired_scores(control, 0.05, 13)}, {"pr_auc": control}, contract)
    evidence = build_evidence(
        evidence_id="EV1",
        experiment_id="E010",
        hypothesis_id="H003",
        control_experiment_id="E000",
        control_fingerprint="fp_ref",
        control_role=ControlRole.REFERENCE,
        verdict=verdict,
    )
    assert evidence.control_role is ControlRole.REFERENCE
    assert evidence.control_fingerprint == "fp_ref"
    assert evidence.status is verdict.status
    assert evidence.ci_low == verdict.primary.ci_low


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], HypothesisStatus.PROPOSED),
        ([HypothesisStatus.SUPPORTED], HypothesisStatus.SUPPORTED),
        ([HypothesisStatus.REJECTED, HypothesisStatus.REJECTED], HypothesisStatus.REJECTED),
        ([HypothesisStatus.INCONCLUSIVE], HypothesisStatus.INCONCLUSIVE),
        (
            [HypothesisStatus.SUPPORTED, HypothesisStatus.INCONCLUSIVE],
            HypothesisStatus.PARTIALLY_SUPPORTED,
        ),
        (
            [HypothesisStatus.SUPPORTED, HypothesisStatus.REJECTED],
            HypothesisStatus.PARTIALLY_SUPPORTED,
        ),
    ],
)
def test_hypothesis_aggregation_covers_every_branch(statuses, expected) -> None:
    assert aggregate_hypothesis_status(statuses) is expected


def test_recommendation_prefers_the_simplest_equivalent_pipeline() -> None:
    """The deliverable is a baseline, so parsimony beats the highest score."""
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=14)
    best = paired_scores(control, 0.05, seed=15)
    simple = [s - 0.002 for s in best]

    candidates = [
        ("complex_best", {"pr_auc": best}, complexity(180, 5)),
        ("simple_equivalent", {"pr_auc": simple}, complexity(40, 2)),
        ("clearly_worse", {"pr_auc": control}, complexity(20, 1)),
    ]
    chosen, alternatives = select_recommended(candidates, "complex_best", contract)
    assert chosen == "simple_equivalent"
    assert alternatives == ("complex_best",)


def test_equivalence_excludes_a_materially_worse_pipeline() -> None:
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=16)
    best = paired_scores(control, 0.05, seed=17)
    assert equivalent_to_best({"pr_auc": [b - 0.002 for b in best]}, {"pr_auc": best}, contract)
    assert not equivalent_to_best({"pr_auc": control}, {"pr_auc": best}, contract)


def test_selection_requires_the_best_to_be_a_candidate() -> None:
    contract = make_contract()
    control = make_scores(0.652, 0.0, seed=18)
    with pytest.raises(ValueError):
        select_recommended([("a", {"pr_auc": control}, complexity(10, 1))], "missing", contract)


def test_empirical_sd_waits_for_enough_comparisons() -> None:
    histories = [[0.01, 0.02, 0.015]] * 4
    assert empirical_fold_sd(histories) is None
    assert empirical_fold_sd(histories * 3) is not None


def test_refreshed_power_profile_supersedes_the_analytic_prior() -> None:
    """Once folds have been observed, the measured sd replaces the assumed one."""
    contract = make_contract()
    refreshed = refresh_power_profile(
        contract.power, observed_sd=0.004, validation=PROTOCOL, delta_practical=0.01
    )
    assert refreshed.empirical
    assert refreshed.method == "empirical_fold_sd"
    assert refreshed.mde < contract.power.mde
    assert refreshed.marginal_se == contract.power.marginal_se


def test_rho_calibration_recovers_a_strong_pairing() -> None:
    control = make_scores(0.652, 0.0, seed=19)
    treatment = paired_scores(control, 0.02, seed=20)
    assert calibrate_rho(treatment, control) > 0.85


def test_rho_calibration_needs_enough_folds() -> None:
    with pytest.raises(ValueError):
        calibrate_rho([0.1, 0.2], [0.1, 0.2])


def test_selection_penalty_quantifies_the_winners_curse() -> None:
    assert selection_penalty(0.712, 0.688) == pytest.approx(0.024)