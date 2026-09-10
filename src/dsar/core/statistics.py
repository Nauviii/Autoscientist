"""Statistical layer: power analysis, paired CV comparison, hypothesis decision rule.

Pure module. Every function is a deterministic mapping from numbers to numbers.
No constant here is calibrated to a single dataset; metric noise comes from
core.noise, which dispatches on task and metric family.

References:
  Nadeau & Bengio (2003), Machine Learning 52(3) - inference for generalisation error.
  Cawley & Talbot (2010), JMLR 11 - overfitting in model selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from scipy import stats

from .contracts import (
    ComplexityVector,
    ControlRole,
    DataContract,
    Evidence,
    HypothesisStatus,
    PowerProfile,
    ValidationSpec,
)
from .noise import MetricFamily, estimate_marginal_se

# Prior correlation between per-fold scores of two pipelines on identical folds.
# Measured at 0.951 on a mid-sized imbalanced binary task and 0.976 on a synthetic
# one; 0.95 sits on the conservative side of both. Replaced by calibrate_rho once
# ten paired comparisons exist, since the value is dataset-specific.
DEFAULT_RHO = 0.95

# Number of completed paired comparisons after which analytic priors are dropped
# in favour of the observed per-fold sd.
EMPIRICAL_SD_MIN_COMPARISONS = 10


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Result of comparing treatment against control on identical folds."""

    metric: str
    mean_delta: float
    corrected_se: float
    ci_low: float
    ci_high: float
    n_folds: int
    raw_sd: float

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low


@dataclass(frozen=True, slots=True)
class Verdict:
    """Decision produced by the rule, plus the reason it can be audited against."""

    status: HypothesisStatus
    primary: PairedComparison
    delta_min: float
    gate_violations: tuple[str, ...]
    underpowered: bool
    reason: str


def estimate_fold_sd(
    marginal_se: float,
    k: int,
    rho: float = DEFAULT_RHO,
) -> float:
    """Estimate the sd of per-fold paired deltas from the full-data marginal SE.

    A single fold tests n/k rows, inflating its SE by sqrt(k). Pairing on identical
    folds then shrinks the delta variance by sqrt(2(1 - rho)).
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    return marginal_se * math.sqrt(k) * math.sqrt(2.0 * (1.0 - rho))


def nadeau_bengio_se(sd: float, validation: ValidationSpec) -> float:
    """Corrected SE for repeated CV, accounting for overlapping training sets.

    The n_test/n_train term does not shrink with more repeats, which is why moving
    from 5-fold to 10-fold helps far more than adding repetitions.
    """
    n = validation.n_fits
    return sd * math.sqrt(1.0 / n + validation.test_train_ratio)


def minimum_detectable_effect(
    sd: float,
    validation: ValidationSpec,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest true improvement detectable at the given alpha and power."""
    df = validation.n_fits - 1
    t_alpha = stats.t.ppf(1.0 - alpha / 2.0, df)
    t_beta = stats.t.ppf(power, df)
    return (t_alpha + t_beta) * nadeau_bengio_se(sd, validation)


def build_power_profile(
    *,
    n_rows: int,
    validation: ValidationSpec,
    metric_family: MetricFamily,
    expected_score: float,
    delta_practical: float,
    n_pos: int | None = None,
    expected_auc: float | None = None,
    rho: float = DEFAULT_RHO,
    candidate_protocols: Sequence[tuple[int, int]] = ((5, 1), (5, 5), (10, 3), (10, 5), (10, 10)),
) -> PowerProfile:
    """Run EDA block F: report what effect sizes this dataset can resolve."""
    estimate = estimate_marginal_se(
        family=metric_family,
        n_rows=n_rows,
        expected_score=expected_score,
        n_pos=n_pos,
        expected_auc=expected_auc,
    )

    table: list[tuple[str, int, float]] = []
    for k, repeats in candidate_protocols:
        spec = ValidationSpec(kind=validation.kind, k=k, repeats=repeats, seed=validation.seed)
        sd = estimate_fold_sd(estimate.marginal_se, k, rho)
        table.append((f"{k}-fold x {repeats}", spec.n_fits, minimum_detectable_effect(sd, spec)))

    sd = estimate_fold_sd(estimate.marginal_se, validation.k, rho)
    corrected = nadeau_bengio_se(sd, validation)
    mde = minimum_detectable_effect(sd, validation)
    underpowered = mde > 2.0 * delta_practical

    note = f"MDE {mde:.4f} at {validation.n_fits} fits via {estimate.method}."
    if underpowered:
        note += (
            f" This exceeds twice the practical threshold {delta_practical:.4f}; typical"
            " feature-engineering gains will land INCONCLUSIVE on this dataset."
        )

    return PowerProfile(
        marginal_se=estimate.marginal_se,
        sd_estimate=sd,
        corrected_se=corrected,
        mde=mde,
        protocol_table=tuple(table),
        underpowered=underpowered,
        method=estimate.method,
        assumptions=estimate.assumptions,
        note=note,
    )


def empirical_fold_sd(paired_histories: Sequence[Sequence[float]]) -> float | None:
    """Pooled sd of observed per-fold deltas, superseding the analytic prior.

    Returns None until enough comparisons exist for the estimate to be stable.
    """
    usable = [h for h in paired_histories if len(h) > 1]
    if len(usable) < EMPIRICAL_SD_MIN_COMPARISONS:
        return None
    total_ss = 0.0
    total_df = 0
    for deltas in usable:
        mean = sum(deltas) / len(deltas)
        total_ss += sum((d - mean) ** 2 for d in deltas)
        total_df += len(deltas) - 1
    return math.sqrt(total_ss / total_df)


def calibrate_rho(treatment: Sequence[float], control: Sequence[float]) -> float:
    """Observed Pearson correlation between paired per-fold scores."""
    n = len(treatment)
    if n != len(control) or n < 3:
        raise ValueError("need at least 3 paired folds")
    mt = sum(treatment) / n
    mc = sum(control) / n
    cov = sum((t - mt) * (c - mc) for t, c in zip(treatment, control))
    var_t = sum((t - mt) ** 2 for t in treatment)
    var_c = sum((c - mc) ** 2 for c in control)
    if var_t <= 0.0 or var_c <= 0.0:
        return 0.0
    return cov / math.sqrt(var_t * var_c)


def paired_delta(
    treatment: Sequence[float],
    control: Sequence[float],
    validation: ValidationSpec,
    metric: str,
    alpha: float = 0.05,
) -> PairedComparison:
    """Compare two experiments on identical folds with the corrected interval."""
    if len(treatment) != len(control):
        raise ValueError("paired comparison requires identical fold counts")
    if len(treatment) != validation.n_fits:
        raise ValueError(f"expected {validation.n_fits} folds, got {len(treatment)}")

    deltas = [t - c for t, c in zip(treatment, control)]
    n = len(deltas)
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / (n - 1)
    sd = math.sqrt(variance)
    se = nadeau_bengio_se(sd, validation)
    margin = stats.t.ppf(1.0 - alpha / 2.0, n - 1) * se

    return PairedComparison(
        metric=metric,
        mean_delta=mean,
        corrected_se=se,
        ci_low=mean - margin,
        ci_high=mean + margin,
        n_folds=n,
        raw_sd=sd,
    )


def check_gates(
    treatment: Mapping[str, Sequence[float]],
    control: Mapping[str, Sequence[float]],
    gates: Mapping[str, float],
    lower_is_better: frozenset[str] = frozenset({"brier", "logloss", "rmse", "mae", "mape"}),
) -> tuple[str, ...]:
    """Return secondary metrics whose degradation exceeds the configured tolerance."""
    violations: list[str] = []
    for metric, tolerance in gates.items():
        if metric not in treatment or metric not in control:
            continue
        t_mean = sum(treatment[metric]) / len(treatment[metric])
        c_mean = sum(control[metric]) / len(control[metric])
        degradation = (t_mean - c_mean) if metric in lower_is_better else (c_mean - t_mean)
        if degradation > tolerance:
            violations.append(f"{metric}: degraded {degradation:+.4f} > {tolerance:.4f}")
    return tuple(violations)


def decide(
    treatment: Mapping[str, Sequence[float]],
    control: Mapping[str, Sequence[float]],
    contract: DataContract,
    alpha: float = 0.05,
) -> Verdict:
    """Map paired evidence to a hypothesis status. Fully deterministic.

    SUPPORTED     lower bound of the interval clears delta_min
    REJECTED      upper bound falls below delta_min, or a secondary gate is violated
    INCONCLUSIVE  the interval straddles delta_min; evidence is insufficient
    """
    metric = contract.primary_metric
    comparison = paired_delta(
        treatment[metric], control[metric], contract.validation, metric, alpha
    )
    delta_min = contract.delta_min
    violations = check_gates(treatment, control, contract.secondary_gates)
    underpowered = comparison.ci_width > 2.0 * delta_min

    if violations:
        status = HypothesisStatus.REJECTED
        reason = f"secondary gate violated ({'; '.join(violations)})"
    elif comparison.ci_low > delta_min:
        status = HypothesisStatus.SUPPORTED
        reason = f"CI lower bound {comparison.ci_low:+.4f} exceeds delta_min {delta_min:.4f}"
    elif comparison.ci_high < delta_min:
        status = HypothesisStatus.REJECTED
        reason = f"CI upper bound {comparison.ci_high:+.4f} below delta_min {delta_min:.4f}"
    else:
        status = HypothesisStatus.INCONCLUSIVE
        reason = (
            f"CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}] straddles "
            f"delta_min {delta_min:.4f}"
        )

    return Verdict(
        status=status,
        primary=comparison,
        delta_min=delta_min,
        gate_violations=violations,
        underpowered=underpowered,
        reason=reason,
    )


def refresh_power_profile(
    profile: PowerProfile,
    observed_sd: float,
    validation: ValidationSpec,
    delta_practical: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerProfile:
    """Replace the analytic prior with the observed per-fold sd once data exists."""
    corrected = nadeau_bengio_se(observed_sd, validation)
    mde = minimum_detectable_effect(observed_sd, validation, alpha, power)
    return PowerProfile(
        marginal_se=profile.marginal_se,
        sd_estimate=observed_sd,
        corrected_se=corrected,
        mde=mde,
        protocol_table=profile.protocol_table,
        underpowered=mde > 2.0 * delta_practical,
        method="empirical_fold_sd",
        assumptions=(f"pooled over observed paired comparisons, prior was {profile.method}",),
        note=f"MDE {mde:.4f} from observed per-fold sd {observed_sd:.4f}.",
        empirical=True,
    )


def build_evidence(
    *,
    evidence_id: str,
    experiment_id: str,
    hypothesis_id: str | None,
    control_experiment_id: str,
    control_fingerprint: str,
    control_role: ControlRole,
    verdict: Verdict,
) -> Evidence:
    """Bind a verdict to the control that produced it, so it stays interpretable."""
    return Evidence(
        evidence_id=evidence_id,
        experiment_id=experiment_id,
        hypothesis_id=hypothesis_id,
        control_experiment_id=control_experiment_id,
        control_fingerprint=control_fingerprint,
        control_role=control_role,
        metric=verdict.primary.metric,
        mean_delta=verdict.primary.mean_delta,
        ci_low=verdict.primary.ci_low,
        ci_high=verdict.primary.ci_high,
        delta_min=verdict.delta_min,
        status=verdict.status,
        underpowered=verdict.underpowered,
        gate_violations=verdict.gate_violations,
        reason=verdict.reason,
    )


def should_promote(verdict: Verdict) -> bool:
    """Ladder rule: the incumbent advances only on evidence that clears delta_min.

    A positive but inconclusive delta never promotes, which is what stops the
    incumbent from accumulating noise across a long session.
    """
    return verdict.status is HypothesisStatus.SUPPORTED


def aggregate_hypothesis_status(statuses: Sequence[HypothesisStatus]) -> HypothesisStatus:
    """Combine per-experiment verdicts into one hypothesis-level status."""
    if not statuses:
        return HypothesisStatus.PROPOSED
    supported = sum(s is HypothesisStatus.SUPPORTED for s in statuses)
    rejected = sum(s is HypothesisStatus.REJECTED for s in statuses)
    inconclusive = sum(s is HypothesisStatus.INCONCLUSIVE for s in statuses)

    if supported and (rejected or inconclusive):
        return HypothesisStatus.PARTIALLY_SUPPORTED
    if supported:
        return HypothesisStatus.SUPPORTED
    if rejected and not inconclusive:
        return HypothesisStatus.REJECTED
    return HypothesisStatus.INCONCLUSIVE


def equivalent_to_best(
    candidate: Mapping[str, Sequence[float]],
    best: Mapping[str, Sequence[float]],
    contract: DataContract,
    alpha: float = 0.05,
) -> bool:
    """True when the candidate is not meaningfully worse than the best pipeline."""
    metric = contract.primary_metric
    gap = paired_delta(best[metric], candidate[metric], contract.validation, metric, alpha)
    return gap.ci_low < contract.delta_min


def select_recommended(
    candidates: Sequence[tuple[str, Mapping[str, Sequence[float]], ComplexityVector]],
    best_id: str,
    contract: DataContract,
    complexity_weights: Mapping[str, float] | None = None,
    alpha: float = 0.05,
) -> tuple[str, tuple[str, ...]]:
    """Pick the simplest pipeline statistically indistinguishable from the best.

    Generalises the one-standard-error rule of Breiman et al. (1984), using
    delta_min and the corrected paired interval instead of a single SE.
    """
    lookup = {cid: (scores, cx) for cid, scores, cx in candidates}
    if best_id not in lookup:
        raise ValueError(f"best_id {best_id} not among candidates")
    best_scores, best_cx = lookup[best_id]

    weights = complexity_weights or {
        "n_features": 0.3,
        "n_cells": 0.3,
        "pipeline_depth": 0.2,
        "fit_seconds": 0.1,
        "ast_nodes": 0.1,
    }

    equivalent = [
        cid
        for cid, scores, _ in candidates
        if cid == best_id or equivalent_to_best(scores, best_scores, contract, alpha)
    ]
    chosen = min(equivalent, key=lambda cid: (lookup[cid][1].scalar(weights, best_cx), cid))
    alternatives = tuple(sorted(cid for cid in equivalent if cid != chosen))
    return chosen, alternatives


def selection_penalty(selection_score: float, holdout_score: float) -> float:
    """Winner's curse: how far the selection-time best overstates true performance."""
    return selection_score - holdout_score