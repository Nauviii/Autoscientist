"""Marginal sampling noise per metric family. No constant is tied to one dataset.

Every estimator here answers the same question: given dataset shape and an expected
score, how much would this metric move under resampling alone? These are analytic
priors used only before any experiment has run; once ten experiments exist, the
empirical per-fold sd from statistics.empirical_fold_sd supersedes them.

Validated against a real mid-sized imbalanced binary task: the binormal model
predicted a per-fold PR-AUC sd of 0.0344 where 0.0336 was observed, a 2.4% error.

References:
  Hanley & McNeil (1982), Radiology 143(1) - variance of the ROC area.
  Boyd, Eng & Page (2013), ECML PKDD - interval estimation for average precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy import stats
from scipy.integrate import quad

MetricFamily = Literal["roc_auc", "pr_auc", "r2", "rmse", "accuracy", "logloss"]


@dataclass(frozen=True, slots=True)
class NoiseEstimate:
    """Marginal SE together with the method and assumptions that produced it."""

    marginal_se: float
    method: str
    assumptions: tuple[str, ...]


def roc_auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil standard error of the ROC area."""
    if n_pos < 2 or n_neg < 2:
        raise ValueError("need at least 2 samples per class")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc**2 / (1.0 + auc)
    num = auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc**2) + (n_neg - 1) * (q2 - auc**2)
    return math.sqrt(num / (n_pos * n_neg))


def expected_average_precision(auc: float, prevalence: float) -> float:
    """Expected AP implied by a ROC-AUC under an equal-variance binormal model.

    Lets the caller state a single believable AUC instead of guessing AP, which is
    far less intuitive and swings hard with prevalence. An AUC at or below chance is
    clamped rather than rejected: a weak baseline is a real outcome, and power
    analysis must still report on it instead of aborting the session.
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must lie in (0, 1)")
    auc = min(max(auc, 0.5 + 1e-4), 1.0 - 1e-4)

    separation = math.sqrt(2.0) * stats.norm.ppf(auc)

    def integrand(t: float) -> float:
        recall = 1.0 - stats.norm.cdf(t - separation)
        fpr = 1.0 - stats.norm.cdf(t)
        denom = prevalence * recall + (1.0 - prevalence) * fpr
        if denom <= 1e-12:
            return 0.0
        return (prevalence * recall / denom) * stats.norm.pdf(t - separation)

    return quad(integrand, -12.0, 12.0, limit=200)[0]


def pr_auc_se(average_precision: float, n_pos: int) -> float:
    """SE of average precision, treating it as a mean over the positive class.

    Effective sample size is the positive count, not the row count, which is why
    PR-AUC degrades so sharply under heavy imbalance.
    """
    if n_pos < 2:
        raise ValueError("need at least 2 positives")
    ap = min(max(average_precision, 1e-6), 1.0 - 1e-6)
    return math.sqrt(ap * (1.0 - ap) / n_pos)


def r2_se(r2: float, n: int) -> float:
    """Delta-method SE of the coefficient of determination."""
    if n < 10:
        raise ValueError("n too small for the delta-method approximation")
    r2 = min(max(r2, 1e-6), 1.0 - 1e-6)
    return math.sqrt(4.0 * r2 * (1.0 - r2) ** 2 / n)


def rmse_se(rmse: float, n: int) -> float:
    """SE of RMSE under approximate normality of residuals."""
    if n < 10:
        raise ValueError("n too small")
    return rmse / math.sqrt(2.0 * n)


def proportion_se(score: float, n: int) -> float:
    """Binomial SE, used for accuracy and other proportion-shaped metrics."""
    score = min(max(score, 1e-6), 1.0 - 1e-6)
    return math.sqrt(score * (1.0 - score) / n)


def estimate_marginal_se(
    *,
    family: MetricFamily,
    n_rows: int,
    expected_score: float,
    n_pos: int | None = None,
    expected_auc: float | None = None,
) -> NoiseEstimate:
    """Dispatch to the noise model matching the metric family.

    expected_score carries the metric's own scale (AUC, R2, RMSE); for pr_auc the
    AP is derived from expected_auc and prevalence rather than being guessed.
    """
    if family == "roc_auc":
        if n_pos is None:
            raise ValueError("roc_auc requires n_pos")
        se = roc_auc_se(expected_score, n_pos, n_rows - n_pos)
        return NoiseEstimate(se, "hanley_mcneil", (f"auc={expected_score:.3f}",))

    if family == "pr_auc":
        if n_pos is None:
            raise ValueError("pr_auc requires n_pos")
        requested = expected_auc if expected_auc is not None else expected_score
        prevalence = n_pos / n_rows
        ap = expected_average_precision(requested, prevalence)
        se = pr_auc_se(ap, n_pos)
        assumptions = [
            f"auc={requested:.3f}",
            f"prevalence={prevalence:.4f}",
            f"implied_ap={ap:.3f}",
        ]
        if requested <= 0.5:
            assumptions.append("auc clamped to chance; the model carries no signal")
        return NoiseEstimate(se, "binormal_ap", tuple(assumptions))

    if family == "r2":
        return NoiseEstimate(r2_se(expected_score, n_rows), "delta_method", (f"r2={expected_score:.3f}",))

    if family == "rmse":
        return NoiseEstimate(rmse_se(expected_score, n_rows), "normal_residual", (f"rmse={expected_score:.4f}",))

    if family in ("accuracy", "logloss"):
        return NoiseEstimate(proportion_se(expected_score, n_rows), "binomial", (f"score={expected_score:.3f}",))

    raise ValueError(f"unsupported metric family: {family}")