"""Metric noise models must react to dataset shape, never to a hardcoded constant."""

from __future__ import annotations

import math

import pytest

from dsar.core.noise import (
    estimate_marginal_se,
    expected_average_precision,
    pr_auc_se,
    proportion_se,
    r2_se,
    rmse_se,
    roc_auc_se,
)


def test_roc_auc_se_matches_published_form() -> None:
    """Hanley-McNeil on a known shape, verified against a hand calculation."""
    assert roc_auc_se(0.84, n_pos=1869, n_neg=5174) == pytest.approx(0.00603, abs=5e-5)


def test_roc_auc_se_shrinks_with_sample_size() -> None:
    small = roc_auc_se(0.80, n_pos=100, n_neg=400)
    large = roc_auc_se(0.80, n_pos=1000, n_neg=4000)
    assert large < small
    assert large == pytest.approx(small / math.sqrt(10), rel=0.15)


def test_roc_auc_se_rejects_degenerate_classes() -> None:
    with pytest.raises(ValueError):
        roc_auc_se(0.80, n_pos=1, n_neg=500)


def test_expected_ap_approaches_prevalence_for_a_useless_model() -> None:
    """An AUC just above chance should imply an AP barely above the base rate."""
    for prevalence in (0.05, 0.30, 0.60):
        assert expected_average_precision(0.51, prevalence) == pytest.approx(prevalence, abs=0.03)


def test_expected_ap_rises_with_discrimination() -> None:
    values = [expected_average_precision(auc, 0.25) for auc in (0.60, 0.75, 0.90)]
    assert values == sorted(values)
    assert values[-1] > 0.5


def test_expected_ap_falls_with_rarer_positives() -> None:
    values = [expected_average_precision(0.85, p) for p in (0.02, 0.10, 0.40)]
    assert values == sorted(values)


def test_expected_ap_validates_prevalence() -> None:
    with pytest.raises(ValueError):
        expected_average_precision(0.80, 1.5)


def test_expected_ap_clamps_a_model_with_no_signal() -> None:
    """A baseline can genuinely land at chance; power analysis must still report."""
    at_chance = expected_average_precision(0.40, 0.20)
    assert at_chance == pytest.approx(0.20, abs=0.02)

    estimate = estimate_marginal_se(
        family="pr_auc", n_rows=7000, n_pos=1400, expected_score=0.48, expected_auc=0.48
    )
    assert estimate.marginal_se > 0
    assert any("clamped" in a for a in estimate.assumptions)


def test_pr_auc_se_scales_with_positive_count_not_row_count() -> None:
    """Effective sample size is the positive class, which is why imbalance hurts."""
    few = pr_auc_se(0.30, n_pos=100)
    many = pr_auc_se(0.30, n_pos=1600)
    assert many == pytest.approx(few / 4.0, rel=1e-6)


def test_regression_estimators_shrink_with_n() -> None:
    assert r2_se(0.55, 20000) < r2_se(0.55, 2000)
    assert rmse_se(45.0, 20000) < rmse_se(45.0, 2000)
    assert proportion_se(0.70, 20000) < proportion_se(0.70, 2000)


@pytest.mark.parametrize(
    "family,kwargs,method",
    [
        ("roc_auc", {"expected_score": 0.84, "n_pos": 1869}, "hanley_mcneil"),
        ("pr_auc", {"expected_score": 0.84, "n_pos": 1869, "expected_auc": 0.84}, "binormal_ap"),
        ("r2", {"expected_score": 0.55}, "delta_method"),
        ("rmse", {"expected_score": 45.0}, "normal_residual"),
        ("accuracy", {"expected_score": 0.70}, "binomial"),
    ],
)
def test_dispatch_covers_every_supported_family(family: str, kwargs: dict, method: str) -> None:
    estimate = estimate_marginal_se(family=family, n_rows=7043, **kwargs)
    assert estimate.method == method
    assert estimate.marginal_se > 0
    assert estimate.assumptions


def test_dispatch_rejects_unknown_family() -> None:
    with pytest.raises(ValueError):
        estimate_marginal_se(family="f1", n_rows=1000, expected_score=0.5)  # type: ignore[arg-type]


def test_pr_auc_requires_positive_count() -> None:
    with pytest.raises(ValueError):
        estimate_marginal_se(family="pr_auc", n_rows=1000, expected_score=0.8)


def test_imbalance_widens_the_pr_auc_interval() -> None:
    """The failure a fixed inflation factor used to hide: rare positives are noisier."""
    balanced = estimate_marginal_se(
        family="pr_auc", n_rows=7000, n_pos=1850, expected_score=0.84, expected_auc=0.84
    )
    rare = estimate_marginal_se(
        family="pr_auc", n_rows=7000, n_pos=140, expected_score=0.84, expected_auc=0.84
    )
    assert rare.marginal_se > 2.5 * balanced.marginal_se