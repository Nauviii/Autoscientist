"""Measure the numbers the statistical layer has so far only assumed.

Everything downstream rests on three quantities that were estimated analytically
before any data was seen: the baseline score, the per-fold sd, and the correlation
between paired folds. This script replaces all three with measurements, and checks
that repeated runs agree to the last digit.

Usage:
    uv run python scripts/calibrate.py data/telco.csv Churn
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

from dsar.modeling.registry import apply_determinism_env

apply_determinism_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dsar.core.contracts import ModelConfig, ValidationSpec  # noqa: E402
from dsar.core.statistics import (  # noqa: E402
    build_power_profile,
    calibrate_rho,
    estimate_fold_sd,
    minimum_detectable_effect,
    nadeau_bengio_se,
    paired_delta,
)
from dsar.modeling.cv import make_folds, run_cv  # noqa: E402
from dsar.modeling.registry import default_params  # noqa: E402


def prepare(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Minimal preparation only: drop ID-like columns, coerce types, cast categories."""
    y_raw = frame[target]
    x = frame.drop(columns=[target])
    dropped: list[str] = []

    for column in list(x.columns):
        series = x[column]
        # Continuous floats are near-unique by nature, so the ratio alone would
        # discard perfectly good features.
        looks_like_key = not pd.api.types.is_float_dtype(series)
        if looks_like_key and series.nunique(dropna=False) / len(series) > 0.99:
            x = x.drop(columns=[column])
            dropped.append(f"{column} (id-like)")
            continue
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            if coerced.notna().mean() > 0.9:
                x[column] = coerced.fillna(coerced.median())
                dropped.append(f"{column} (coerced to numeric)")
            else:
                x[column] = series.astype("category")

    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.astype(int)
    else:
        y = y_raw.astype(str).str.strip().str.lower().isin({"yes", "1", "true"}).astype(int)
    return x, y, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the noise model on real data.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("target")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-practical", type=float, default=0.01)
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"dataset not found: {args.dataset}")

    raw = pd.read_csv(args.dataset)
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()[:16]
    x, y, notes = prepare(raw, args.target)

    n_pos = int(y.sum())
    print(f"dataset      {args.dataset.name}  hash={digest}")
    print(f"shape        {raw.shape[0]} rows, {x.shape[1]} features after preparation")
    print(f"target       {n_pos} positive ({n_pos / len(y):.1%})")
    for note in notes:
        print(f"  prepared   {note}")

    validation = ValidationSpec(
        kind="stratified_kfold", k=args.k, repeats=args.repeats, seed=args.seed
    )
    folds = make_folds(validation, y)
    print(f"protocol     {args.k}-fold x {args.repeats} = {len(folds)} fits")

    print("\nrunning reference baseline")
    started = time.monotonic()
    model = ModelConfig(family="lightgbm", params=default_params("lightgbm", "binary"))
    reference = run_cv(
        frame=x, target=y, folds=folds, model=model, task="binary", seed=args.seed
    )
    elapsed = time.monotonic() - started

    print(f"\n{'metric':<10}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}")
    for metric, scores in reference.fold_scores.items():
        values = np.array(scores)
        print(
            f"{metric:<10}{values.mean():>10.4f}{values.std(ddof=1):>10.4f}"
            f"{values.min():>10.4f}{values.max():>10.4f}"
        )
    print(f"\nruntime      {elapsed:.1f}s for {len(folds)} fits "
          f"({elapsed / len(folds):.2f}s per fit)")

    print("\nchecking determinism")
    replay = run_cv(
        frame=x, target=y, folds=folds, model=model, task="binary", seed=args.seed
    )
    identical = all(
        np.array_equal(reference.fold_scores[m], replay.fold_scores[m])
        for m in reference.fold_scores
    )
    print(f"  repeated run {'matches to the last digit' if identical else 'DIFFERS'}")

    print("\nrunning a perturbed variant to measure paired behaviour")
    perturbed = ModelConfig(
        family="lightgbm",
        params={**default_params("lightgbm", "binary"), "num_leaves": 63},
    )
    variant = run_cv(
        frame=x, target=y, folds=folds, model=perturbed, task="binary", seed=args.seed
    )

    primary = "pr_auc"
    control = list(reference.fold_scores[primary])
    treatment = list(variant.fold_scores[primary])
    comparison = paired_delta(treatment, control, validation, primary)
    observed_rho = calibrate_rho(treatment, control)

    marginal_sd = float(np.std(control, ddof=1))
    predicted = build_power_profile(
        n_rows=len(y),
        n_pos=n_pos,
        validation=validation,
        metric_family="pr_auc",
        expected_score=float(np.mean(reference.fold_scores["roc_auc"])),
        expected_auc=float(np.mean(reference.fold_scores["roc_auc"])),
        delta_practical=args.delta_practical,
    )
    empirical_mde = minimum_detectable_effect(comparison.raw_sd, validation)

    print(f"\n{'quantity':<22}{'predicted':>12}{'observed':>12}")
    print(f"{'per-fold sd (' + primary + ')':<22}{'-':>12}{marginal_sd:>12.4f}")
    print(f"{'paired delta sd':<22}{predicted.sd_estimate:>12.4f}{comparison.raw_sd:>12.4f}")
    print(f"{'rho between folds':<22}{0.90:>12.4f}{observed_rho:>12.4f}")
    print(f"{'corrected se':<22}{predicted.corrected_se:>12.4f}"
          f"{nadeau_bengio_se(comparison.raw_sd, validation):>12.4f}")
    print(f"{'mde':<22}{predicted.mde:>12.4f}{empirical_mde:>12.4f}")

    implied_sd = estimate_fold_sd(predicted.marginal_se, validation.k, observed_rho)
    print(f"\nwith the observed rho, the analytic sd would be {implied_sd:.4f}")
    print(f"delta_min would become {max(args.delta_practical, empirical_mde):.4f} "
          f"(was {max(args.delta_practical, predicted.mde):.4f})")
    print(f"\nvariant delta {comparison.mean_delta:+.4f} "
          f"CI [{comparison.ci_low:+.4f}, {comparison.ci_high:+.4f}]")


if __name__ == "__main__":
    main()
