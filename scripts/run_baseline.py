"""Run the three baseline anchors on a real dataset. No LLM involved.

Everything here is deterministic: the pipeline is a single cell we wrote, the folds
come from the seed, and the verdicts come from the decision rule. Rerunning with the
same arguments reproduces the numbers exactly.

Usage:
    uv run python scripts/run_baseline.py data/telco.csv Churn
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from dsar.modeling.registry import apply_determinism_env

apply_determinism_env()

import pandas as pd  # noqa: E402

from dsar.adapters.sandbox import ForkExecutor, run_gauntlet  # noqa: E402
from dsar.core.state import SessionBudget, build_state  # noqa: E402
from dsar.ports import FoldSpec, ResourceLimits  # noqa: E402
from dsar.stages.baseline import MINIMAL_PREPARATION, run_baselines  # noqa: E402
from dsar.stages.eda import prepared_frame, run_eda  # noqa: E402

TIER_NAMES = {"E000": "trivial", "E001": "reference", "E002": "tuned"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three baseline anchors.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("target")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delta-practical", type=float, default=0.01)
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    if not args.dataset.exists():
        sys.exit(f"dataset not found: {args.dataset}")

    raw = pd.read_csv(args.dataset)
    digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()[:16]

    artifact, contract = run_eda(
        frame=raw,
        target_column=args.target,
        delta_practical=args.delta_practical,
        k=args.k,
        repeats=args.repeats,
        seed=args.seed,
        secondary_gates={"brier": 0.01},
    )
    x, y = prepared_frame(raw, contract)

    print(f"dataset      {args.dataset.name}  hash={contract.dataset_hash}  file={digest}")
    print(f"shape        {artifact.integrity.n_rows} rows, {len(contract.schema)} features "
          f"({artifact.integrity.memory_mb:.1f} MB)")
    print(f"task         {contract.task}  primary metric {contract.primary_metric}")
    if contract.target_distribution.prevalence is not None:
        print(f"target       {contract.target_distribution.n_positive} positive "
              f"({contract.target_distribution.prevalence:.1%})")
    print(f"validation   {contract.validation.kind}  {args.k}-fold x {args.repeats} = "
          f"{contract.validation.n_fits} fits  [{artifact.structure.reason}]")
    print(f"delta_min    {contract.delta_min:.4f}  (practical {args.delta_practical:.4f}, "
          f"detectable {contract.power.mde:.4f}, via {contract.power.method})")
    if contract.excluded_columns:
        print(f"excluded     {', '.join(contract.excluded_columns)}")

    if artifact.leakage:
        print("\nleakage screen")
        for flag in artifact.leakage:
            print(f"  [{flag.severity.value:<5}] {flag.column:<18}{flag.test:<26}"
                  f"{flag.statistic:>7.3f}  {flag.detail}")
    else:
        print("\nleakage screen: nothing flagged")

    for warning in artifact.warnings:
        print(f"warning      {warning}")

    executor = ForkExecutor(x, y)
    limits = ResourceLimits(memory_mb=4096, cpu_seconds=300, wall_seconds=300, max_features=500)

    print("\nchecking the preparation cell against the guard gauntlet")
    half = len(x) // 2
    gauntlet = run_gauntlet(
        [MINIMAL_PREPARATION],
        executor,
        FoldSpec(0, 0, tuple(range(half)), tuple(range(half, len(x)))),
        limits,
    )
    for outcome in gauntlet.outcomes:
        mark = "pass" if outcome.passed else "FAIL"
        print(f"  {mark}  {outcome.guard.value}{'' if outcome.passed else '  ' + outcome.detail}")
    if not gauntlet.passed:
        sys.exit("preparation cell failed its own gauntlet")

    print("\nrunning baselines")
    outcome = run_baselines(
        frame=x,
        target=y,
        contract=contract,
        executor=executor,
        limits=limits,
        environment=digest,
        tune=not args.no_tune,
    )

    metrics = (contract.primary_metric, *contract.secondary_metrics)
    header = f"{'tier':<12}{'id':<7}" + "".join(f"{m:>10}" for m in metrics) + f"{'sec':>8}"
    print("\n" + header)
    for experiment in outcome.experiments:
        row = f"{TIER_NAMES.get(experiment.experiment_id, '?'):<12}{experiment.experiment_id:<7}"
        row += "".join(f"{experiment.mean(m):>10.4f}" for m in metrics)
        print(row + f"{experiment.runtime_s:>8.1f}")

    print(f"\n{'evidence':<14}{'delta':>9}  {'interval':<22}{'verdict'}")
    for evidence in outcome.evidences:
        interval = f"[{evidence.ci_low:+.4f}, {evidence.ci_high:+.4f}]"
        print(f"{evidence.evidence_id:<14}{evidence.mean_delta:>+9.4f}  "
              f"{interval:<22}{evidence.status.value}")

    state, trace = build_state(
        session_id="baseline-only",
        contract=contract,
        experiments=outcome.experiments,
        evidences=outcome.evidences,
        slots=(),
        budget=SessionBudget(experiments=50, tokens=1_500_000, seconds=5400.0),
        reference_id=outcome.reference_id,
    )

    print(f"\nreference    {state.reference_experiment_id}  {state.reference_score:.4f}")
    print(f"incumbent    {state.incumbent_experiment_id}  {state.incumbent_score:.4f}  "
          f"({trace.promotions} promotions)")
    if outcome.chosen_params:
        print(f"tuned params {dict(outcome.chosen_params)}")
    print(f"\nheadroom above the trivial floor: "
          f"{state.reference_score - outcome.by_id['E000'].mean('pr_auc'):+.4f}")


if __name__ == "__main__":
    main()