"""Run every config in one pass and put the results side by side.

A single dataset only ever proves the pipeline handles that dataset. Running all of
them together is what shows the decisions are derived rather than hardcoded: the
task, the resampling scheme, the metric and the detectable effect all come out
different, from the same code path.

Usage:
    uv run python scripts/run_all.py
    uv run python scripts/run_all.py --configs configs --no-tune
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dsar.modeling.registry import apply_determinism_env

apply_determinism_env()

import pandas as pd  # noqa: E402

from dsar.adapters.config import load_research_config  # noqa: E402
from dsar.adapters.sandbox import ForkExecutor  # noqa: E402
from dsar.core.contracts import ExperimentStatus  # noqa: E402
from dsar.ports import ResourceLimits  # noqa: E402
from dsar.stages.baseline import REFERENCE_ID, TRIVIAL_ID, run_baselines  # noqa: E402
from dsar.stages.eda import prepared_frame, run_eda  # noqa: E402

LIMITS = ResourceLimits(memory_mb=8192, cpu_seconds=1800, wall_seconds=1800, max_features=1000)


@dataclass(frozen=True, slots=True)
class Row:
    """One dataset's outcome, reduced to what a comparison needs."""

    name: str
    rows: int
    features: int
    task: str
    validation: str
    fits: int
    metric: str
    delta_min: float
    floor: float
    reference: float
    tuning: str
    flags: int
    seconds: float


def run_one(path: Path, tune: bool) -> Row:
    """Run one config end to end and reduce it to a comparison row."""
    started = time.monotonic()
    config = load_research_config(path)
    raw = pd.read_csv(config.dataset)

    artifact, contract = run_eda(
        frame=raw,
        target_column=config.target,
        dossier=config.dossier,
        primary_metric=config.evaluation.primary_metric,
        delta_practical=config.evaluation.delta_practical,
        secondary_gates=config.evaluation.secondary_gates,
        k=config.evaluation.k,
        repeats=config.evaluation.repeats,
        seed=config.seed,
    )
    features, target = prepared_frame(raw, contract)
    outcome = run_baselines(
        frame=features,
        target=target,
        contract=contract,
        executor=ForkExecutor(features, target),
        limits=LIMITS,
        environment=contract.dataset_hash,
        tune=tune,
    )

    tuned = outcome.by_id.get("E002")
    if tuned is None:
        verdict = "skipped"
    elif tuned.status is ExperimentStatus.DUPLICATE:
        verdict = "no change"
    else:
        evidence = next((e for e in outcome.evidences if e.evidence_id == "EV_tuning"), None)
        verdict = evidence.status.value if evidence else "-"

    metric = contract.primary_metric
    return Row(
        name=path.stem,
        rows=len(raw),
        features=len(contract.schema),
        task=contract.task,
        validation=contract.validation.kind,
        fits=contract.validation.n_fits,
        metric=metric,
        delta_min=contract.delta_min,
        floor=outcome.by_id[TRIVIAL_ID].mean(metric),
        reference=outcome.by_id[REFERENCE_ID].mean(metric),
        tuning=verdict,
        flags=len(artifact.leakage),
        seconds=time.monotonic() - started,
    )


def render(rows: list[Row]) -> str:
    """Fixed-width comparison table sized to its contents."""
    headers = (
        "config", "rows", "feat", "task", "validation", "fits",
        "metric", "delta_min", "floor", "reference", "tuning", "flags", "sec",
    )
    body = [
        (
            r.name, f"{r.rows:,}", str(r.features), r.task, r.validation, str(r.fits),
            r.metric, f"{r.delta_min:.4f}", f"{r.floor:.4f}", f"{r.reference:.4f}",
            r.tuning, str(r.flags), f"{r.seconds:.0f}",
        )
        for r in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body)) + 2 for i in range(len(headers))
    ]
    lines = ["".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines += ["".join(c.ljust(w) for c, w in zip(row, widths)).rstrip() for row in body]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every config and compare.")
    parser.add_argument("--configs", type=Path, default=Path("configs"))
    parser.add_argument("--no-tune", action="store_true")
    args = parser.parse_args()

    # system.yaml holds engine defaults; research.yaml is the scaffold placeholder
    # that the per-dataset files replaced.
    skip = {"system.yaml", "research.yaml"}
    paths = sorted(p for p in args.configs.glob("*.yaml") if p.name not in skip)
    if not paths:
        sys.exit(f"no configs found in {args.configs}")

    rows: list[Row] = []
    for path in paths:
        try:
            rows.append(run_one(path, tune=not args.no_tune))
            print(f"  {path.stem:<10} done in {rows[-1].seconds:.0f}s")
        except FileNotFoundError as exc:
            print(f"  {path.stem:<10} skipped: {exc.filename or exc} not present")
        except Exception as exc:  # noqa: BLE001 - one bad config must not stop the sweep
            print(f"  {path.stem:<10} FAILED: {type(exc).__name__}: {exc}")

    if not rows:
        sys.exit("nothing ran")

    print("\n" + render(rows))
    print(
        "\nfloor is the trivial prediction: the base rate for PR-AUC, zero for R2. "
        "A tuning verdict other than 'supported' means the incumbent stayed put."
    )


if __name__ == "__main__":
    main()
