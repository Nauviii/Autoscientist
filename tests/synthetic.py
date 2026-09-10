"""Synthetic datasets that exercise contract shapes Telco never reaches.

Not a model-quality benchmark. These cases answer a narrower question: does the
pipeline produce a sensible DataContract when the data does not look like the
development dataset? Each case isolates one axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Task = Literal["binary", "multiclass", "regression"]


@dataclass(frozen=True)
class SyntheticCase:
    """One dataset plus the contract properties EDA is expected to recover."""

    name: str
    frame: pd.DataFrame
    target: str
    task: Task
    expects_validation: str
    expects_group_candidate: str | None = None
    expects_group_column: str | None = None
    expects_time_column: str | None = None
    expects_underpowered: bool = False


def make_case(
    *,
    name: str,
    task: Task = "binary",
    n: int = 6000,
    n_numeric: int = 4,
    n_categorical: int = 3,
    cardinality: int = 4,
    prevalence: float = 0.30,
    n_classes: int = 3,
    with_datetime: bool = False,
    with_groups: bool = False,
    missing_rate: float = 0.0,
    target_skew: float = 0.0,
    seed: int = 0,
    **expectations: object,
) -> SyntheticCase:
    """Build one dataset varying a single axis of the contract space."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    for i in range(n_numeric):
        data[f"num_{i}"] = rng.normal(size=n)
    for i in range(n_categorical):
        data[f"cat_{i}"] = rng.integers(0, cardinality, size=n).astype(str)

    frame = pd.DataFrame(data)
    signal = frame[[c for c in frame if c.startswith("num_")]].to_numpy() @ rng.normal(size=n_numeric)

    if task == "binary":
        cutoff = np.quantile(signal, 1.0 - prevalence)
        frame["target"] = (signal + rng.normal(scale=0.8, size=n) > cutoff).astype(int)
    elif task == "multiclass":
        edges = np.quantile(signal, np.linspace(0, 1, n_classes + 1)[1:-1])
        frame["target"] = np.digitize(signal + rng.normal(scale=0.5, size=n), edges)
    else:
        values = signal + rng.normal(scale=0.5, size=n)
        frame["target"] = np.exp(values * target_skew) if target_skew else values

    if with_datetime:
        frame["event_time"] = pd.date_range("2023-01-01", periods=n, freq="h")
    if with_groups:
        frame["entity_id"] = rng.integers(0, max(n // 8, 2), size=n).astype(str)

    if missing_rate > 0.0:
        for col in [c for c in frame if c.startswith(("num_", "cat_"))]:
            mask = rng.random(n) < missing_rate
            frame.loc[mask, col] = np.nan

    return SyntheticCase(
        name=name,
        frame=frame,
        target="target",
        task=task,
        expects_validation=str(expectations.get("expects_validation", "stratified_kfold")),
        expects_group_candidate=expectations.get("expects_group_candidate"),  # type: ignore[arg-type]
        expects_group_column=expectations.get("expects_group_column"),  # type: ignore[arg-type]
        expects_time_column=expectations.get("expects_time_column"),  # type: ignore[arg-type]
        expects_underpowered=bool(expectations.get("expects_underpowered", False)),
    )


def contract_cases() -> list[SyntheticCase]:
    """The minimum set every release must pass before claiming general support."""
    return [
        make_case(name="binary_balanced", prevalence=0.50, seed=1),
        make_case(
            name="binary_extreme_imbalance",
            prevalence=0.02,
            seed=2,
            expects_underpowered=True,
        ),
        make_case(name="regression_plain", task="regression", expects_validation="kfold", seed=3),
        make_case(
            name="regression_skewed",
            task="regression",
            target_skew=1.5,
            expects_validation="kfold",
            seed=4,
        ),
        make_case(name="multiclass_5", task="multiclass", n_classes=5, seed=5),
        make_case(
            name="temporal",
            with_datetime=True,
            expects_validation="time_series",
            expects_time_column="event_time",
            seed=6,
        ),
        # Grouping is surfaced for confirmation rather than assumed, so the default
        # scheme stands until a human says otherwise.
        make_case(
            name="grouped",
            with_groups=True,
            expects_validation="stratified_kfold",
            expects_group_candidate="entity_id",
            seed=7,
        ),
        make_case(name="high_cardinality", cardinality=500, seed=8),
        make_case(name="wide", n_numeric=250, n_categorical=50, seed=9),
        make_case(name="heavy_missing", missing_rate=0.45, seed=10),
        make_case(name="tiny", n=300, seed=11, expects_underpowered=True),
    ]