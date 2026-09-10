"""Cross-validation with fixed folds and per-fold scores.

Two invariants make the statistical layer valid, and both live here.

Folds are generated once from the contract seed and reused by every experiment in
the session. Paired comparison is meaningless otherwise: comparing an average over
one partition against an average over a different one measures the partitions as
much as the pipelines.

Scores are kept per fold, never averaged away. The mean is recoverable from the
folds; the paired deltas are not recoverable from the mean, and they are what the
decision rule needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from ..core.contracts import ModelConfig, Task, ValidationSpec
from ..ports import FoldSpec
from .registry import fit_predict

# Transform applied inside a fold: (train_x, train_y, valid_x) -> (train_x, valid_x).
# Supplied by the caller so this module stays unaware of how cells are executed.
FoldTransform = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]
]

METRICS_BY_TASK: Mapping[Task, tuple[str, ...]] = {
    "binary": ("pr_auc", "roc_auc", "brier", "logloss"),
    "multiclass": ("accuracy", "macro_f1", "logloss"),
    "regression": ("r2", "rmse", "mae"),
}

LOWER_IS_BETTER = frozenset({"brier", "logloss", "rmse", "mae"})


@dataclass(frozen=True, slots=True)
class CVResult:
    """Per-fold scores plus the timing the complexity vector needs."""

    fold_scores: Mapping[str, tuple[float, ...]]
    fit_seconds: float
    n_features: int

    def mean(self, metric: str) -> float:
        scores = self.fold_scores[metric]
        return sum(scores) / len(scores)


def make_folds(
    validation: ValidationSpec,
    target: pd.Series,
    groups: pd.Series | None = None,
) -> tuple[FoldSpec, ...]:
    """Generate every fold once. Reused by every experiment in the session.

    Temporal splits are never repeated, since forward chaining is fixed by the row
    order. Grouped splits are shuffled by seed, so repeats do produce distinct
    partitions and are honoured.
    """
    n = len(target)
    indices = np.arange(n)
    folds: list[FoldSpec] = []

    for repeat in range(validation.effective_repeats):
        seed = validation.seed + repeat

        if validation.kind == "stratified_kfold":
            splitter: Any = StratifiedKFold(validation.k, shuffle=True, random_state=seed)
            split = splitter.split(indices, target)
        elif validation.kind == "kfold":
            splitter = KFold(validation.k, shuffle=True, random_state=seed)
            split = splitter.split(indices)
        elif validation.kind == "group_kfold":
            if groups is None:
                raise ValueError("group_kfold requires a group column")
            splitter = StratifiedGroupKFold(validation.k, shuffle=True, random_state=seed)
            split = splitter.split(indices, target, groups)
        elif validation.kind == "time_series":
            splitter = TimeSeriesSplit(n_splits=validation.k)
            split = splitter.split(indices)
        else:
            raise ValueError(f"unsupported validation kind: {validation.kind}")

        for fold_index, (train, validate) in enumerate(split):
            folds.append(
                FoldSpec(
                    fold_index=fold_index,
                    repeat=repeat,
                    train=tuple(int(i) for i in train),
                    validate=tuple(int(i) for i in validate),
                )
            )
    return tuple(folds)


def score(task: Task, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """One metric on one fold. Raises on an unsupported name rather than guessing."""
    if task == "binary":
        if metric == "pr_auc":
            return float(average_precision_score(y_true, y_pred))
        if metric == "roc_auc":
            return float(roc_auc_score(y_true, y_pred))
        if metric == "brier":
            return float(brier_score_loss(y_true, y_pred))
        if metric == "logloss":
            return float(log_loss(y_true, np.clip(y_pred, 1e-7, 1 - 1e-7), labels=[0, 1]))
    elif task == "multiclass":
        labels = np.argmax(y_pred, axis=1)
        if metric == "accuracy":
            return float(accuracy_score(y_true, labels))
        if metric == "macro_f1":
            return float(f1_score(y_true, labels, average="macro"))
        if metric == "logloss":
            return float(log_loss(y_true, y_pred))
    else:
        if metric == "r2":
            return float(r2_score(y_true, y_pred))
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"metric {metric} is not defined for task {task}")


def run_cv(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[FoldSpec],
    model: ModelConfig,
    task: Task,
    seed: int,
    transform: FoldTransform | None = None,
    metrics: Sequence[str] | None = None,
    monotonic: Callable[[], float] | None = None,
) -> CVResult:
    """Fit and score every fold, keeping each fold's number.

    The transform is applied inside the fold, after the split, which is what makes
    fold safety structural rather than a rule the caller has to remember.
    """
    import time

    clock = monotonic or time.monotonic
    active = tuple(metrics or METRICS_BY_TASK[task])
    classes = sorted(target.unique()) if task != "regression" else None
    collected: dict[str, list[float]] = {m: [] for m in active}

    started = clock()
    n_features = frame.shape[1]

    for fold in folds:
        train_x = frame.iloc[list(fold.train)]
        train_y = target.iloc[list(fold.train)]
        valid_x = frame.iloc[list(fold.validate)]
        valid_y = target.iloc[list(fold.validate)]

        if transform is not None:
            train_x, valid_x = transform(train_x, train_y, valid_x)
            n_features = train_x.shape[1]

        predictions = fit_predict(model, task, train_x, train_y, valid_x, seed, classes)
        for metric in active:
            collected[metric].append(score(task, valid_y.to_numpy(), predictions, metric))

    return CVResult(
        fold_scores={m: tuple(v) for m, v in collected.items()},
        fit_seconds=clock() - started,
        n_features=n_features,
    )


@dataclass
class ControlMemo:
    """Cache of control scores, keyed by fingerprint.

    A paired experiment runs treatment against a control that usually has not
    changed, so refitting it every time doubles the cost of the session for nothing.
    """

    _scores: dict[str, CVResult] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._scores = {}

    def get(self, fingerprint: str) -> CVResult | None:
        return self._scores.get(fingerprint)

    def put(self, fingerprint: str, result: CVResult) -> None:
        self._scores.setdefault(fingerprint, result)

    def resolve(self, fingerprint: str, compute: Callable[[], CVResult]) -> tuple[CVResult, bool]:
        """Return the cached result, or compute and store it. Second value is hit."""
        cached = self.get(fingerprint)
        if cached is not None:
            return cached, True
        result = compute()
        self.put(fingerprint, result)
        return result, False

    def __len__(self) -> int:
        return len(self._scores)