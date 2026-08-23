"""Three baselines that anchor everything the session later claims.

Reporting a single baseline invites the oldest mistake in applied modelling: a
weak starting point makes any later gain look impressive. Three tiers make the
claim honest by separating what came from where.

  trivial    a constant prediction; the floor any model must clear
  reference  an untuned booster on minimally prepared data; the session anchor
  tuned      the same booster after a small fixed search; what tuning alone buys

The gap between reference and tuned matters most. Feature work that looks valuable
against an undertuned control often evaporates once the control is tuned too, so
measuring it up front keeps later comparisons from taking credit for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.complexity import build_complexity
from ..core.contracts import (
    ControlRole,
    DataContract,
    Evidence,
    ExperimentResult,
    ExperimentStatus,
    ModelConfig,
    ValidationSpec,
)
from ..core.fingerprint import experiment_fingerprint
from ..core.statistics import build_evidence, decide
from ..modeling.cv import METRICS_BY_TASK, CVResult, make_folds, run_cv, score
from ..modeling.registry import default_params
from ..ports import FoldSpec

TRIVIAL_ID = "E000"
REFERENCE_ID = "E001"
TUNED_ID = "E002"

# Written by us, not by an agent, but expressed as a cell so the baseline travels
# the same execution and fingerprinting path as everything that follows.
MINIMAL_PREPARATION = '''
import pandas as pd


class FeatureStep:
    """Minimal preparation: coerce numeric-looking text, leave the rest categorical."""

    def fit(self, X, y):
        self.numeric_ = []
        self.categorical_ = []
        self.fill_ = {}
        for column in X.columns:
            series = X[column]
            if series.dtype.kind in "ifb":
                self.numeric_.append(column)
                self.fill_[column] = series.median()
                continue
            coerced = series.astype("string").str.strip()
            numbers = coerced.str.fullmatch(r"-?\\d+(\\.\\d+)?").fillna(False)
            if numbers.mean() > 0.9:
                self.numeric_.append(column)
                values = coerced.where(numbers).astype("float64")
                self.fill_[column] = values.median()
            else:
                self.categorical_.append(column)
                self.fill_[column] = sorted(series.astype("string").dropna().unique())
        return self

    def transform(self, X):
        out = X.copy()
        for column in self.numeric_:
            if out[column].dtype.kind not in "ifb":
                coerced = out[column].astype("string").str.strip()
                out[column] = pd.to_numeric(coerced, errors="coerce")
            out[column] = out[column].fillna(self.fill_[column])
        for column in self.categorical_:
            out[column] = out[column].astype("string").astype(
                pd.CategoricalDtype(categories=self.fill_[column])
            )
        return out
'''

# Fixed and ordered, so the search is reproducible without drawing random numbers.
TUNING_GRID: tuple[Mapping[str, Any], ...] = (
    {"num_leaves": 15, "learning_rate": 0.05, "min_child_samples": 40},
    {"num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20},
    {"num_leaves": 31, "learning_rate": 0.02, "min_child_samples": 40},
    {"num_leaves": 63, "learning_rate": 0.02, "min_child_samples": 60},
    {"num_leaves": 15, "learning_rate": 0.10, "min_child_samples": 20},
    {"num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 100},
)


@dataclass(frozen=True, slots=True)
class BaselineOutcome:
    """The three anchors, the evidence between them, and where the ladder starts."""

    experiments: tuple[ExperimentResult, ...]
    evidences: tuple[Evidence, ...]
    reference_id: str
    incumbent_id: str
    tuning_gain: float
    chosen_params: Mapping[str, Any]

    @property
    def by_id(self) -> Mapping[str, ExperimentResult]:
        return {e.experiment_id: e for e in self.experiments}


def _cell_transform(cells: Sequence[str], executor: Any, limits: Any):
    """Adapt the executor into the per-fold transform run_cv expects."""

    def transform(train_x, train_y, valid_x):
        fold = FoldSpec(
            fold_index=0,
            repeat=0,
            train=tuple(train_x.index.map(executor.frame.index.get_loc)),
            validate=tuple(valid_x.index.map(executor.frame.index.get_loc)),
        )
        artifacts = executor.run_cells(cells, fold, limits)
        if artifacts.failed:
            raise RuntimeError(f"baseline preparation failed: {artifacts.error}")
        return artifacts.train, artifacts.validate

    return transform


def trivial_scores(
    target: pd.Series, folds: Sequence[FoldSpec], contract: DataContract
) -> CVResult:
    """Score a constant prediction: the training prevalence, or the training mean.

    Establishes the floor. On an imbalanced task this is exactly where PR-AUC sits
    with no model at all, which is the number a reported score should be read
    against.
    """
    metrics = METRICS_BY_TASK[contract.task]
    collected: dict[str, list[float]] = {m: [] for m in metrics}

    for fold in folds:
        train_y = target.iloc[list(fold.train)]
        valid_y = target.iloc[list(fold.validate)].to_numpy()

        if contract.task == "regression":
            prediction = np.full(len(valid_y), float(train_y.mean()))
        elif contract.task == "binary":
            prediction = np.full(len(valid_y), float(train_y.mean()))
        else:
            counts = train_y.value_counts(normalize=True).sort_index()
            prediction = np.tile(counts.to_numpy(), (len(valid_y), 1))

        for metric in metrics:
            collected[metric].append(score(contract.task, valid_y, prediction, metric))

    return CVResult(
        fold_scores={m: tuple(v) for m, v in collected.items()},
        fit_seconds=0.0,
        n_features=0,
    )


def _to_experiment(
    experiment_id: str,
    result: CVResult,
    model: ModelConfig,
    cells: Sequence[str],
    contract: DataContract,
    environment: str,
    slot_id: str | None = None,
    parent_id: str | None = None,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        fingerprint=experiment_fingerprint(
            cells=list(cells),
            model=model,
            validation=contract.validation,
            data_hash=contract.dataset_hash,
            environment=environment,
            seed=contract.seed,
        ),
        hypothesis_id=None,
        parent_id=parent_id,
        slot_id=slot_id,
        model=model,
        fold_scores=result.fold_scores,
        complexity=build_complexity(list(cells), result.n_features, result.fit_seconds),
        runtime_s=result.fit_seconds,
        token_cost=0,
        status=ExperimentStatus.COMPLETED,
    )


def search_hyperparameters(
    frame: pd.DataFrame,
    target: pd.Series,
    contract: DataContract,
    transform: Any,
    grid: Sequence[Mapping[str, Any]] = TUNING_GRID,
) -> Mapping[str, Any]:
    """Pick a configuration on a cheap protocol, then hand it to the full one.

    Searching on the evaluation protocol would select against the same folds the
    result is later reported on, inflating it. Five fits per candidate is enough to
    rank them; the winner is then measured properly.
    """
    cheap = ValidationSpec(
        kind=contract.validation.kind,
        k=5,
        repeats=1,
        seed=contract.seed,
        group_column=contract.validation.group_column,
        time_column=contract.validation.time_column,
    )
    folds = make_folds(cheap, target)
    metric = contract.primary_metric
    lower_is_better = metric in {"brier", "logloss", "rmse", "mae"}

    best_params: Mapping[str, Any] = grid[0]
    best_score = float("inf") if lower_is_better else float("-inf")

    for params in grid:
        model = ModelConfig(
            family="lightgbm", params={**default_params("lightgbm", contract.task), **params}
        )
        result = run_cv(
            frame=frame,
            target=target,
            folds=folds,
            model=model,
            task=contract.task,
            seed=contract.seed,
            transform=transform,
            metrics=(metric,),
        )
        value = result.mean(metric)
        if (value < best_score) if lower_is_better else (value > best_score):
            best_score, best_params = value, params

    return best_params


def run_baselines(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    contract: DataContract,
    executor: Any,
    limits: Any,
    environment: str,
    tune: bool = True,
) -> BaselineOutcome:
    """Establish the three anchors and the evidence between them."""
    folds = make_folds(contract.validation, target)
    cells = (MINIMAL_PREPARATION,)
    transform = _cell_transform(cells, executor, limits)

    trivial = _to_experiment(
        TRIVIAL_ID,
        trivial_scores(target, folds, contract),
        ModelConfig(family="lightgbm", params={"constant": True}),
        (),
        contract,
        environment,
    )

    reference_model = ModelConfig(
        family="lightgbm", params=default_params("lightgbm", contract.task)
    )
    reference = _to_experiment(
        REFERENCE_ID,
        run_cv(
            frame=frame,
            target=target,
            folds=folds,
            model=reference_model,
            task=contract.task,
            seed=contract.seed,
            transform=transform,
        ),
        reference_model,
        cells,
        contract,
        environment,
        parent_id=TRIVIAL_ID,
    )

    experiments = [trivial, reference]
    evidences: list[Evidence] = []
    incumbent_id = REFERENCE_ID
    tuning_gain = 0.0
    chosen: Mapping[str, Any] = {}

    if tune:
        chosen = search_hyperparameters(frame, target, contract, transform)
        tuned_model = ModelConfig(
            family="lightgbm",
            params={**default_params("lightgbm", contract.task), **chosen},
        )
        tuned = _to_experiment(
            TUNED_ID,
            run_cv(
                frame=frame,
                target=target,
                folds=folds,
                model=tuned_model,
                task=contract.task,
                seed=contract.seed,
                transform=transform,
            ),
            tuned_model,
            cells,
            contract,
            environment,
            slot_id="hyperparameter_tuning",
            parent_id=REFERENCE_ID,
        )
        experiments.append(tuned)

        verdict = decide(tuned.fold_scores, reference.fold_scores, contract)
        evidences.append(
            build_evidence(
                evidence_id="EV_tuning",
                experiment_id=TUNED_ID,
                hypothesis_id=None,
                control_experiment_id=REFERENCE_ID,
                control_fingerprint=reference.fingerprint,
                control_role=ControlRole.INCUMBENT,
                verdict=verdict,
            )
        )
        tuning_gain = verdict.primary.mean_delta
        if verdict.status.value == "supported":
            incumbent_id = TUNED_ID

    lift = decide(reference.fold_scores, trivial.fold_scores, contract)
    evidences.append(
        build_evidence(
            evidence_id="EV_reference",
            experiment_id=REFERENCE_ID,
            hypothesis_id=None,
            control_experiment_id=TRIVIAL_ID,
            control_fingerprint=trivial.fingerprint,
            control_role=ControlRole.REFERENCE,
            verdict=lift,
        )
    )

    return BaselineOutcome(
        experiments=tuple(experiments),
        evidences=tuple(evidences),
        reference_id=REFERENCE_ID,
        incumbent_id=incumbent_id,
        tuning_gain=tuning_gain,
        chosen_params=chosen,
    )