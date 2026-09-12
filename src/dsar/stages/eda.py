"""Data understanding, blocks A through G. Every block feeds a specific decision.

The discipline that keeps this from becoming a data dump: a block earns its place
only by filling a field of the DataContract. Anything interesting but unused is
left for a probe to fetch on demand, where the request is recorded as part of the
reasoning trail rather than paid for on every run.

Column statistics are computed on an exploration slice rather than the full data.
The profile reaches the model, so designing features against statistics drawn from
the validation rows would leak through the agent's judgement while every guard
stayed green. The cost is a little precision in the profile; the gain is that the
separation holds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ..core.contracts import (
    ColumnSpec,
    DataContract,
    LeakageFlag,
    Severity,
    TargetDistribution,
    Task,
    ValidationSpec,
)
from ..core.dossier import Dossier, apply_dossier, settles_grouping
from ..core.statistics import build_power_profile

# A single column that predicts this well is almost always leakage rather than a
# strong feature.
SINGLE_COLUMN_ALARM = 0.95
NEAR_PERFECT_ASSOCIATION = 0.98
ID_UNIQUE_RATIO = 0.99
MISSINGNESS_ALARM = 0.70
HIGH_CARDINALITY = 15

# Share of values that must parse before a text column is read as numeric or as a date.
NUMERIC_COERCION_RATE = 0.90
DATETIME_COERCION_RATE = 0.90

# A depth-2 stump cannot express a monotone relationship, so regression needs more
# room before its R2 says anything about how much one column explains.
REGRESSION_TREE_DEPTH = 4

# One column explaining this much of a continuous target is suspicious in the same
# way a near-perfect classifier is.
REGRESSION_POWER_ALARM = 0.85
MULTICLASS_LIMIT = 20

# Below this, a grouping key repeats often enough that rows within a group cannot
# be treated as independent.
GROUP_REPEAT_RATIO = 0.60

# Share of distinct values occurring exactly once. A key repeats nearly all of its
# values, while a measurement leaves a third or more appearing a single time. The
# cost of the stricter bound is missing a key whose groups hold only two rows, and
# such a key leaks little anyway.
GROUP_SINGLETON_LIMIT = 0.20


@dataclass(frozen=True, slots=True)
class Integrity:
    """Block A. Structural problems that must be settled before anything else."""

    n_rows: int
    n_columns: int
    duplicate_rows: int
    constant_columns: tuple[str, ...]
    empty_columns: tuple[str, ...]
    memory_mb: float


@dataclass(frozen=True, slots=True)
class Structure:
    """Block E. What the row grain and the time axis imply about resampling.

    requires_confirmation marks a recommendation that data alone cannot settle. A
    grouping key and a discretised measurement are indistinguishable by their
    statistics, so the choice belongs to the human gate rather than a heuristic.
    """

    datetime_columns: tuple[str, ...]
    ordered_by_time: bool
    group_candidates: tuple[str, ...]
    recommended: Literal["stratified_kfold", "kfold", "group_kfold", "time_series"]
    reason: str
    requires_confirmation: bool = False


@dataclass(frozen=True, slots=True)
class EDAArtifact:
    """Full-fidelity record. Only a bounded projection of this reaches the model."""

    dataset_hash: str
    integrity: Integrity
    target: TargetDistribution
    schema: Mapping[str, ColumnSpec]
    leakage: tuple[LeakageFlag, ...]
    structure: Structure
    exploration_rows: int
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocking(self) -> tuple[LeakageFlag, ...]:
        return tuple(f for f in self.leakage if f.severity is Severity.BLOCK)

    @property
    def to_confirm(self) -> tuple[LeakageFlag, ...]:
        return tuple(f for f in self.leakage if f.severity is Severity.WARN)


def dataset_hash(frame: pd.DataFrame) -> str:
    """Content hash over the values, so an unchanged file yields an unchanged id."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    ).hexdigest()[:16]


def exploration_index(target: pd.Series, fraction: float, seed: int) -> np.ndarray:
    """Deterministic slice reserved for looking at the data.

    Stratified where possible so a rare class is still visible in the profile.
    """
    rng = np.random.default_rng(seed)
    size = max(50, int(len(target) * fraction))
    if target.nunique() <= MULTICLASS_LIMIT:
        picked: list[np.ndarray] = []
        for value in sorted(target.unique()):
            positions = np.flatnonzero((target == value).to_numpy())
            take = max(1, round(size * len(positions) / len(target)))
            picked.append(rng.choice(positions, min(take, len(positions)), replace=False))
        return np.sort(np.concatenate(picked))
    return np.sort(rng.choice(len(target), min(size, len(target)), replace=False))


def block_a_integrity(frame: pd.DataFrame) -> Integrity:
    """Structural checks that decide which columns are worth profiling at all."""
    constant = tuple(c for c in frame.columns if frame[c].nunique(dropna=False) <= 1)
    empty = tuple(c for c in frame.columns if frame[c].isna().all())
    return Integrity(
        n_rows=len(frame),
        n_columns=frame.shape[1],
        duplicate_rows=int(frame.duplicated().sum()),
        constant_columns=constant,
        empty_columns=empty,
        memory_mb=float(frame.memory_usage(deep=True).sum()) / 1e6,
    )


def infer_task(target: pd.Series) -> Task:
    """Binary, multiclass or regression, from the target alone."""
    unique = target.nunique(dropna=True)
    if unique == 2:
        return "binary"
    if not pd.api.types.is_numeric_dtype(target) or (
        pd.api.types.is_integer_dtype(target) and unique <= MULTICLASS_LIMIT
    ):
        return "multiclass"
    return "regression"


def encode_target(target: pd.Series, task: Task) -> pd.Series:
    """Normalise the target into the shape the metrics expect."""
    if task == "regression":
        return target.astype(float)
    if task == "binary":
        if pd.api.types.is_numeric_dtype(target):
            return (target > target.min()).astype(int)
        text = target.astype(str).str.strip().str.lower()
        positive = {"yes", "true", "1", "y", "churn", "positive"}
        if text.isin(positive).any():
            return text.isin(positive).astype(int)
        return (text == sorted(text.unique())[-1]).astype(int)
    return pd.Series(pd.factorize(target, sort=True)[0], index=target.index)


def block_b_target(target: pd.Series, task: Task) -> TargetDistribution:
    """Block B. The target's shape drives metric choice and the power calculation."""
    if task == "regression":
        return TargetDistribution(
            task=task,
            n_rows=len(target),
            mean=float(target.mean()),
            std=float(target.std(ddof=1)),
            skew=float(target.skew()),
        )
    counts = target.value_counts().sort_index()
    return TargetDistribution(
        task=task,
        n_rows=len(target),
        n_positive=int(target.sum()) if task == "binary" else None,
        class_counts={str(k): int(v) for k, v in counts.items()},
    )


def coerce_numeric(series: pd.Series) -> pd.Series | None:
    """Return the numeric reading of a text column, or None if it is not one.

    Numbers stored as text are common and easy to miss: left as categorical they
    acquire thousands of levels, which distorts association measures and invites a
    false leakage flag.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    coerced = pd.to_numeric(series.astype("string").str.strip(), errors="coerce")
    return coerced if coerced.notna().mean() > NUMERIC_COERCION_RATE else None


def coerce_datetime(series: pd.Series) -> pd.Series | None:
    """Return the datetime reading of a text column, or None if it is not one.

    CSV readers hand back dates as plain strings, so without this the temporal path
    is unreachable on any real file: the column looks categorical and the validation
    scheme never becomes time_series.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if pd.api.types.is_numeric_dtype(series):
        return None
    try:
        coerced = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return None
    return coerced if coerced.notna().mean() > DATETIME_COERCION_RATE else None


def classify_column(name: str, series: pd.Series) -> ColumnSpec:
    """Block C. One column, profiled with the fields later blocks actually read."""
    numeric = coerce_numeric(series)
    # Dates are only considered once numeric parsing has failed, or a bare year
    # column would be read as a timestamp.
    stamps = coerce_datetime(series) if numeric is None else None
    values = numeric if numeric is not None else (stamps if stamps is not None else series)
    unique = int(values.nunique(dropna=False))
    ratio = unique / max(len(values), 1)

    # Identity is checked before type. A row counter parses as a number, so testing
    # numeric first would let it through as a feature; a continuous float is also
    # near-unique but is a measurement, which is why floats are exempt.
    near_unique = ratio > ID_UNIQUE_RATIO and not pd.api.types.is_float_dtype(values)

    if stamps is not None:
        role: Any = "datetime"
    elif near_unique:
        role = "id"
    elif numeric is not None:
        role = "numeric"
    else:
        role = "categorical"

    summary = None
    if role == "numeric":
        described = values.describe()
        summary = {
            key: float(described[key]) for key in ("mean", "std", "min", "50%", "max")
        }

    return ColumnSpec(
        name=name,
        dtype=str(series.dtype),
        role=role,
        missing_rate=float(values.isna().mean()),
        cardinality=unique,
        unique_ratio=float(ratio),
        top_values=tuple(str(v) for v in values.value_counts().head(5).index),
        numeric_summary=summary,
    )


def _single_column_power(series: pd.Series, target: pd.Series, task: Task) -> float:
    """How well one column predicts the target on its own, via a depth-2 stump."""
    values = series
    if not pd.api.types.is_numeric_dtype(values):
        values = pd.Series(pd.factorize(values)[0], index=series.index)
    x = values.fillna(values.median() if values.notna().any() else 0).to_numpy().reshape(-1, 1)

    if task == "regression":
        tree = DecisionTreeRegressor(
            max_depth=REGRESSION_TREE_DEPTH, random_state=0
        ).fit(x, target)
        explained = float(max(0.0, tree.score(x, target)))
        # A column that is a linear component of the target beats any shallow tree,
        # so take whichever reading is stronger.
        linear = float(abs(pd.Series(x.ravel()).corr(target.reset_index(drop=True))) ** 2)
        return max(explained, 0.0 if linear != linear else linear)
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, target)
    proba = tree.predict_proba(x)
    if proba.shape[1] < 2:
        return 0.5
    binary = target if task == "binary" else (target == target.mode().iloc[0]).astype(int)
    column = 1 if task == "binary" else 0
    try:
        return float(roc_auc_score(binary, proba[:, column]))
    except ValueError:
        return 0.5


def _cramers_v(series: pd.Series, target: pd.Series) -> float:
    """Bias-corrected association between two categorical variables.

    The uncorrected statistic rises toward one as the number of levels approaches
    the number of rows, since almost every cell then holds a single observation.
    Without the correction of Bergsma (2013) a near-unique column looks like perfect
    leakage on any dataset.
    """
    table = pd.crosstab(series.astype("string"), target)
    if table.size == 0 or min(table.shape) < 2:
        return 0.0

    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total < 2:
        return 0.0
    expected = np.outer(observed.sum(1), observed.sum(0)) / total
    chi2 = float(((observed - expected) ** 2 / np.where(expected == 0, 1, expected)).sum())

    rows, columns = table.shape
    phi2 = max(0.0, chi2 / total - (rows - 1) * (columns - 1) / (total - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (total - 1)
    columns_corrected = columns - (columns - 1) ** 2 / (total - 1)
    denominator = max(min(rows_corrected, columns_corrected) - 1, 1e-9)
    return float(np.sqrt(phi2 / denominator))


def block_d_leakage(
    frame: pd.DataFrame, target: pd.Series, task: Task, schema: Mapping[str, ColumnSpec]
) -> tuple[LeakageFlag, ...]:
    """Block D. Four per-column screens, run before any model is trained.

    The missingness test is the one people rarely run: an indicator of whether a
    value is missing can itself predict the outcome, which usually means the column
    was filled in after the outcome was known.

    Duplicate rows are checked in block A instead. Duplication is a property of the
    whole table, and on an exploration slice the two copies of a pair are unlikely
    to both be drawn.
    """
    flags: list[LeakageFlag] = []

    for name, spec in schema.items():
        series = frame[name]

        if spec.role == "id":
            flags.append(
                LeakageFlag(name, "id_like", Severity.BLOCK, spec.unique_ratio,
                            f"unique ratio {spec.unique_ratio:.3f}; carries no signal to generalise")
            )
            continue

        if spec.missing_rate < 1.0 and spec.role in ("numeric", "categorical"):
            power = _single_column_power(series, target, task)
            alarm = REGRESSION_POWER_ALARM if task == "regression" else SINGLE_COLUMN_ALARM
            if power > alarm:
                unit = "R2" if task == "regression" else "AUC"
                flags.append(
                    LeakageFlag(name, "single_column_power", Severity.WARN, power,
                                f"alone reaches {unit} {power:.3f}; "
                                "verify it exists at prediction time")
                )

        if task != "regression" and spec.role == "categorical":
            association = _cramers_v(series, target)
            if association > NEAR_PERFECT_ASSOCIATION:
                flags.append(
                    LeakageFlag(name, "near_perfect_association", Severity.WARN, association,
                                f"Cramer's V {association:.3f} with the target")
                )
        elif spec.role == "numeric" and task == "regression":
            correlation = abs(float(series.corr(target)))
            if correlation > NEAR_PERFECT_ASSOCIATION:
                flags.append(
                    LeakageFlag(name, "near_perfect_association", Severity.WARN, correlation,
                                f"correlation {correlation:.3f} with the target")
                )

        if 0.0 < spec.missing_rate < 1.0 and task != "regression":
            indicator = series.isna().astype(int)
            if indicator.nunique() > 1:
                try:
                    auc = float(roc_auc_score(target, indicator))
                except ValueError:
                    auc = 0.5
                signal = max(auc, 1.0 - auc)
                if signal > MISSINGNESS_ALARM:
                    flags.append(
                        LeakageFlag(name, "informative_missingness", Severity.WARN, signal,
                                    f"whether the value is missing predicts the target at {signal:.3f}")
                    )

    return tuple(flags)


def _singleton_ratio(series: pd.Series) -> float:
    """Fraction of distinct values seen exactly once."""
    counts = series.value_counts()
    return float((counts == 1).mean()) if len(counts) else 1.0


def _looks_like_key(series: pd.Series, floor: float) -> bool:
    """Whether a column repeats the way an identifier does rather than a measurement."""
    cardinality = series.nunique(dropna=False)
    if cardinality <= floor or cardinality / max(len(series), 1) >= GROUP_REPEAT_RATIO:
        return False
    if _has_fractional_values(series):
        return False
    return _singleton_ratio(series) < GROUP_SINGLETON_LIMIT


def _has_fractional_values(series: pd.Series) -> bool:
    """True when the column holds real measurements rather than discrete labels."""
    numeric = coerce_numeric(series)
    if numeric is None:
        return False
    values = numeric.dropna()
    return bool(len(values)) and not bool((values % 1 == 0).all())


def block_e_structure(
    frame: pd.DataFrame,
    task: Task,
    schema: Mapping[str, ColumnSpec],
    total_rows: int | None = None,
) -> Structure:
    """Block E. Choose a resampling scheme, but never silently.

    A date column does not make a task temporal, and a repeated key does not always
    mean grouped. Both are surfaced as recommendations for the human gate to confirm.
    """
    datetimes = tuple(n for n, s in schema.items() if s.role == "datetime")
    ordered = False
    if datetimes:
        parsed = coerce_datetime(frame[datetimes[0]])
        column = (parsed if parsed is not None else frame[datetimes[0]]).dropna()
        ordered = bool(column.is_monotonic_increasing) and len(column) > 1

    # A key has many distinct values while repeating; a measurement binned into
    # levels leaves many appearing once. The n^0.6 threshold sits between the two,
    # but the separation is not reliable enough to act on without asking.
    #
    # Every statistic here is taken from the whole frame rather than the exploration
    # slice. On a sixth of the rows a key with a dozen members per group looks like a
    # column of singletons, which is the opposite of the signal being tested for.
    floor = len(frame) ** 0.6
    groups = tuple(
        name
        for name, spec in schema.items()
        if spec.role != "datetime"
        and name in frame.columns
        and _looks_like_key(frame[name], floor)
    )

    if ordered:
        return Structure(datetimes, ordered, groups, "time_series",
                         f"{datetimes[0]} is monotonic; rows appear ordered in time")

    default = "kfold" if task == "regression" else "stratified_kfold"
    if groups:
        return Structure(
            datetimes,
            ordered,
            groups,
            default,  # type: ignore[arg-type]
            f"{groups[0]} repeats across rows and may be a grouping key; "
            f"confirm before switching to group_kfold",
            requires_confirmation=True,
        )
    reason = (
        "continuous target, no grouping"
        if task == "regression"
        else "discrete target with no grouping or time order"
    )
    return Structure(datetimes, ordered, groups, default, reason)  # type: ignore[arg-type]


def choose_metric(task: Task, distribution: TargetDistribution) -> tuple[str, str, tuple[str, ...]]:
    """Block G. Primary metric, its noise family, and the secondaries to gate on."""
    if task == "binary":
        prevalence = distribution.prevalence or 0.5
        if min(prevalence, 1 - prevalence) < 0.40:
            return "pr_auc", "pr_auc", ("roc_auc", "brier", "logloss")
        return "roc_auc", "roc_auc", ("pr_auc", "brier", "logloss")
    if task == "multiclass":
        return "macro_f1", "accuracy", ("accuracy", "logloss")
    return "r2", "r2", ("rmse", "mae")


def run_eda(
    *,
    frame: pd.DataFrame,
    target_column: str,
    delta_practical: float = 0.01,
    k: int = 10,
    repeats: int = 5,
    seed: int = 42,
    exploration_frac: float = 0.15,
    expected_score: float | None = None,
    secondary_gates: Mapping[str, float] | None = None,
    primary_metric: str | None = None,
    dossier: Dossier | None = None,
) -> tuple[EDAArtifact, DataContract]:
    """Run every block and assemble the contract the rest of the session inherits.

    The dossier is applied last, so stated knowledge overrides what was inferred
    while the inferred findings it did not mention are kept.
    """
    digest = dataset_hash(frame)
    raw_target = frame[target_column]
    task = infer_task(raw_target)
    target = encode_target(raw_target, task)
    features = frame.drop(columns=[target_column])

    integrity = block_a_integrity(frame)
    distribution = block_b_target(target, task)

    explore = exploration_index(target, exploration_frac, seed)
    sample = features.iloc[explore]
    sample_target = target.iloc[explore]

    schema = {name: classify_column(name, sample[name]) for name in sample.columns}
    schema = {
        name: spec
        for name, spec in schema.items()
        if name not in integrity.constant_columns and name not in integrity.empty_columns
    }

    leakage = list(block_d_leakage(sample, sample_target, task, schema))
    if integrity.duplicate_rows:
        leakage.append(
            LeakageFlag(
                "<rows>",
                "duplicate_rows",
                Severity.WARN,
                float(integrity.duplicate_rows),
                f"{integrity.duplicate_rows} identical rows may span train and validation",
            )
        )
    leakage = tuple(leakage)
    # The full frame, not the slice: group structure is a property of every row.
    structure = block_e_structure(features, task, schema, total_rows=len(frame))

    excluded = tuple(sorted(
        {f.column for f in leakage if f.severity is Severity.BLOCK and f.column != "<rows>"}
        | set(integrity.constant_columns)
        | set(integrity.empty_columns)
    ))
    schema = {name: spec for name, spec in schema.items() if name not in excluded}

    inferred, family, secondaries = choose_metric(task, distribution)
    primary = primary_metric or inferred
    validation = ValidationSpec(
        kind=structure.recommended,
        k=k,
        repeats=repeats,
        seed=seed,
        group_column=structure.group_candidates[0] if structure.recommended == "group_kfold" else None,
        time_column=structure.datetime_columns[0] if structure.recommended == "time_series" else None,
    )

    defaults = {"binary": 0.84, "multiclass": 0.70, "regression": 0.55}
    score = expected_score if expected_score is not None else defaults[task]
    power = build_power_profile(
        n_rows=len(frame),
        n_pos=distribution.n_positive,
        validation=validation,
        metric_family=family,  # type: ignore[arg-type]
        expected_score=score,
        expected_auc=score if family in ("pr_auc", "roc_auc") else None,
        delta_practical=delta_practical,
    )

    warnings: list[str] = []
    if power.underpowered:
        warnings.append("typical feature-engineering gains fall below what this dataset resolves")
    if integrity.duplicate_rows:
        warnings.append(f"{integrity.duplicate_rows} duplicate rows")
    if structure.requires_confirmation and not settles_grouping(dossier):
        warnings.append(
            f"possible grouping key ({', '.join(structure.group_candidates[:3])}); "
            "confirm whether rows within a group must stay in the same fold"
        )
    # Restricted to categorical columns: continuous numerics are high-cardinality by
    # nature, and would otherwise trigger this on every dataset.
    if any(
        spec.role == "categorical" and spec.cardinality > HIGH_CARDINALITY
        for spec in schema.values()
    ):
        warnings.append("high-cardinality categoricals present; target encoding is worth testing")

    contract = DataContract(
        dataset_hash=digest,
        task=task,
        target=target_column,
        target_distribution=distribution,
        schema=schema,
        validation=validation,
        primary_metric=primary,
        metric_family=family,
        secondary_metrics=secondaries,
        secondary_gates=dict(secondary_gates or {}),
        delta_practical=delta_practical,
        power=power,
        leakage_flags=leakage,
        excluded_columns=excluded,
        exploration_frac=exploration_frac,
        seed=seed,
    )

    if dossier is not None:
        contract = apply_dossier(contract, dossier)
        schema = contract.schema
        excluded = contract.excluded_columns

    artifact = EDAArtifact(
        dataset_hash=digest,
        integrity=integrity,
        target=distribution,
        schema=schema,
        leakage=leakage,
        structure=structure,
        exploration_rows=len(explore),
        warnings=tuple(warnings),
    )
    return artifact, contract


def prepared_frame(frame: pd.DataFrame, contract: DataContract) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the contract's exclusions and target encoding. No feature work."""
    target = encode_target(frame[contract.target], contract.task)
    features = frame.drop(columns=[contract.target, *contract.excluded_columns], errors="ignore")
    return features, target