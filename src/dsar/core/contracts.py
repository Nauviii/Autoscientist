"""Frozen data contracts shared across every stage. Pure: no I/O, no LLM, no clock."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Mapping

Task = Literal["binary", "multiclass", "regression"]


def canonical_hash(payload: object) -> str:
    """Stable sha256 over a JSON-serialisable payload with sorted keys."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Severity(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    INFO = "info"


class HypothesisStatus(str, Enum):
    """Verdicts. PARTIALLY_SUPPORTED is reserved for mixed multi-experiment evidence."""

    PROPOSED = "proposed"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    COMPLETED = "completed"
    DUPLICATE = "duplicate"
    INVALID_CODE = "invalid_code"
    INVALID_GUARD = "invalid_guard"
    INVALID_BUDGET = "invalid_budget"


class ControlRole(str, Enum):
    """What an experiment was compared against. Recorded on every piece of evidence."""

    REFERENCE = "reference"
    INCUMBENT = "incumbent"


class SlotStatus(str, Enum):
    """Coverage state of a standard modelling decision."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    UNRESOLVABLE = "unresolvable"


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """Profile of a single raw column, produced by EDA block C."""

    name: str
    dtype: str
    role: Literal["numeric", "categorical", "datetime", "id", "target", "unknown"]
    missing_rate: float
    cardinality: int
    unique_ratio: float
    top_values: tuple[str, ...] = ()
    numeric_summary: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class TargetDistribution:
    """Target shape. Kept on the contract so power can be recomputed at any point."""

    task: Task
    n_rows: int
    n_positive: int | None = None
    class_counts: Mapping[str, int] | None = None
    mean: float | None = None
    std: float | None = None
    skew: float | None = None

    @property
    def prevalence(self) -> float | None:
        if self.n_positive is None:
            return None
        return self.n_positive / self.n_rows


@dataclass(frozen=True, slots=True)
class LeakageFlag:
    """One finding from the deterministic leakage screen (EDA block D)."""

    column: str
    test: str
    severity: Severity
    statistic: float
    detail: str


@dataclass(frozen=True, slots=True)
class ValidationSpec:
    """Resampling protocol. Fold indices are generated once and reused everywhere."""

    kind: Literal["stratified_kfold", "kfold", "group_kfold", "time_series"]
    k: int
    repeats: int
    seed: int
    group_column: str | None = None
    time_column: str | None = None

    @property
    def n_fits(self) -> int:
        return self.k * self.repeats

    @property
    def test_train_ratio(self) -> float:
        """n_test / n_train, the irreducible term in the Nadeau-Bengio correction."""
        return 1.0 / (self.k - 1)


@dataclass(frozen=True, slots=True)
class PowerProfile:
    """Output of EDA block F: what effect sizes this dataset can actually resolve."""

    marginal_se: float
    sd_estimate: float
    corrected_se: float
    mde: float
    protocol_table: tuple[tuple[str, int, float], ...]
    underpowered: bool
    method: str
    assumptions: tuple[str, ...]
    note: str
    empirical: bool = False


@dataclass(frozen=True, slots=True)
class DataContract:
    """Decisions from the understanding phase. Inherited by every later experiment."""

    dataset_hash: str
    task: Task
    target: str
    target_distribution: TargetDistribution
    schema: Mapping[str, ColumnSpec]
    validation: ValidationSpec
    primary_metric: str
    metric_family: str
    secondary_metrics: tuple[str, ...]
    secondary_gates: Mapping[str, float]
    delta_practical: float
    power: PowerProfile
    leakage_flags: tuple[LeakageFlag, ...]
    excluded_columns: tuple[str, ...]
    protected_columns: tuple[str, ...] = ()
    protected_policy: Literal["exclude", "audit", "allow"] = "audit"
    selection_rule: Literal[
        "min_complexity_within_equivalence", "argmax_utility"
    ] = "min_complexity_within_equivalence"
    exploration_frac: float = 0.15
    seed: int = 42

    @property
    def n_rows(self) -> int:
        return self.target_distribution.n_rows

    @property
    def delta_min(self) -> float:
        """Effective threshold: what is meaningful, floored by what is detectable."""
        return max(self.delta_practical, self.power.mde)

    def fingerprint(self) -> str:
        return canonical_hash(
            {
                "dataset": self.dataset_hash,
                "task": self.task,
                "target": self.target,
                "validation": [
                    self.validation.kind,
                    self.validation.k,
                    self.validation.repeats,
                    self.validation.seed,
                    self.validation.group_column,
                    self.validation.time_column,
                ],
                "primary": self.primary_metric,
                "gates": dict(sorted(self.secondary_gates.items())),
                "excluded": sorted(self.excluded_columns),
                "protected": [sorted(self.protected_columns), self.protected_policy],
                "selection_rule": self.selection_rule,
                "delta_min": round(self.delta_min, 6),
                "seed": self.seed,
            }
        )


@dataclass(frozen=True, slots=True)
class ComplexityVector:
    """Complexity components, kept unscalarised so the Pareto front stays visible."""

    n_features: int
    n_cells: int
    pipeline_depth: int
    fit_seconds: float
    ast_nodes: int

    def scalar(self, weights: Mapping[str, float], reference: "ComplexityVector") -> float:
        """Weighted sum of components normalised against a reference pipeline."""
        ratios = {
            "n_features": self.n_features / max(reference.n_features, 1),
            "n_cells": self.n_cells / max(reference.n_cells, 1),
            "pipeline_depth": self.pipeline_depth / max(reference.pipeline_depth, 1),
            "fit_seconds": self.fit_seconds / max(reference.fit_seconds, 1e-6),
            "ast_nodes": self.ast_nodes / max(reference.ast_nodes, 1),
        }
        return sum(weights.get(k, 0.0) * v for k, v in ratios.items())

    def dominates(self, other: "ComplexityVector") -> bool:
        """True when no worse on every component and strictly better on at least one."""
        pairs = (
            (self.n_features, other.n_features),
            (self.n_cells, other.n_cells),
            (self.pipeline_depth, other.pipeline_depth),
            (self.fit_seconds, other.fit_seconds),
            (self.ast_nodes, other.ast_nodes),
        )
        return all(a <= b for a, b in pairs) and any(a < b for a, b in pairs)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model selection is config, never agent-written code."""

    family: Literal["lightgbm", "xgboost", "catboost"]
    params: Mapping[str, object]
    n_threads: int = 4

    def canonical(self) -> Mapping[str, object]:
        return {
            "family": self.family,
            "params": dict(sorted(self.params.items())),
            "n_threads": self.n_threads,
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """One completed experiment. fold_scores stores every fold, never the mean."""

    experiment_id: str
    fingerprint: str
    hypothesis_id: str | None
    parent_id: str | None
    slot_id: str | None
    model: ModelConfig
    fold_scores: Mapping[str, tuple[float, ...]]
    complexity: ComplexityVector
    runtime_s: float
    token_cost: int
    status: ExperimentStatus
    artifact_ids: Mapping[str, str] = field(default_factory=dict)
    guard_failure: str | None = None

    def mean(self, metric: str) -> float:
        scores = self.fold_scores[metric]
        return sum(scores) / len(scores)


@dataclass(frozen=True, slots=True)
class Evidence:
    """A verdict bound to the exact control it was measured against.

    The incumbent advances during a session, so a verdict only means something
    alongside its control: an early rejection need not hold against the final one.
    """

    evidence_id: str
    experiment_id: str
    hypothesis_id: str | None
    control_experiment_id: str
    control_fingerprint: str
    control_role: ControlRole
    metric: str
    mean_delta: float
    ci_low: float
    ci_high: float
    delta_min: float
    status: HypothesisStatus
    underpowered: bool
    gate_violations: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Finding:
    """A conclusion phrased in the vocabulary used to assemble a baseline."""

    finding_id: str
    slot_id: str
    statement: str
    status: HypothesisStatus
    effect: float
    ci_low: float
    ci_high: float
    underpowered: bool
    evidence_ids: tuple[str, ...]

    @property
    def mattered(self) -> bool:
        return self.status is HypothesisStatus.SUPPORTED


@dataclass(frozen=True, slots=True)
class DecisionSlot:
    """A standard modelling decision the run is expected to close out.

    Routing is coverage-driven: open slots outrank further pursuit of a direction
    that already looks promising.
    """

    slot_id: str
    label: str
    options: tuple[str, ...]
    status: SlotStatus = SlotStatus.OPEN
    experiment_ids: tuple[str, ...] = ()
    resolution: str | None = None


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A testable claim, linked to the experiments that produced its evidence."""

    hypothesis_id: str
    slot_id: str
    statement: str
    rationale: str
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    experiment_ids: tuple[str, ...] = ()
    source: Literal["observation", "literature", "human"] = "observation"


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Read-only observation over the exploration split. Small and JSON-serialisable."""

    probe_id: str
    kind: str
    params: Mapping[str, object]
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ResearchState:
    """Fixed-size projection of the artifact store. Rebuilt from scratch each turn."""

    session_id: str
    contract_fingerprint: str
    reference_experiment_id: str
    reference_score: float
    incumbent_experiment_id: str
    incumbent_score: float
    incumbent_promotions: int
    slots: tuple[DecisionSlot, ...]
    active_hypotheses: tuple[Hypothesis, ...]
    recent_experiments: tuple[ExperimentResult, ...]
    experiments_used: int
    experiments_budget: int
    tokens_used: int
    tokens_budget: int
    seconds_used: float
    seconds_budget: float

    @property
    def budget_exhausted(self) -> bool:
        return (
            self.experiments_used >= self.experiments_budget
            or self.tokens_used >= self.tokens_budget
            or self.seconds_used >= self.seconds_budget
        )

    @property
    def open_slots(self) -> tuple[DecisionSlot, ...]:
        return tuple(s for s in self.slots if s.status is SlotStatus.OPEN)

    @property
    def coverage(self) -> float:
        """Fraction of standard decisions that now carry a verdict."""
        if not self.slots:
            return 1.0
        closed = sum(
            1 for s in self.slots if s.status in (SlotStatus.RESOLVED, SlotStatus.UNRESOLVABLE)
        )
        return closed / len(self.slots)


@dataclass(frozen=True, slots=True)
class BaselineRecipe:
    """The deliverable: a justified starting point, not a production model.

    Chosen as the simplest pipeline statistically indistinguishable from the final
    incumbent, so the recommendation is defensible rather than merely highest.
    """

    pipeline: tuple[str, ...]
    model: ModelConfig
    validation: ValidationSpec
    reference_score: float
    incumbent_score: float
    recommended_score: float
    holdout_score: float
    delta_vs_reference: tuple[float, float, float]
    delta_vs_incumbent: tuple[float, float, float]
    complexity: ComplexityVector
    equivalent_alternatives: tuple[str, ...]
    decisions_that_mattered: tuple[Finding, ...]
    decisions_that_did_not: tuple[Finding, ...]
    open_questions: tuple[str, ...]
    caveats: tuple[str, ...]
    out_of_scope: tuple[str, ...] = (
        "threshold tuning against business cost",
        "probability calibration for production",
        "temporal stability beyond the stated validation scheme",
        "fairness analysis beyond subgroup audit",
    )

    @property
    def selection_penalty(self) -> float:
        """Winner's curse: how far the selection-time score overstates the holdout."""
        return self.recommended_score - self.holdout_score