"""Every decision the loop makes, expressed as pure functions of the research state.

The LLM proposes; this module decides. Routing, candidate selection, slot closure
and stopping are all deterministic, so a session can be replayed and every choice
explained by a rule rather than by what a model happened to say.

Routing is coverage-driven rather than greedy. The deliverable is a baseline, so
closing out the standard modelling decisions is worth more than squeezing the most
promising direction: a decision tested and found not to matter removes work from
the reader, which a marginally higher score does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .complexity import ast_node_count, nesting_depth
from .contracts import (
    ControlRole,
    DataContract,
    DecisionSlot,
    Evidence,
    HypothesisStatus,
    ModelConfig,
    ResearchState,
    SlotStatus,
)
from .fingerprint import shared_prefix_length

# Fraction of the experiment budget held back to retest earlier rejections against
# the final incumbent. Without it, verdicts measured against an old control ship
# unconfirmed.
CONFIRMATION_RESERVE = 0.15

# Consecutive experiments without a promotion before a slot is considered closed.
SLOT_PATIENCE = 3

# Consecutive experiments without a promotion, once coverage is complete, before
# the recommendation is treated as stable.
CONVERGENCE_PATIENCE = 5

MAX_PROBES_PER_SLOT = 3


class ActionKind(str, Enum):
    """Class of work the loop should do next. The LLM fills in the content."""

    PROBE = "probe"
    FEATURE = "feature"
    MODEL = "model"
    CONFIRM = "confirm"
    SIMPLIFY = "simplify"
    STOP = "stop"


class StopReason(str, Enum):
    BUDGET = "budget_exhausted"
    CONVERGED = "recommendation_stable"
    NO_WORK = "no_actionable_work"


@dataclass(frozen=True, slots=True)
class ActionPlan:
    """The next action, with the rule that produced it recorded for the audit trail."""

    kind: ActionKind
    slot_id: str | None
    rationale: str
    allow_probe: bool = False
    target_experiment_id: str | None = None
    stop_reason: StopReason | None = None


@dataclass(frozen=True, slots=True)
class Proposal:
    """One candidate from the LLM, before any deterministic filtering."""

    proposal_id: str
    slot_id: str
    cells: tuple[str, ...]
    model: ModelConfig
    rationale: str


@dataclass(frozen=True, slots=True)
class Selection:
    """Chosen proposal plus the ones dropped, so rejections stay explainable."""

    chosen: Proposal | None
    rejected: tuple[tuple[str, str], ...]


# Standard decisions, in the order a data scientist would normally settle them.
# Each entry: slot_id, label, options, action kind.
_SLOT_CATALOGUE: tuple[tuple[str, str, tuple[str, ...], ActionKind], ...] = (
    (
        "missing_handling",
        "How should missing values be handled",
        ("native", "median_or_mode", "indicator_plus_fill"),
        ActionKind.FEATURE,
    ),
    (
        "categorical_encoding",
        "How should categorical columns be encoded",
        ("native", "one_hot", "frequency", "target_encoding"),
        ActionKind.FEATURE,
    ),
    (
        "model_family",
        "Which gradient boosting implementation fits best",
        ("lightgbm", "xgboost", "catboost"),
        ActionKind.MODEL,
    ),
    (
        "imbalance_handling",
        "Does the class imbalance need explicit handling",
        ("none", "class_weight", "scale_pos_weight"),
        ActionKind.MODEL,
    ),
    (
        "target_transform",
        "Does the target need a transform",
        ("none", "log1p", "quantile"),
        ActionKind.FEATURE,
    ),
    (
        "feature_interaction",
        "Do engineered interactions or aggregates help",
        ("none", "ratios", "aggregates", "binning"),
        ActionKind.FEATURE,
    ),
    (
        "hyperparameter_tuning",
        "How much does tuning move the result",
        ("defaults", "small_search"),
        ActionKind.MODEL,
    ),
)

SLOT_ACTIONS: Mapping[str, ActionKind] = {
    slot_id: kind for slot_id, _, _, kind in _SLOT_CATALOGUE
}

# Scaling and normalisation are deliberately absent: tree ensembles are invariant to
# monotone rescaling, so testing them would spend budget to confirm a known null.
EXCLUDED_BY_DESIGN = ("feature_scaling", "normalisation", "pca")


def default_slots(contract: DataContract) -> tuple[DecisionSlot, ...]:
    """Build the checklist for this dataset, dropping decisions it cannot face.

    Conditioning on the contract is what keeps the checklist from encoding the
    shape of whichever dataset the system was developed against.
    """
    distribution = contract.target_distribution
    prevalence = distribution.prevalence
    has_categorical = any(spec.role == "categorical" for spec in contract.schema.values())
    has_missing = any(spec.missing_rate > 0.0 for spec in contract.schema.values())
    high_cardinality = any(
        spec.role == "categorical" and spec.cardinality > 15 for spec in contract.schema.values()
    )

    slots: list[DecisionSlot] = []
    for slot_id, label, options, _ in _SLOT_CATALOGUE:
        if slot_id == "missing_handling" and not has_missing:
            continue
        if slot_id == "categorical_encoding" and not has_categorical:
            continue
        if slot_id == "target_transform" and contract.task != "regression":
            continue
        if slot_id == "imbalance_handling":
            if contract.task != "binary" or prevalence is None or prevalence > 0.35:
                continue
        active = options
        if slot_id == "categorical_encoding" and not high_cardinality:
            active = tuple(o for o in options if o != "target_encoding")
        slots.append(DecisionSlot(slot_id=slot_id, label=label, options=active))
    return tuple(slots)


def slot_status_from_evidence(
    slot: DecisionSlot,
    evidences: Sequence[Evidence],
    patience: int = SLOT_PATIENCE,
) -> tuple[SlotStatus, str | None]:
    """Close a slot once it has a verdict, distinguishing a null from a blind spot."""
    if not evidences:
        return SlotStatus.OPEN, None

    statuses = [e.status for e in evidences]
    if HypothesisStatus.SUPPORTED in statuses:
        best = max(
            (e for e in evidences if e.status is HypothesisStatus.SUPPORTED),
            key=lambda e: e.mean_delta,
        )
        return SlotStatus.RESOLVED, f"improved by {best.mean_delta:+.4f}"

    tested = len(evidences)
    if all(e.underpowered for e in evidences) and tested >= patience:
        return SlotStatus.UNRESOLVABLE, "every comparison was underpowered"
    if tested >= min(patience, len(slot.options)):
        return SlotStatus.RESOLVED, "no option produced a measurable difference"
    return SlotStatus.IN_PROGRESS, None


def confirmation_queue(
    evidences: Sequence[Evidence],
    incumbent_fingerprint: str,
    limit: int,
) -> tuple[str, ...]:
    """Verdicts measured against a superseded control that deserve a retest.

    Feature interactions are real, so a decision rejected against an early
    incumbent may matter against the final one. Only near misses are queued;
    a clear rejection is unlikely to reverse.
    """
    stale = [
        e
        for e in evidences
        if e.control_role is ControlRole.INCUMBENT
        and e.control_fingerprint != incumbent_fingerprint
        and e.status is not HypothesisStatus.SUPPORTED
        and e.ci_high > 0.0
    ]
    stale.sort(key=lambda e: e.ci_high, reverse=True)
    return tuple(e.experiment_id for e in stale[:limit])


def stagnation(state: ResearchState, promotions_at: Sequence[int]) -> int:
    """Experiments run since the incumbent last advanced."""
    return state.experiments_used - (promotions_at[-1] if promotions_at else 0)


def route(
    state: ResearchState,
    contract: DataContract,
    evidences: Sequence[Evidence] = (),
    promotions_at: Sequence[int] = (),
    probes_used: Mapping[str, int] | None = None,
    simplification_attempted: bool = False,
) -> ActionPlan:
    """Choose the next class of action. Deterministic given the state.

    Order of precedence: budget, then confirmation of stale verdicts, then coverage
    of open slots, then simplification, then stop.
    """
    if state.budget_exhausted:
        return ActionPlan(
            ActionKind.STOP,
            None,
            "budget exhausted",
            stop_reason=StopReason.BUDGET,
        )

    remaining = state.experiments_budget - state.experiments_used
    reserve = max(1, int(round(CONFIRMATION_RESERVE * state.experiments_budget)))
    pending = confirmation_queue(evidences, state.incumbent_experiment_id, reserve)
    open_slots = state.open_slots

    if pending and (not open_slots or remaining <= reserve):
        return ActionPlan(
            ActionKind.CONFIRM,
            None,
            f"{len(pending)} verdicts were measured against a superseded incumbent",
            target_experiment_id=pending[0],
        )

    if open_slots:
        slot = open_slots[0]
        used = (probes_used or {}).get(slot.slot_id, 0)
        return ActionPlan(
            kind=SLOT_ACTIONS.get(slot.slot_id, ActionKind.FEATURE),
            slot_id=slot.slot_id,
            rationale=f"slot {slot.slot_id} is still open ({state.coverage:.0%} coverage)",
            allow_probe=used < MAX_PROBES_PER_SLOT,
        )

    if not simplification_attempted:
        return ActionPlan(
            ActionKind.SIMPLIFY,
            None,
            "coverage complete; testing whether the incumbent can be reduced",
            target_experiment_id=state.incumbent_experiment_id,
        )

    if stagnation(state, promotions_at) >= CONVERGENCE_PATIENCE:
        return ActionPlan(
            ActionKind.STOP,
            None,
            f"no promotion in {CONVERGENCE_PATIENCE} experiments with full coverage",
            stop_reason=StopReason.CONVERGED,
        )

    return ActionPlan(
        ActionKind.STOP,
        None,
        "no open slots and nothing left to confirm",
        stop_reason=StopReason.NO_WORK,
    )


def static_cost(cells: Sequence[str]) -> tuple[int, int, int]:
    """Cheap pre-run complexity proxy: nodes, cell count, deepest nesting."""
    return (
        sum(ast_node_count(cell) for cell in cells),
        len(cells),
        max((nesting_depth(cell) for cell in cells), default=0),
    )


def select_proposal(
    proposals: Sequence[Proposal],
    seen_fingerprints: Sequence[str],
    incumbent_cells: Sequence[str],
    fingerprint_of: Mapping[str, str],
    max_cells: int,
) -> Selection:
    """Pick one proposal deterministically, recording why the others were dropped.

    Ranking prefers the simplest candidate, then the one sharing the longest prefix
    with the incumbent, since a shared prefix can be reused from cache. Creativity
    stays with the proposer; accountability stays here.
    """
    seen = set(seen_fingerprints)
    survivors: list[Proposal] = []
    rejected: list[tuple[str, str]] = []

    for proposal in proposals:
        fingerprint = fingerprint_of.get(proposal.proposal_id)
        if fingerprint is None:
            rejected.append((proposal.proposal_id, "no fingerprint supplied"))
        elif fingerprint in seen:
            rejected.append((proposal.proposal_id, "duplicate of an earlier experiment"))
        elif len(proposal.cells) > max_cells:
            rejected.append(
                (proposal.proposal_id, f"{len(proposal.cells)} cells exceeds limit {max_cells}")
            )
        else:
            survivors.append(proposal)

    if not survivors:
        return Selection(None, tuple(rejected))

    def rank(proposal: Proposal) -> tuple[int, int, int, int, str]:
        nodes, cells, depth = static_cost(proposal.cells)
        reuse = shared_prefix_length(incumbent_cells, proposal.cells)
        return (cells, depth, nodes, -reuse, proposal.proposal_id)

    ordered = sorted(survivors, key=rank)
    chosen = ordered[0]
    rejected += [(p.proposal_id, "ranked below the chosen proposal") for p in ordered[1:]]
    return Selection(chosen, tuple(rejected))


def invalid_rate(statuses: Sequence[str], window: int = 5) -> float:
    """Share of recent experiments that failed a guard, used as a health trigger."""
    recent = list(statuses)[-window:]
    if not recent:
        return 0.0
    return sum(1 for s in recent if s.startswith("invalid")) / len(recent)