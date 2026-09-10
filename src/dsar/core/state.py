"""Derive the research state from stored artifacts. Pure: a fold, never a mutation.

The state is recomputed from scratch every turn rather than accumulated. Two
consequences follow, and both are the point: it cannot drift, and it cannot grow.

Growth is the failure this prevents. Appending each experiment to the prompt makes
turn n carry O(n) history and the session O(n^2) tokens, while reasoning quality
falls as the context fills with material no longer relevant. Here the state is
capped regardless of how long the session runs; full fidelity stays in the store,
and only this projection reaches the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import (
    ControlRole,
    DataContract,
    DecisionSlot,
    Evidence,
    ExperimentResult,
    ExperimentStatus,
    Finding,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
    SlotStatus,
)
from .policy import slot_status_from_evidence

RECENT_EXPERIMENTS = 5
ACTIVE_HYPOTHESES = 6


@dataclass(frozen=True, slots=True)
class SessionBudget:
    """Ceilings for one session, all three enforced independently."""

    experiments: int
    tokens: int
    seconds: float


@dataclass(frozen=True, slots=True)
class IncumbentTrace:
    """Current incumbent plus the positions at which it advanced.

    The promotion positions are what let the router measure stagnation, and they
    also show how rarely the ladder actually moves.
    """

    experiment_id: str
    score: float
    promotions_at: tuple[int, ...]

    @property
    def promotions(self) -> int:
        return len(self.promotions_at)


def usage(experiments: Sequence[ExperimentResult]) -> tuple[int, int, float]:
    """Consumed experiments, tokens and seconds.

    Duplicates count against the experiment budget because they still cost an LLM
    call, even though the fits themselves were skipped.
    """
    return (
        len(experiments),
        sum(e.token_cost for e in experiments),
        sum(e.runtime_s for e in experiments),
    )


def resolve_incumbent(
    experiments: Sequence[ExperimentResult],
    evidences: Sequence[Evidence],
    reference_id: str,
    metric: str,
) -> IncumbentTrace:
    """Replay the promotion ladder in order to find the current incumbent.

    Only evidence measured against the incumbent and marked SUPPORTED promotes, so
    a run of positive but inconclusive deltas cannot walk the incumbent upward.
    """
    by_id = {e.experiment_id: e for e in experiments}
    if reference_id not in by_id:
        raise ValueError(f"reference experiment {reference_id} is not in the store")

    promoting = {
        ev.experiment_id
        for ev in evidences
        if ev.control_role is ControlRole.INCUMBENT and ev.status is HypothesisStatus.SUPPORTED
    }

    incumbent = by_id[reference_id]
    promotions: list[int] = []
    for position, experiment in enumerate(experiments, start=1):
        if experiment.experiment_id in promoting and experiment.status is ExperimentStatus.COMPLETED:
            incumbent = experiment
            promotions.append(position)

    return IncumbentTrace(
        experiment_id=incumbent.experiment_id,
        score=incumbent.mean(metric),
        promotions_at=tuple(promotions),
    )


def evidence_by_slot(
    experiments: Sequence[ExperimentResult], evidences: Sequence[Evidence]
) -> Mapping[str, tuple[Evidence, ...]]:
    """Group evidence by the decision slot the experiment was run against."""
    slot_of = {e.experiment_id: e.slot_id for e in experiments}
    grouped: dict[str, list[Evidence]] = {}
    for ev in evidences:
        slot_id = slot_of.get(ev.experiment_id)
        if slot_id:
            grouped.setdefault(slot_id, []).append(ev)
    return {slot_id: tuple(items) for slot_id, items in grouped.items()}


def apply_slot_statuses(
    slots: Sequence[DecisionSlot],
    experiments: Sequence[ExperimentResult],
    evidences: Sequence[Evidence],
) -> tuple[DecisionSlot, ...]:
    """Recompute every slot status from the evidence currently on record."""
    grouped = evidence_by_slot(experiments, evidences)
    updated: list[DecisionSlot] = []
    for slot in slots:
        items = grouped.get(slot.slot_id, ())
        status, resolution = slot_status_from_evidence(slot, items)
        updated.append(
            DecisionSlot(
                slot_id=slot.slot_id,
                label=slot.label,
                options=slot.options,
                status=status,
                experiment_ids=tuple(ev.experiment_id for ev in items),
                resolution=resolution,
            )
        )
    return tuple(updated)


def derive_findings(
    slots: Sequence[DecisionSlot],
    experiments: Sequence[ExperimentResult],
    evidences: Sequence[Evidence],
) -> tuple[Finding, ...]:
    """Turn closed slots into statements phrased for a human building a baseline.

    Written from a template rather than generated, so the report says exactly what
    the numbers support and nothing beyond it.
    """
    grouped = evidence_by_slot(experiments, evidences)
    findings: list[Finding] = []

    for slot in slots:
        items = grouped.get(slot.slot_id, ())
        if not items or slot.status is SlotStatus.OPEN:
            continue

        supported = [ev for ev in items if ev.status is HypothesisStatus.SUPPORTED]
        best = max(supported or items, key=lambda ev: ev.mean_delta)

        if supported:
            status = HypothesisStatus.SUPPORTED
            statement = f"{slot.label}: a measurable improvement of {best.mean_delta:+.4f}"
        elif slot.status is SlotStatus.UNRESOLVABLE:
            status = HypothesisStatus.INCONCLUSIVE
            statement = (
                f"{slot.label}: undecided, every comparison was below the detectable effect"
            )
        else:
            status = HypothesisStatus.REJECTED
            statement = (
                f"{slot.label}: no option changed the result beyond {best.delta_min:.4f}"
            )

        findings.append(
            Finding(
                finding_id=f"F_{slot.slot_id}",
                slot_id=slot.slot_id,
                statement=statement,
                status=status,
                effect=best.mean_delta,
                ci_low=best.ci_low,
                ci_high=best.ci_high,
                underpowered=all(ev.underpowered for ev in items),
                evidence_ids=tuple(ev.evidence_id for ev in items),
            )
        )
    return tuple(findings)


def split_findings(
    findings: Sequence[Finding],
) -> tuple[tuple[Finding, ...], tuple[Finding, ...], tuple[str, ...]]:
    """Separate what mattered, what did not, and what remains unknown.

    The middle group is the one that saves the reader work, and the third must not
    be folded into it: a measured null and a blind spot are opposite conclusions.
    """
    mattered = tuple(f for f in findings if f.status is HypothesisStatus.SUPPORTED)
    did_not = tuple(f for f in findings if f.status is HypothesisStatus.REJECTED)
    open_questions = tuple(
        f.statement for f in findings if f.status is HypothesisStatus.INCONCLUSIVE
    )
    return mattered, did_not, open_questions


def active_hypotheses(
    hypotheses: Sequence[Hypothesis], limit: int = ACTIVE_HYPOTHESES
) -> tuple[Hypothesis, ...]:
    """Keep only hypotheses still awaiting a verdict, newest first."""
    pending = [h for h in hypotheses if h.status is HypothesisStatus.PROPOSED]
    return tuple(reversed(pending[-limit:]))


def build_state(
    *,
    session_id: str,
    contract: DataContract,
    experiments: Sequence[ExperimentResult],
    evidences: Sequence[Evidence],
    slots: Sequence[DecisionSlot],
    hypotheses: Sequence[Hypothesis] = (),
    budget: SessionBudget,
    reference_id: str,
    recent: int = RECENT_EXPERIMENTS,
) -> tuple[ResearchState, IncumbentTrace]:
    """Fold the store into a bounded state. Identical stores yield identical states."""
    metric = contract.primary_metric
    by_id = {e.experiment_id: e for e in experiments}
    if reference_id not in by_id:
        raise ValueError(f"reference experiment {reference_id} is not in the store")

    trace = resolve_incumbent(experiments, evidences, reference_id, metric)
    used, tokens, seconds = usage(experiments)

    state = ResearchState(
        session_id=session_id,
        contract_fingerprint=contract.fingerprint(),
        reference_experiment_id=reference_id,
        reference_score=by_id[reference_id].mean(metric),
        incumbent_experiment_id=trace.experiment_id,
        incumbent_score=trace.score,
        incumbent_promotions=trace.promotions,
        slots=apply_slot_statuses(slots, experiments, evidences),
        active_hypotheses=active_hypotheses(hypotheses),
        recent_experiments=tuple(experiments[-recent:]),
        experiments_used=used,
        experiments_budget=budget.experiments,
        tokens_used=tokens,
        tokens_budget=budget.tokens,
        seconds_used=seconds,
        seconds_budget=budget.seconds,
    )
    return state, trace