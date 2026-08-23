"""The state must stay bounded, stay deterministic, and never drift."""

from __future__ import annotations

import pytest

from dsar.core.contracts import (
    ComplexityVector,
    ControlRole,
    DecisionSlot,
    Evidence,
    ExperimentResult,
    ExperimentStatus,
    HypothesisStatus,
    ModelConfig,
    SlotStatus,
)
from dsar.core.state import (
    RECENT_EXPERIMENTS,
    SessionBudget,
    apply_slot_statuses,
    build_state,
    derive_findings,
    resolve_incumbent,
    split_findings,
    usage,
)
from tests.unit.test_policy import make_contract

BUDGET = SessionBudget(experiments=50, tokens=1_000_000, seconds=5400.0)
COMPLEXITY = ComplexityVector(
    n_features=40, n_cells=2, pipeline_depth=1, fit_seconds=1.0, ast_nodes=200
)


def experiment(
    experiment_id: str,
    score: float,
    slot_id: str | None = "missing_handling",
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    tokens: int = 1000,
    seconds: float = 30.0,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        fingerprint=f"fp_{experiment_id}",
        hypothesis_id=None,
        parent_id=None,
        slot_id=slot_id,
        model=ModelConfig(family="lightgbm", params={}),
        fold_scores={"pr_auc": tuple([score] * 50)},
        complexity=COMPLEXITY,
        runtime_s=seconds,
        token_cost=tokens,
        status=status,
    )


def evidence(
    experiment_id: str,
    status: HypothesisStatus,
    delta: float = 0.0,
    role: ControlRole = ControlRole.INCUMBENT,
    underpowered: bool = False,
) -> Evidence:
    return Evidence(
        evidence_id=f"EV_{experiment_id}",
        experiment_id=experiment_id,
        hypothesis_id=None,
        control_experiment_id="E000",
        control_fingerprint="fp_E000",
        control_role=role,
        metric="pr_auc",
        mean_delta=delta,
        ci_low=delta - 0.005,
        ci_high=delta + 0.005,
        delta_min=0.015,
        status=status,
        underpowered=underpowered,
    )


def slots() -> tuple[DecisionSlot, ...]:
    return (
        DecisionSlot("missing_handling", "Missing value handling", ("a", "b", "c")),
        DecisionSlot("categorical_encoding", "Categorical encoding", ("a", "b", "c")),
    )


def test_usage_counts_duplicates_against_the_experiment_budget() -> None:
    """A duplicate skips the fits but still costs an LLM call."""
    experiments = [
        experiment("E000", 0.65),
        experiment("E001", 0.66, status=ExperimentStatus.DUPLICATE, seconds=0.0),
    ]
    used, tokens, seconds = usage(experiments)
    assert used == 2
    assert tokens == 2000
    assert seconds == 30.0


def test_only_supported_evidence_advances_the_incumbent() -> None:
    experiments = [experiment("E000", 0.650), experiment("E001", 0.658)]
    evidences = [evidence("E001", HypothesisStatus.INCONCLUSIVE, delta=0.008)]
    trace = resolve_incumbent(experiments, evidences, "E000", "pr_auc")
    assert trace.experiment_id == "E000"
    assert trace.promotions == 0


def test_the_ladder_climbs_only_on_measured_gains() -> None:
    experiments = [experiment(f"E00{i}", 0.650 + 0.01 * i) for i in range(4)]
    evidences = [
        evidence("E001", HypothesisStatus.INCONCLUSIVE, delta=0.008),
        evidence("E002", HypothesisStatus.SUPPORTED, delta=0.030),
        evidence("E003", HypothesisStatus.REJECTED, delta=0.002),
    ]
    trace = resolve_incumbent(experiments, evidences, "E000", "pr_auc")
    assert trace.experiment_id == "E002"
    assert trace.promotions_at == (3,)


def test_promotion_ignores_evidence_measured_against_the_reference() -> None:
    """Everything beats the naive reference; only the incumbent comparison promotes."""
    experiments = [experiment("E000", 0.650), experiment("E001", 0.680)]
    evidences = [
        evidence("E001", HypothesisStatus.SUPPORTED, delta=0.03, role=ControlRole.REFERENCE)
    ]
    assert resolve_incumbent(experiments, evidences, "E000", "pr_auc").experiment_id == "E000"


def test_an_invalid_experiment_cannot_become_incumbent() -> None:
    experiments = [
        experiment("E000", 0.650),
        experiment("E001", 0.900, status=ExperimentStatus.INVALID_GUARD),
    ]
    evidences = [evidence("E001", HypothesisStatus.SUPPORTED, delta=0.25)]
    assert resolve_incumbent(experiments, evidences, "E000", "pr_auc").experiment_id == "E000"


def test_a_missing_reference_is_an_error() -> None:
    with pytest.raises(ValueError):
        resolve_incumbent([experiment("E001", 0.65)], [], "E000", "pr_auc")


def test_slot_statuses_are_recomputed_from_evidence() -> None:
    experiments = [experiment(f"E00{i}", 0.65, slot_id="missing_handling") for i in range(3)]
    evidences = [evidence(f"E00{i}", HypothesisStatus.REJECTED) for i in range(3)]
    updated = apply_slot_statuses(slots(), experiments, evidences)
    resolved = next(s for s in updated if s.slot_id == "missing_handling")
    untouched = next(s for s in updated if s.slot_id == "categorical_encoding")
    assert resolved.status is SlotStatus.RESOLVED
    assert len(resolved.experiment_ids) == 3
    assert untouched.status is SlotStatus.OPEN


def test_findings_separate_a_measured_null_from_a_blind_spot() -> None:
    """These are opposite conclusions and must never be merged in the report."""
    experiments = [
        experiment("E001", 0.68, slot_id="missing_handling"),
        experiment("E002", 0.65, slot_id="missing_handling"),
        experiment("E003", 0.65, slot_id="missing_handling"),
        experiment("E004", 0.65, slot_id="categorical_encoding"),
        experiment("E005", 0.65, slot_id="categorical_encoding"),
        experiment("E006", 0.65, slot_id="categorical_encoding"),
    ]
    evidences = [
        evidence("E001", HypothesisStatus.SUPPORTED, delta=0.030),
        evidence("E002", HypothesisStatus.REJECTED),
        evidence("E003", HypothesisStatus.REJECTED),
        evidence("E004", HypothesisStatus.INCONCLUSIVE, underpowered=True),
        evidence("E005", HypothesisStatus.INCONCLUSIVE, underpowered=True),
        evidence("E006", HypothesisStatus.INCONCLUSIVE, underpowered=True),
    ]
    updated = apply_slot_statuses(slots(), experiments, evidences)
    mattered, did_not, unknown = split_findings(derive_findings(updated, experiments, evidences))

    assert [f.slot_id for f in mattered] == ["missing_handling"]
    assert did_not == ()
    assert len(unknown) == 1
    assert "undecided" in unknown[0]


def test_a_tested_null_lands_in_the_did_not_matter_group() -> None:
    experiments = [experiment(f"E00{i}", 0.65, slot_id="missing_handling") for i in range(3)]
    evidences = [evidence(f"E00{i}", HypothesisStatus.REJECTED) for i in range(3)]
    updated = apply_slot_statuses(slots(), experiments, evidences)
    mattered, did_not, unknown = split_findings(derive_findings(updated, experiments, evidences))
    assert mattered == () and unknown == ()
    assert [f.slot_id for f in did_not] == ["missing_handling"]


def test_open_slots_produce_no_finding() -> None:
    updated = apply_slot_statuses(slots(), [], [])
    assert derive_findings(updated, [], []) == ()


def make_session(n: int) -> tuple[list[ExperimentResult], list[Evidence]]:
    experiments = [experiment("E000", 0.650)]
    experiments += [experiment(f"E{i:03d}", 0.650 + 0.001 * i) for i in range(1, n)]
    evidences = [evidence("E002", HypothesisStatus.SUPPORTED, delta=0.03)]
    return experiments, evidences


def test_state_stays_bounded_as_the_session_grows() -> None:
    """This cap is what keeps context linear instead of quadratic."""
    contract = make_contract()
    for n in (10, 60, 300):
        experiments, evidences = make_session(n)
        state, _ = build_state(
            session_id="S1",
            contract=contract,
            experiments=experiments,
            evidences=evidences,
            slots=slots(),
            budget=BUDGET,
            reference_id="E000",
        )
        assert len(state.recent_experiments) == RECENT_EXPERIMENTS
        assert state.experiments_used == n


def test_rebuilding_from_the_same_store_gives_the_same_state() -> None:
    contract = make_contract()
    experiments, evidences = make_session(20)
    args = dict(
        session_id="S1",
        contract=contract,
        experiments=experiments,
        evidences=evidences,
        slots=slots(),
        budget=BUDGET,
        reference_id="E000",
    )
    first, trace_a = build_state(**args)
    second, trace_b = build_state(**args)
    assert first == second
    assert trace_a == trace_b


def test_state_exposes_reference_and_incumbent_separately() -> None:
    contract = make_contract()
    experiments, evidences = make_session(10)
    state, trace = build_state(
        session_id="S1",
        contract=contract,
        experiments=experiments,
        evidences=evidences,
        slots=slots(),
        budget=BUDGET,
        reference_id="E000",
    )
    assert state.reference_experiment_id == "E000"
    assert state.incumbent_experiment_id == "E002"
    assert state.incumbent_score > state.reference_score
    assert state.incumbent_promotions == trace.promotions == 1


def test_budget_exhaustion_trips_on_any_ceiling() -> None:
    contract = make_contract()
    experiments, evidences = make_session(6)
    tight = SessionBudget(experiments=50, tokens=1000, seconds=5400.0)
    state, _ = build_state(
        session_id="S1",
        contract=contract,
        experiments=experiments,
        evidences=evidences,
        slots=slots(),
        budget=tight,
        reference_id="E000",
    )
    assert state.experiments_used < state.experiments_budget
    assert state.budget_exhausted


def test_coverage_reflects_closed_slots() -> None:
    contract = make_contract()
    experiments = [experiment("E000", 0.65)]
    experiments += [experiment(f"E00{i}", 0.65, slot_id="missing_handling") for i in range(1, 4)]
    evidences = [evidence(f"E00{i}", HypothesisStatus.REJECTED) for i in range(1, 4)]
    state, _ = build_state(
        session_id="S1",
        contract=contract,
        experiments=experiments,
        evidences=evidences,
        slots=slots(),
        budget=BUDGET,
        reference_id="E000",
    )
    assert state.coverage == 0.5
    assert [s.slot_id for s in state.open_slots] == ["categorical_encoding"]
