"""Routing must be explainable by a rule, and the checklist must fit the dataset."""

from __future__ import annotations

import pytest

from dsar.core.contracts import (
    ColumnSpec,
    ControlRole,
    DataContract,
    DecisionSlot,
    Evidence,
    HypothesisStatus,
    ModelConfig,
    ResearchState,
    SlotStatus,
    TargetDistribution,
    ValidationSpec,
)
from dsar.core.policy import (
    CONVERGENCE_PATIENCE,
    ActionKind,
    Proposal,
    StopReason,
    confirmation_queue,
    default_slots,
    invalid_rate,
    route,
    select_proposal,
    slot_status_from_evidence,
    static_cost,
)
from dsar.core.statistics import build_power_profile

PROTOCOL = ValidationSpec(kind="stratified_kfold", k=10, repeats=5, seed=42)

CELL_A = "class FeatureStep:\n    def fit(self, X, y):\n        self.m_ = X['num_0'].median()\n        return self\n\n    def transform(self, X):\n        out = X.copy()\n        out['num_0'] = out['num_0'].fillna(self.m_)\n        return out\n"
CELL_B = "class FeatureStep:\n    def fit(self, X, y):\n        self.q_ = X['num_1'].quantile(0.5)\n        return self\n\n    def transform(self, X):\n        out = X.copy()\n        out['flag'] = (out['num_1'] > self.q_).astype(int)\n        return out\n"
CELL_NESTED = "class FeatureStep:\n    def fit(self, X, y):\n        self.cols_ = list(X.columns)\n        return self\n\n    def transform(self, X):\n        out = X.copy()\n        for a in self.cols_:\n            if out[a].dtype == 'float64':\n                for b in self.cols_:\n                    out[f'{a}_{b}'] = 1\n        return out\n"


def column(name: str, role: str, missing: float = 0.0, cardinality: int = 4) -> ColumnSpec:
    return ColumnSpec(
        name=name,
        dtype="object" if role == "categorical" else "float64",
        role=role,  # type: ignore[arg-type]
        missing_rate=missing,
        cardinality=cardinality,
        unique_ratio=0.5,
    )


def make_contract(
    task: str = "binary",
    prevalence: float = 0.265,
    schema: dict[str, ColumnSpec] | None = None,
) -> DataContract:
    n_rows = 7043
    distribution = TargetDistribution(
        task=task,  # type: ignore[arg-type]
        n_rows=n_rows,
        n_positive=int(n_rows * prevalence) if task == "binary" else None,
    )
    family = "pr_auc" if task == "binary" else "r2"
    power = build_power_profile(
        n_rows=n_rows,
        n_pos=distribution.n_positive,
        validation=PROTOCOL,
        metric_family=family,  # type: ignore[arg-type]
        expected_score=0.84 if task == "binary" else 0.55,
        expected_auc=0.84 if task == "binary" else None,
        delta_practical=0.01,
    )
    default_schema = {
        "num_0": column("num_0", "numeric", missing=0.05),
        "cat_0": column("cat_0", "categorical", cardinality=4),
    }
    return DataContract(
        dataset_hash="test",
        task=task,  # type: ignore[arg-type]
        target="y",
        target_distribution=distribution,
        schema=schema if schema is not None else default_schema,
        validation=PROTOCOL,
        primary_metric=family,
        metric_family=family,
        secondary_metrics=(),
        secondary_gates={},
        delta_practical=0.01,
        power=power,
        leakage_flags=(),
        excluded_columns=(),
    )


def make_state(
    slots: tuple[DecisionSlot, ...],
    used: int = 5,
    budget: int = 50,
    incumbent: str = "E005",
) -> ResearchState:
    return ResearchState(
        session_id="S1",
        contract_fingerprint="fp",
        reference_experiment_id="E000",
        reference_score=0.652,
        incumbent_experiment_id=incumbent,
        incumbent_score=0.681,
        incumbent_promotions=1,
        slots=slots,
        active_hypotheses=(),
        recent_experiments=(),
        experiments_used=used,
        experiments_budget=budget,
        tokens_used=100,
        tokens_budget=1_000_000,
        seconds_used=100.0,
        seconds_budget=5400.0,
    )


def evidence(
    experiment_id: str,
    status: HypothesisStatus,
    delta: float = 0.0,
    ci_high: float = 0.0,
    control_fingerprint: str = "fp_old",
    role: ControlRole = ControlRole.INCUMBENT,
    underpowered: bool = False,
) -> Evidence:
    return Evidence(
        evidence_id=f"EV_{experiment_id}",
        experiment_id=experiment_id,
        hypothesis_id=None,
        control_experiment_id="E005",
        control_fingerprint=control_fingerprint,
        control_role=role,
        metric="pr_auc",
        mean_delta=delta,
        ci_low=delta - 0.01,
        ci_high=ci_high,
        delta_min=0.015,
        status=status,
        underpowered=underpowered,
    )


def test_checklist_drops_decisions_the_dataset_cannot_face() -> None:
    """A balanced binary task has no imbalance decision to settle."""
    ids = {s.slot_id for s in default_slots(make_contract(prevalence=0.50))}
    assert "imbalance_handling" not in ids
    assert "target_transform" not in ids


def test_imbalance_slot_appears_only_when_positives_are_rare() -> None:
    ids = {s.slot_id for s in default_slots(make_contract(prevalence=0.08))}
    assert "imbalance_handling" in ids


def test_regression_gains_a_target_transform_slot() -> None:
    slots = default_slots(make_contract(task="regression"))
    ids = {s.slot_id for s in slots}
    assert "target_transform" in ids
    assert "imbalance_handling" not in ids


def test_slots_are_skipped_when_the_data_has_no_such_columns() -> None:
    numeric_only = {"num_0": column("num_0", "numeric", missing=0.0)}
    ids = {s.slot_id for s in default_slots(make_contract(schema=numeric_only))}
    assert "categorical_encoding" not in ids
    assert "missing_handling" not in ids


def test_target_encoding_is_offered_only_for_high_cardinality() -> None:
    low = {"cat_0": column("cat_0", "categorical", cardinality=4)}
    high = {"cat_0": column("cat_0", "categorical", cardinality=400)}
    low_options = next(
        s.options for s in default_slots(make_contract(schema=low)) if s.slot_id == "categorical_encoding"
    )
    high_options = next(
        s.options for s in default_slots(make_contract(schema=high)) if s.slot_id == "categorical_encoding"
    )
    assert "target_encoding" not in low_options
    assert "target_encoding" in high_options


def test_a_supported_result_resolves_the_slot() -> None:
    slot = DecisionSlot("missing_handling", "label", ("a", "b", "c"))
    status, resolution = slot_status_from_evidence(
        slot, [evidence("E1", HypothesisStatus.SUPPORTED, delta=0.03)]
    )
    assert status is SlotStatus.RESOLVED
    assert "improved" in (resolution or "")


def test_a_tested_null_resolves_the_slot_too() -> None:
    """Knowing a decision does not matter is a result, not a gap."""
    slot = DecisionSlot("missing_handling", "label", ("a", "b", "c"))
    status, resolution = slot_status_from_evidence(
        slot, [evidence(f"E{i}", HypothesisStatus.REJECTED) for i in range(3)]
    )
    assert status is SlotStatus.RESOLVED
    assert "no option" in (resolution or "")


def test_all_underpowered_marks_the_slot_unresolvable() -> None:
    """A blind spot must stay distinguishable from a measured null."""
    slot = DecisionSlot("missing_handling", "label", ("a", "b", "c"))
    status, resolution = slot_status_from_evidence(
        slot,
        [evidence(f"E{i}", HypothesisStatus.INCONCLUSIVE, underpowered=True) for i in range(3)],
    )
    assert status is SlotStatus.UNRESOLVABLE
    assert "underpowered" in (resolution or "")


def test_a_partially_tested_slot_stays_in_progress() -> None:
    slot = DecisionSlot("categorical_encoding", "label", ("a", "b", "c", "d"))
    status, _ = slot_status_from_evidence(slot, [evidence("E1", HypothesisStatus.REJECTED)])
    assert status is SlotStatus.IN_PROGRESS


def test_open_slots_outrank_further_exploitation() -> None:
    contract = make_contract()
    slots = default_slots(contract)
    plan = route(make_state(slots), contract)
    assert plan.kind in (ActionKind.FEATURE, ActionKind.MODEL)
    assert plan.slot_id == slots[0].slot_id


def test_probes_are_allowed_until_the_slot_budget_runs_out() -> None:
    contract = make_contract()
    slots = default_slots(contract)
    first = slots[0].slot_id
    assert route(make_state(slots), contract, probes_used={first: 0}).allow_probe
    assert not route(make_state(slots), contract, probes_used={first: 3}).allow_probe


def test_exhausted_budget_stops_before_anything_else() -> None:
    contract = make_contract()
    plan = route(make_state(default_slots(contract), used=50, budget=50), contract)
    assert plan.kind is ActionKind.STOP
    assert plan.stop_reason is StopReason.BUDGET


def test_simplification_follows_full_coverage() -> None:
    contract = make_contract()
    closed = tuple(
        DecisionSlot(s.slot_id, s.label, s.options, status=SlotStatus.RESOLVED)
        for s in default_slots(contract)
    )
    plan = route(make_state(closed), contract)
    assert plan.kind is ActionKind.SIMPLIFY
    assert plan.target_experiment_id == "E005"


def test_convergence_stops_the_run_once_nothing_moves() -> None:
    contract = make_contract()
    closed = tuple(
        DecisionSlot(s.slot_id, s.label, s.options, status=SlotStatus.RESOLVED)
        for s in default_slots(contract)
    )
    state = make_state(closed, used=20)
    plan = route(
        state,
        contract,
        promotions_at=[20 - CONVERGENCE_PATIENCE],
        simplification_attempted=True,
    )
    assert plan.kind is ActionKind.STOP
    assert plan.stop_reason is StopReason.CONVERGED


def test_stale_near_misses_are_queued_for_confirmation() -> None:
    """Interactions are real, so a rejection against an old incumbent may reverse."""
    evidences = [
        evidence("E1", HypothesisStatus.INCONCLUSIVE, ci_high=0.012, control_fingerprint="old"),
        evidence("E2", HypothesisStatus.REJECTED, ci_high=0.004, control_fingerprint="old"),
        evidence("E3", HypothesisStatus.REJECTED, ci_high=-0.02, control_fingerprint="old"),
        evidence("E4", HypothesisStatus.SUPPORTED, ci_high=0.05, control_fingerprint="old"),
    ]
    queued = confirmation_queue(evidences, incumbent_fingerprint="E005", limit=5)
    assert queued == ("E1", "E2")


def test_confirmation_ignores_verdicts_against_the_current_incumbent() -> None:
    evidences = [
        evidence("E1", HypothesisStatus.REJECTED, ci_high=0.01, control_fingerprint="E005")
    ]
    assert confirmation_queue(evidences, incumbent_fingerprint="E005", limit=5) == ()


def test_confirmation_runs_once_coverage_is_complete() -> None:
    contract = make_contract()
    closed = tuple(
        DecisionSlot(s.slot_id, s.label, s.options, status=SlotStatus.RESOLVED)
        for s in default_slots(contract)
    )
    evidences = [
        evidence("E7", HypothesisStatus.INCONCLUSIVE, ci_high=0.012, control_fingerprint="old")
    ]
    plan = route(make_state(closed), contract, evidences=evidences)
    assert plan.kind is ActionKind.CONFIRM
    assert plan.target_experiment_id == "E7"


def test_static_cost_reflects_branching() -> None:
    assert static_cost([CELL_NESTED])[2] > static_cost([CELL_A])[2]
    assert static_cost([CELL_A, CELL_B])[1] == 2


def test_selection_prefers_the_simplest_proposal() -> None:
    proposals = [
        Proposal("P1", "s", (CELL_A, CELL_NESTED), ModelConfig("lightgbm", {}), "two cells"),
        Proposal("P2", "s", (CELL_B,), ModelConfig("lightgbm", {}), "one cell"),
    ]
    selection = select_proposal(
        proposals,
        seen_fingerprints=[],
        incumbent_cells=[CELL_A],
        fingerprint_of={"P1": "f1", "P2": "f2"},
        max_cells=5,
    )
    assert selection.chosen is not None and selection.chosen.proposal_id == "P2"
    assert selection.rejected == (("P1", "ranked below the chosen proposal"),)


def test_selection_drops_duplicates() -> None:
    proposals = [Proposal("P1", "s", (CELL_A,), ModelConfig("lightgbm", {}), "seen before")]
    selection = select_proposal(
        proposals,
        seen_fingerprints=["f1"],
        incumbent_cells=[],
        fingerprint_of={"P1": "f1"},
        max_cells=5,
    )
    assert selection.chosen is None
    assert selection.rejected[0][1].startswith("duplicate")


def test_selection_enforces_the_cell_budget() -> None:
    proposals = [
        Proposal("P1", "s", (CELL_A, CELL_B, CELL_NESTED), ModelConfig("lightgbm", {}), "too long")
    ]
    selection = select_proposal(
        proposals,
        seen_fingerprints=[],
        incumbent_cells=[],
        fingerprint_of={"P1": "f1"},
        max_cells=2,
    )
    assert selection.chosen is None
    assert "exceeds limit" in selection.rejected[0][1]


def test_selection_breaks_ties_by_prefix_reuse() -> None:
    """A shared head can be reused from cache, so it wins an otherwise even match."""
    proposals = [
        Proposal("P_far", "s", (CELL_B,), ModelConfig("lightgbm", {}), "no shared head"),
        Proposal("P_near", "s", (CELL_A,), ModelConfig("lightgbm", {}), "shares the head"),
    ]
    selection = select_proposal(
        proposals,
        seen_fingerprints=[],
        incumbent_cells=[CELL_A, CELL_B],
        fingerprint_of={"P_far": "f1", "P_near": "f2"},
        max_cells=5,
    )
    assert selection.chosen is not None and selection.chosen.proposal_id == "P_near"


def test_selection_is_deterministic_under_repetition() -> None:
    proposals = [
        Proposal("P1", "s", (CELL_A,), ModelConfig("lightgbm", {}), ""),
        Proposal("P2", "s", (CELL_B,), ModelConfig("lightgbm", {}), ""),
    ]
    args = dict(
        seen_fingerprints=[],
        incumbent_cells=[],
        fingerprint_of={"P1": "f1", "P2": "f2"},
        max_cells=5,
    )
    first = select_proposal(proposals, **args)
    second = select_proposal(list(reversed(proposals)), **args)
    assert first.chosen.proposal_id == second.chosen.proposal_id  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "statuses,expected",
    [
        ([], 0.0),
        (["completed"] * 5, 0.0),
        (["invalid_code", "invalid_guard", "completed", "completed", "completed"], 0.4),
    ],
)
def test_invalid_rate_tracks_recent_failures(statuses: list[str], expected: float) -> None:
    assert invalid_rate(statuses) == pytest.approx(expected)
