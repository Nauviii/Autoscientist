"""The projection must stay bounded, stay ordered for caching, and stay deterministic."""

from __future__ import annotations

import pytest

from dsar.core.contracts import ColumnSpec, Finding, HypothesisStatus
from dsar.core.projection import (
    Context,
    build_context,
    estimate_tokens,
    render_contract,
    render_findings,
    render_recent,
    render_schema,
    render_slots,
    trim,
)
from dsar.core.state import SessionBudget, apply_slot_statuses, build_state
from tests.unit.test_policy import column, make_contract
from tests.unit.test_state import evidence, experiment, slots

BUDGET = SessionBudget(experiments=50, tokens=1_000_000, seconds=5400.0)
SYSTEM = "You compose feature cells. Statistics must be learned in fit()."


def make_state(n: int = 12, contract=None):
    contract = contract or make_contract()
    experiments = [experiment("E000", 0.650)]
    experiments += [
        experiment(f"E{i:03d}", 0.650 + 0.002 * i, slot_id="missing_handling")
        for i in range(1, n)
    ]
    evidences = [evidence("E002", HypothesisStatus.SUPPORTED, delta=0.030)]
    evidences += [evidence(f"E{i:03d}", HypothesisStatus.REJECTED) for i in range(3, 6)]
    state, _ = build_state(
        session_id="S1",
        contract=contract,
        experiments=experiments,
        evidences=evidences,
        slots=slots(),
        budget=BUDGET,
        reference_id="E000",
    )
    return state, contract, experiments, evidences


def test_cacheable_blocks_come_first() -> None:
    """A cache prefix ends at the first byte that changes, so ordering is the whole game."""
    state, contract, _, _ = make_state()
    context = build_context(system_prompt=SYSTEM, contract=contract, state=state)
    names = [b.name for b in context.blocks]
    flags = [b.cacheable for b in context.blocks]

    assert names[:3] == ["system", "contract", "schema"]
    assert flags == sorted(flags, reverse=True)
    assert context.cache_breakpoint == 3


def test_a_useful_share_of_the_prompt_is_cacheable() -> None:
    state, contract, _, _ = make_state()
    context = build_context(system_prompt=SYSTEM * 40, contract=contract, state=state)
    assert context.cacheable_tokens / context.total_tokens > 0.5


def test_the_projection_does_not_grow_with_the_session() -> None:
    state_short, contract, _, _ = make_state(10)
    state_long, _, _, _ = make_state(400, contract)
    short = build_context(system_prompt=SYSTEM, contract=contract, state=state_short)
    long = build_context(system_prompt=SYSTEM, contract=contract, state=state_long)
    assert long.total_tokens < short.total_tokens * 1.5


def test_rendering_is_deterministic() -> None:
    state, contract, _, _ = make_state()
    first = build_context(system_prompt=SYSTEM, contract=contract, state=state).render()
    second = build_context(system_prompt=SYSTEM, contract=contract, state=state).render()
    assert first == second


def test_contract_block_carries_the_thresholds_the_model_must_respect() -> None:
    _, contract, _, _ = make_state()
    text = render_contract(contract)
    assert "delta_min" in text
    assert f"{contract.delta_min:.4f}" in text
    assert contract.validation.kind in text


def test_contract_block_warns_when_the_dataset_is_underpowered() -> None:
    contract = make_contract(prevalence=0.01)
    assert "warning" in render_contract(contract)


def test_protected_columns_are_surfaced_with_their_policy() -> None:
    contract = make_contract()
    contract = type(contract)(
        **{
            **{f.name: getattr(contract, f.name) for f in contract.__dataclass_fields__.values()},
            "protected_columns": ("cat_0",),
        }
    )
    text = render_contract(contract)
    assert "protected" in text and "audit" in text


def test_wide_schemas_collapse_by_role() -> None:
    """Listing 300 columns would swamp the prompt and add nothing the model can use."""
    wide = {f"num_{i}": column(f"num_{i}", "numeric") for i in range(300)}
    contract = make_contract(schema=wide)
    text = render_schema(contract)
    assert "grouped by role" in text
    assert "num_250" not in text
    assert estimate_tokens(text) < 200


def test_narrow_schemas_list_every_column() -> None:
    contract = make_contract()
    text = render_schema(contract)
    assert "num_0" in text and "cat_0" in text


def test_checklist_reports_coverage_and_resolutions() -> None:
    state, _, _, _ = make_state()
    text = render_slots(state)
    assert "closed" in text
    assert "missing_handling" in text
    assert "categorical_encoding" in text


def test_progress_block_shows_reference_and_incumbent() -> None:
    state, contract, _, _ = make_state()
    context = build_context(system_prompt=SYSTEM, contract=contract, state=state)
    text = next(b.text for b in context.blocks if b.name == "progress")
    assert state.reference_experiment_id in text
    assert state.incumbent_experiment_id in text
    assert "promotions" in text


def test_recent_block_is_capped() -> None:
    state, contract, _, _ = make_state(60)
    text = render_recent(state, contract.primary_metric)
    assert len(text.splitlines()) <= 2 + 5


def test_recent_block_survives_a_missing_metric() -> None:
    """An invalid experiment has no scores, and must not break the render."""
    state, contract, _, _ = make_state()
    assert render_recent(state, "metric_that_does_not_exist")


def test_findings_render_effect_with_its_interval() -> None:
    findings = [
        Finding(
            finding_id="F1",
            slot_id="missing_handling",
            statement="s",
            status=HypothesisStatus.SUPPORTED,
            effect=0.031,
            ci_low=0.020,
            ci_high=0.042,
            underpowered=False,
            evidence_ids=("EV1",),
        )
    ]
    text = render_findings(findings)
    assert "+0.0310" in text and "+0.0200" in text


def test_empty_findings_render_cleanly() -> None:
    assert "none yet" in render_findings([])


def test_task_block_carries_the_narrow_question() -> None:
    state, contract, _, _ = make_state()
    context = build_context(
        system_prompt=SYSTEM,
        contract=contract,
        state=state,
        action="feature",
        slot_id="categorical_encoding",
        rationale="slot still open",
        extra={"avoid": "target_encoding"},
    )
    text = context.blocks[-1].text
    assert "categorical_encoding" in text
    assert "slot still open" in text
    assert "target_encoding" in text


def test_trimming_drops_volatile_blocks_first() -> None:
    state, contract, _, _ = make_state()
    context = build_context(system_prompt=SYSTEM, contract=contract, state=state)
    trimmed = trim(context, token_budget=context.total_tokens // 3)
    names = [b.name for b in trimmed.blocks]
    assert "system" in names and "contract" in names and "task" in names
    assert "findings" not in names


def test_the_token_budget_is_enforced_at_build_time() -> None:
    state, contract, _, _ = make_state()
    full = build_context(system_prompt=SYSTEM, contract=contract, state=state)
    capped = build_context(
        system_prompt=SYSTEM, contract=contract, state=state, token_budget=full.total_tokens // 2
    )
    assert capped.total_tokens < full.total_tokens


@pytest.mark.parametrize("text,expected", [("", 1), ("a" * 400, 100)])
def test_token_estimate_is_a_simple_character_ratio(text: str, expected: int) -> None:
    assert estimate_tokens(text) == expected


def test_empty_context_reports_a_breakpoint_at_zero() -> None:
    assert Context(()).cache_breakpoint == 0
