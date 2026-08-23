"""Every fake must satisfy its protocol, and the boundary must stay one-way."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsar.core.contracts import (
    ComplexityVector,
    ControlRole,
    ExperimentResult,
    ExperimentStatus,
    HypothesisStatus,
    Evidence,
    ModelConfig,
)
from dsar.ports import (
    ApprovalPort,
    ApprovalRequest,
    ClockPort,
    ExecutorPort,
    FoldSpec,
    LLMPort,
    LLMRequest,
    LLMResponse,
    ResourceLimits,
    StorePort,
)
from tests.conftest import CORE_DIR, LEAKY_CELLS_DIR, make_frame
from tests.fakes import (
    AutoApprover,
    FrozenClock,
    InProcessExecutor,
    MemoryStore,
    ScriptedLLM,
)
from tests.unit.test_projection import SYSTEM, make_state
from dsar.core.projection import build_context

LIMITS = ResourceLimits()
FOLD = FoldSpec(fold_index=0, repeat=0, train=tuple(range(40)), validate=tuple(range(40, 60)))


def experiment(experiment_id: str, fingerprint: str) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        fingerprint=fingerprint,
        hypothesis_id=None,
        parent_id=None,
        slot_id=None,
        model=ModelConfig(family="lightgbm", params={}),
        fold_scores={"pr_auc": (0.65,) * 50},
        complexity=ComplexityVector(40, 2, 1, 1.0, 200),
        runtime_s=10.0,
        token_cost=500,
        status=ExperimentStatus.COMPLETED,
    )


def test_core_does_not_import_ports() -> None:
    """The arrow runs one way, or the pure layer stops being testable in isolation."""
    for path in CORE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            module = (
                node.module
                if isinstance(node, ast.ImportFrom) and node.module
                else ""
            )
            assert "ports" not in module, f"{path.name} imports {module}"


@pytest.mark.parametrize(
    "fake,protocol",
    [
        (ScriptedLLM(), LLMPort),
        (MemoryStore(), StorePort),
        (FrozenClock(), ClockPort),
        (AutoApprover(), ApprovalPort),
    ],
)
def test_fakes_satisfy_their_protocol(fake: object, protocol: type) -> None:
    assert isinstance(fake, protocol)


def test_executor_fake_satisfies_its_protocol() -> None:
    frame = make_frame()
    executor = InProcessExecutor(frame, pd.Series(np.zeros(len(frame))))
    assert isinstance(executor, ExecutorPort)


def test_llm_response_reports_cache_usage() -> None:
    response = LLMResponse(
        text="ok", model="m", input_tokens=1000, output_tokens=100, cached_input_tokens=600
    )
    assert response.total_tokens == 1100
    assert response.cache_hit_ratio == pytest.approx(0.6)


def test_llm_request_exposes_the_cache_breakpoint() -> None:
    state, contract, _, _ = make_state()
    context = build_context(system_prompt=SYSTEM, contract=contract, state=state)
    request = LLMRequest(
        context=context, model="m", temperature=0.7, max_tokens=2000, purpose="propose"
    )
    assert request.cache_breakpoint == context.cache_breakpoint > 0


def test_scripted_llm_returns_queued_replies_in_order() -> None:
    state, contract, _, _ = make_state()
    context = build_context(system_prompt=SYSTEM, contract=contract, state=state)
    llm = ScriptedLLM(replies=["first", "second"])
    request = LLMRequest(
        context=context, model="m", temperature=0.0, max_tokens=10, purpose="propose"
    )
    assert llm.complete(request).text == "first"
    assert llm.complete(request).text == "second"
    assert len(llm.requests) == 2


def test_executor_runs_a_clean_cell_and_collects_guard_evidence() -> None:
    frame = make_frame()
    executor = InProcessExecutor(frame, pd.Series(np.zeros(len(frame))))
    code = (LEAKY_CELLS_DIR / "clean_reference.py").read_text()

    artifacts = executor.run_cells([code], FOLD, LIMITS, collect_guard_evidence=True)
    assert not artifacts.failed
    assert artifacts.validate is not None and len(artifacts.validate) == 20
    assert artifacts.singles and artifacts.replay is not None


def test_executor_reports_a_failure_instead_of_raising() -> None:
    """A broken cell must become an INVALID experiment, not crash the loop."""
    frame = make_frame()
    executor = InProcessExecutor(frame, pd.Series(np.zeros(len(frame))))
    artifacts = executor.run_cells(["class FeatureStep:\n    pass\n"], FOLD, LIMITS)
    assert artifacts.failed
    assert artifacts.error is not None


def test_store_is_append_only() -> None:
    store = MemoryStore()
    store.record_experiment(experiment("E001", "fp1"))
    with pytest.raises(ValueError):
        store.record_experiment(experiment("E001", "fp2"))


def test_store_addresses_artifacts_by_content() -> None:
    store = MemoryStore()
    first = store.put_artifact("code", b"same bytes")
    second = store.put_artifact("code", b"same bytes")
    assert first.artifact_id == second.artifact_id
    assert store.get_artifact(first.artifact_id) == b"same bytes"


def test_store_raises_on_a_missing_artifact() -> None:
    with pytest.raises(KeyError):
        MemoryStore().get_artifact("nope")


def test_store_supports_duplicate_detection() -> None:
    store = MemoryStore()
    store.record_experiment(experiment("E001", "fp1"))
    assert "fp1" in store.fingerprints()
    assert store.find_by_fingerprint("fp1") is not None
    assert store.find_by_fingerprint("fp2") is None


def test_store_keeps_evidence_alongside_experiments() -> None:
    store = MemoryStore()
    store.record_evidence(
        Evidence(
            evidence_id="EV1",
            experiment_id="E001",
            hypothesis_id=None,
            control_experiment_id="E000",
            control_fingerprint="fp0",
            control_role=ControlRole.INCUMBENT,
            metric="pr_auc",
            mean_delta=0.03,
            ci_low=0.02,
            ci_high=0.04,
            delta_min=0.015,
            status=HypothesisStatus.SUPPORTED,
            underpowered=False,
        )
    )
    assert len(store.evidences()) == 1


def test_frozen_clock_advances_predictably() -> None:
    clock = FrozenClock()
    first, second = clock.now(), clock.now()
    assert second > first
    assert (second - first).total_seconds() == 1.0
    assert FrozenClock().now() == first


def test_approval_diff_shows_only_what_departs_from_the_default() -> None:
    """The gate should take seconds to read, so identical fields are not shown."""
    request = ApprovalRequest(
        gate="data_contract",
        summary="proposed contract",
        proposed={"primary_metric": "pr_auc", "k": "10", "seed": "42"},
        defaults={"primary_metric": "roc_auc", "k": "10", "seed": "42"},
    )
    assert request.diff == (("primary_metric", "roc_auc", "pr_auc"),)


def test_auto_approver_records_what_a_human_would_have_seen() -> None:
    approver = AutoApprover()
    decision = approver.request(
        ApprovalRequest(gate="open_holdout", summary="final", proposed={})
    )
    assert decision.approved and decision.auto_approved
    assert len(approver.requests) == 1
