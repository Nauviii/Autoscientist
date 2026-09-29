"""The store is the session's memory, so what it loses or changes is what replay loses."""

from __future__ import annotations

import pytest

from dsar.adapters.store import StoreError, list_sessions, open_store
from dsar.core.contracts import (
    ComplexityVector,
    ControlRole,
    Evidence,
    ExperimentResult,
    ExperimentStatus,
    HypothesisStatus,
    ModelConfig,
    ProbeResult,
)
from dsar.core.state import SessionBudget, build_state
from dsar.ports import StorePort
from tests.unit.test_policy import make_contract


def experiment(
    experiment_id: str = "E001",
    fingerprint: str = "fp1",
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    score: float = 0.65,
) -> ExperimentResult:
    return ExperimentResult(
        experiment_id=experiment_id,
        fingerprint=fingerprint,
        hypothesis_id="H001",
        parent_id="E000",
        slot_id="missing_handling",
        model=ModelConfig(family="lightgbm", params={"num_leaves": 31, "lr": 0.05}),
        fold_scores={"pr_auc": tuple(score + i * 1e-4 for i in range(50))},
        complexity=ComplexityVector(40, 2, 1, 1.25, 200),
        runtime_s=14.5,
        token_cost=1200,
        status=status,
        artifact_ids={"code": "abc123"},
        guard_failure=None,
    )


def evidence(evidence_id: str = "EV1", delta: float = 0.03) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        experiment_id="E001",
        hypothesis_id="H001",
        control_experiment_id="E000",
        control_fingerprint="fp0",
        control_role=ControlRole.INCUMBENT,
        metric="pr_auc",
        mean_delta=delta,
        ci_low=delta - 0.01,
        ci_high=delta + 0.01,
        delta_min=0.015,
        status=HypothesisStatus.SUPPORTED,
        underpowered=False,
        gate_violations=("brier: degraded +0.02 > 0.005",),
        reason="CI lower bound clears delta_min",
    )


@pytest.fixture
def store(tmp_path):
    with open_store(tmp_path, "S1") as opened:
        yield opened


def test_the_store_satisfies_its_port(store) -> None:
    assert isinstance(store, StorePort)


def test_an_experiment_survives_the_round_trip_exactly(store) -> None:
    """Replay is only meaningful if what comes back is what went in."""
    original = experiment()
    store.record_experiment(original)
    assert store.experiments() == (original,)


def test_every_fold_is_kept_not_the_mean(store) -> None:
    """Paired comparison cannot be recovered from an average."""
    store.record_experiment(experiment())
    restored = store.experiments()[0]
    assert len(restored.fold_scores["pr_auc"]) == 50
    assert isinstance(restored.fold_scores["pr_auc"], tuple)


def test_evidence_survives_the_round_trip_exactly(store) -> None:
    original = evidence()
    store.record_evidence(original)
    assert store.evidences() == (original,)


def test_insertion_order_is_preserved(store) -> None:
    """The promotion ladder is replayed in order, so order is part of the record."""
    for i in range(5):
        store.record_experiment(experiment(f"E{i:03d}", f"fp{i}"))
    assert [e.experiment_id for e in store.experiments()] == [f"E{i:03d}" for i in range(5)]


def test_recording_the_same_experiment_twice_is_an_error(store) -> None:
    """An editable record would make the ladder unreplayable."""
    store.record_experiment(experiment())
    with pytest.raises(StoreError):
        store.record_experiment(experiment(score=0.99))


def test_recording_the_same_evidence_twice_is_an_error(store) -> None:
    store.record_evidence(evidence())
    with pytest.raises(StoreError):
        store.record_evidence(evidence(delta=0.99))


def test_duplicate_detection_reads_from_the_store(store) -> None:
    store.record_experiment(experiment("E001", "fp1"))
    store.record_experiment(experiment("E002", "fp2"))

    assert store.fingerprints() == frozenset({"fp1", "fp2"})
    assert store.find_by_fingerprint("fp1").experiment_id == "E001"
    assert store.find_by_fingerprint("nope") is None


def test_identical_bytes_are_stored_once(store) -> None:
    """Cell source repeats across experiments; paying for it once is the point."""
    first = store.put_artifact("code", b"class FeatureStep: pass")
    second = store.put_artifact("code", b"class FeatureStep: pass")

    assert first.artifact_id == second.artifact_id
    assert store.get_artifact(first.artifact_id) == b"class FeatureStep: pass"
    assert store.summary()["artifacts"] == 1


def test_different_bytes_get_different_ids(store) -> None:
    a = store.put_artifact("code", b"one")
    b = store.put_artifact("code", b"two")
    assert a.artifact_id != b.artifact_id


def test_a_missing_artifact_raises(store) -> None:
    with pytest.raises(KeyError):
        store.get_artifact("does-not-exist")


def test_artifacts_are_sharded_on_disk(store) -> None:
    """Flat directories with tens of thousands of files get slow to walk."""
    ref = store.put_artifact("code", b"payload")
    shards = [p for p in store.objects_dir.iterdir() if p.is_dir()]
    assert shards and shards[0].name == ref.artifact_id[:2]


def test_probes_are_recorded_as_part_of_the_trail(store) -> None:
    """A report can then say which observation a hypothesis followed from."""
    probe = ProbeResult(
        probe_id="P1",
        kind="missing_pattern",
        params={"columns": ["income", "dependents"]},
        payload={"co_missing_rate": 0.021},
    )
    store.record_probe(probe)
    assert store.probes() == (probe,)


def test_session_metadata_is_readable_back(store) -> None:
    store.set_meta("contract_fingerprint", "abc123")
    store.set_meta("environment", "env-hash")

    assert store.get_meta("contract_fingerprint") == "abc123"
    assert store.get_meta("missing") is None
    assert set(store.meta()) == {"contract_fingerprint", "environment"}


def test_a_session_reopens_with_its_contents_intact(tmp_path) -> None:
    """Resuming after a stop is the whole reason this is on disk."""
    with open_store(tmp_path, "S1") as first:
        first.record_experiment(experiment())
        first.record_evidence(evidence())
        first.set_meta("seed", "42")

    with open_store(tmp_path, "S1") as second:
        assert len(second.experiments()) == 1
        assert len(second.evidences()) == 1
        assert second.get_meta("seed") == "42"


def test_sessions_do_not_see_each_other(tmp_path) -> None:
    """A benchmark sweep runs many at once and must not blend them."""
    with open_store(tmp_path, "A") as a:
        a.record_experiment(experiment("E001", "fp-a"))
    with open_store(tmp_path, "B") as b:
        b.record_experiment(experiment("E001", "fp-b"))
        assert b.fingerprints() == frozenset({"fp-b"})

    assert list_sessions(tmp_path) == ("A", "B")


def test_listing_sessions_on_an_empty_root_is_not_an_error(tmp_path) -> None:
    assert list_sessions(tmp_path / "nothing") == ()


def test_the_store_feeds_build_state_directly(tmp_path) -> None:
    """No translation layer: what the store returns is what the fold consumes."""
    contract = make_contract()
    with open_store(tmp_path, "S1") as store:
        store.record_experiment(experiment("E000", "fp0", score=0.62))
        store.record_experiment(experiment("E001", "fp1", score=0.68))
        store.record_evidence(evidence())

        state, trace = build_state(
            session_id="S1",
            contract=contract,
            experiments=store.experiments(),
            evidences=store.evidences(),
            slots=(),
            budget=SessionBudget(experiments=50, tokens=1_000_000, seconds=5400.0),
            reference_id="E000",
        )

    assert state.reference_experiment_id == "E000"
    assert state.incumbent_experiment_id == "E001"
    assert trace.promotions == 1


def test_summary_counts_every_table(store) -> None:
    store.record_experiment(experiment())
    store.record_evidence(evidence())
    store.put_artifact("code", b"x")

    assert store.summary() == {
        "experiments": 1,
        "evidence": 1,
        "probes": 0,
        "artifacts": 1,
    }