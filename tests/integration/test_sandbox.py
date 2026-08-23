"""The sandbox must contain failures, enforce limits, and catch what static cannot."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from dsar.adapters.sandbox import ForkExecutor, GauntletOutcome, run_gauntlet
from dsar.core.guards import GuardId
from dsar.ports import ExecutorPort, FoldSpec, ResourceLimits
from tests.conftest import LEAKY_CELLS_DIR, make_frame

pytestmark = pytest.mark.skipif(os.name != "posix", reason="sandbox requires fork and setrlimit")

FOLD = FoldSpec(fold_index=0, repeat=0, train=tuple(range(40)), validate=tuple(range(40, 60)))
LIMITS = ResourceLimits(memory_mb=1024, cpu_seconds=20, wall_seconds=15, max_features=50)

CLEAN = (LEAKY_CELLS_DIR / "clean_reference.py").read_text()
BATCH_STATISTIC = (LEAKY_CELLS_DIR / "batch_statistic.py").read_text()
BATCH_IMPUTE = (LEAKY_CELLS_DIR / "batch_impute.py").read_text()
FIT_IN_TRANSFORM = (LEAKY_CELLS_DIR / "fit_in_transform.py").read_text()
TARGET_RETENTION = (LEAKY_CELLS_DIR / "target_in_transform.py").read_text()

RAISES = """
class FeatureStep:
    def fit(self, X, y):
        return self

    def transform(self, X):
        raise ValueError("deliberate failure")
"""

SPINS = """
class FeatureStep:
    def fit(self, X, y):
        return self

    def transform(self, X):
        total = 0
        while True:
            total += 1
        return X
"""

EATS_MEMORY = """
class FeatureStep:
    def fit(self, X, y):
        return self

    def transform(self, X):
        blocks = []
        for _ in range(10000):
            blocks.append(bytearray(16 * 1024 * 1024))
        return X
"""

EXPLODES_WIDTH = """
class FeatureStep:
    def fit(self, X, y):
        self.cols_ = [c for c in X.columns if X[c].dtype.kind == "f"]
        return self

    def transform(self, X):
        out = X.copy()
        for i, a in enumerate(self.cols_):
            for j in range(200):
                out[f"{a}_{j}"] = out[a] * j
        return out
"""


@pytest.fixture(scope="module")
def executor() -> ForkExecutor:
    frame = make_frame(n=60)
    target = pd.Series(np.random.default_rng(0).integers(0, 2, size=60), index=frame.index)
    return ForkExecutor(frame, target)


def test_executor_satisfies_the_port(executor: ForkExecutor) -> None:
    assert isinstance(executor, ExecutorPort)


def test_a_clean_cell_runs_and_returns_both_folds(executor: ForkExecutor) -> None:
    artifacts = executor.run_cells([CLEAN], FOLD, LIMITS)
    assert not artifacts.failed
    assert artifacts.train is not None and len(artifacts.train) == 40
    assert artifacts.validate is not None and len(artifacts.validate) == 20


def test_guard_evidence_is_gathered_in_one_execution(executor: ForkExecutor) -> None:
    """Collecting it here avoids re-entering the sandbox once per guard."""
    artifacts = executor.run_cells([CLEAN], FOLD, LIMITS, collect_guard_evidence=True)
    assert artifacts.singles and artifacts.replay is not None
    assert all(len(single) == 1 for single in artifacts.singles.values())


def test_an_exception_is_reported_not_propagated(executor: ForkExecutor) -> None:
    """A broken cell becomes an INVALID experiment; it must not stop the loop."""
    artifacts = executor.run_cells([RAISES], FOLD, LIMITS)
    assert artifacts.failed
    assert "deliberate failure" in (artifacts.error or "")


def test_an_infinite_loop_is_killed_by_the_wall_clock(executor: ForkExecutor) -> None:
    limits = ResourceLimits(memory_mb=512, cpu_seconds=30, wall_seconds=3, max_features=50)
    started = __import__("time").monotonic()
    artifacts = executor.run_cells([SPINS], FOLD, limits)
    elapsed = __import__("time").monotonic() - started

    assert artifacts.failed
    assert elapsed < 10
    assert "timeout" in (artifacts.error or "").lower()


def test_runaway_allocation_hits_the_memory_limit(executor: ForkExecutor) -> None:
    limits = ResourceLimits(memory_mb=256, cpu_seconds=20, wall_seconds=20, max_features=50)
    artifacts = executor.run_cells([EATS_MEMORY], FOLD, limits)
    assert artifacts.failed


def test_the_parent_survives_every_failure_mode(executor: ForkExecutor) -> None:
    """The point of forking: a child dying must leave the parent able to continue."""
    for code in (RAISES, EATS_MEMORY):
        executor.run_cells([code], FOLD, ResourceLimits(memory_mb=256, wall_seconds=10))
    assert not executor.run_cells([CLEAN], FOLD, LIMITS).failed


def test_gauntlet_clears_a_clean_cell(executor: ForkExecutor) -> None:
    outcome = run_gauntlet([CLEAN], executor, FOLD, LIMITS)
    assert outcome.passed
    assert outcome.artifacts is not None
    assert {o.guard for o in outcome.outcomes} >= {
        GuardId.SCHEMA,
        GuardId.DETERMINISM,
        GuardId.ROW_INDEPENDENCE,
    }


@pytest.mark.parametrize(
    "code,guard",
    [
        (FIT_IN_TRANSFORM, GuardId.FIT_IN_TRANSFORM),
        (TARGET_RETENTION, GuardId.TARGET_RETENTION),
    ],
)
def test_static_guards_stop_a_cell_before_it_runs(
    executor: ForkExecutor, code: str, guard: GuardId
) -> None:
    outcome = run_gauntlet([code], executor, FOLD, LIMITS)
    assert not outcome.passed
    assert outcome.failure is not None and outcome.failure.guard is guard
    assert outcome.artifacts is None


@pytest.mark.parametrize("code", [BATCH_STATISTIC, BATCH_IMPUTE])
def test_row_independence_catches_what_the_ast_cannot(
    executor: ForkExecutor, code: str
) -> None:
    """These read as ordinary pandas; only execution exposes the batch dependence."""
    outcome = run_gauntlet([code], executor, FOLD, LIMITS)
    assert not outcome.passed
    assert outcome.failure is not None
    assert outcome.failure.guard is GuardId.ROW_INDEPENDENCE
    assert "fit" in outcome.failure.repair_hint


def test_gauntlet_enforces_the_feature_budget(executor: ForkExecutor) -> None:
    outcome = run_gauntlet([EXPLODES_WIDTH], executor, FOLD, LIMITS)
    assert not outcome.passed
    assert outcome.failure is not None and outcome.failure.guard is GuardId.SCHEMA


def test_gauntlet_reports_a_runtime_error_as_a_smoke_failure(executor: ForkExecutor) -> None:
    outcome = run_gauntlet([RAISES], executor, FOLD, LIMITS)
    assert not outcome.passed
    assert outcome.failure is not None and outcome.failure.guard is GuardId.SMOKE


def test_a_pipeline_of_several_cells_composes_in_order(executor: ForkExecutor) -> None:
    second = """
class FeatureStep:
    def fit(self, X, y):
        self.mean_ = X["num_0"].mean()
        return self

    def transform(self, X):
        out = X.copy()
        out["centred"] = out["num_0"] - self.mean_
        return out
"""
    outcome = run_gauntlet([CLEAN, second], executor, FOLD, LIMITS)
    assert outcome.passed
    assert "centred" in outcome.artifacts.validate.columns  # type: ignore[union-attr]


def test_restricted_builtins_block_file_access(executor: ForkExecutor) -> None:
    """Belt and braces: the AST allowlist already rejects this before execution."""
    code = """
class FeatureStep:
    def fit(self, X, y):
        return self

    def transform(self, X):
        open("/etc/passwd").read()
        return X
"""
    assert executor.run_cells([code], FOLD, LIMITS).failed


def test_outcome_reports_no_failure_when_everything_passed() -> None:
    assert GauntletOutcome((), None).failure is None
