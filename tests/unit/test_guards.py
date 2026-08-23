"""Guards must catch every leaky fixture and clear the clean control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.conftest import LEAKY_CELLS_DIR as FIXTURES
from tests.conftest import instantiate_cell, make_frame
from dsar.core.guards import (
    GuardId,
    check_batch_invariance,
    check_determinism,
    check_permutation,
    check_row_independence,
    check_schema,
    first_failure,
    run_static_gauntlet,
)

# Rejected before execution, by reading the AST alone.
STATIC_CATCHES: dict[str, GuardId] = {
    "target_in_transform.py": GuardId.TARGET_RETENTION,
    "fit_in_transform.py": GuardId.FIT_IN_TRANSFORM,
    "nondeterministic.py": GuardId.RANDOMNESS,
}

# Indistinguishable from ordinary pandas in the AST; only execution exposes them.
RUNTIME_CATCHES = {"batch_statistic.py", "batch_impute.py"}

# Sampling individual rows only sees what those rows provoke. A rank shifts every
# row, so it is caught; an imputation only touches the rows that are missing, and
# is missed unless one of them happens to be sampled. Checking a whole slice at
# once removes that dependence on luck.
SAMPLING_CATCHES = {"batch_statistic.py"}


@pytest.mark.parametrize("name,guard", sorted(STATIC_CATCHES.items()))
def test_static_guards_reject_leaky_cells(name: str, guard: GuardId) -> None:
    failure = first_failure(run_static_gauntlet((FIXTURES / name).read_text()))
    assert failure is not None, f"{name} passed the static gauntlet"
    assert failure.guard is guard
    assert failure.repair_hint


@pytest.mark.parametrize("name", sorted(RUNTIME_CATCHES | {"clean_reference.py"}))
def test_static_guards_clear_runtime_only_cells(name: str) -> None:
    """Batch-statistic leakage is invisible to the AST and must reach the runtime guard."""
    assert first_failure(run_static_gauntlet((FIXTURES / name).read_text())) is None


@pytest.mark.parametrize("name", sorted(SAMPLING_CATCHES))
def test_row_sampling_catches_a_statistic_that_shifts_every_row(name: str) -> None:
    frame = make_frame()
    cell = instantiate_cell((FIXTURES / name).read_text(), name)
    cell.fit(frame, pd.Series(np.zeros(len(frame))))
    full = cell.transform(frame)
    singles = {i: cell.transform(frame.iloc[[i]]) for i in (3, 11, 25)}

    outcome = check_row_independence(full, singles)
    assert not outcome.passed
    assert outcome.guard is GuardId.ROW_INDEPENDENCE


@pytest.mark.parametrize("name", sorted(RUNTIME_CATCHES - SAMPLING_CATCHES))
def test_row_sampling_can_miss_a_leak_confined_to_certain_rows(name: str) -> None:
    """Documents the blind spot that motivates the whole-slice check below."""
    frame = make_frame()
    cell = instantiate_cell((FIXTURES / name).read_text(), name)
    cell.fit(frame, pd.Series(np.zeros(len(frame))))
    full = cell.transform(frame)
    unaffected = {i: cell.transform(frame.iloc[[i]]) for i in (3, 11, 25)}

    assert check_row_independence(full, unaffected).passed


@pytest.mark.parametrize("name", sorted(RUNTIME_CATCHES))
def test_batch_invariance_catches_every_runtime_leak(name: str) -> None:
    """No sampling luck involved: the slice covers half the rows at once."""
    frame = make_frame()
    cell = instantiate_cell((FIXTURES / name).read_text(), name)
    cell.fit(frame, pd.Series(np.zeros(len(frame))))
    positions = tuple(range(len(frame) // 2))

    outcome = check_batch_invariance(
        cell.transform(frame), cell.transform(frame.iloc[list(positions)]), positions
    )
    assert not outcome.passed
    assert outcome.guard is GuardId.ROW_INDEPENDENCE


def test_row_independence_clears_clean_cell() -> None:
    frame = make_frame()
    cell = instantiate_cell((FIXTURES / "clean_reference.py").read_text())
    cell.fit(frame, pd.Series(np.zeros(len(frame))))
    full = cell.transform(frame)
    singles = {i: cell.transform(frame.iloc[[i]]) for i in (3, 11, 25)}

    assert check_row_independence(full, singles).passed


def test_schema_guard_flags_broken_output() -> None:
    before = make_frame()
    assert check_schema(before, before, max_features=10).passed
    assert not check_schema(before, before.head(10), max_features=10).passed
    assert not check_schema(before, before, max_features=2).passed

    with_null = before.copy()
    with_null["dead"] = np.nan
    assert not check_schema(before, with_null, max_features=10).passed


def test_determinism_guard() -> None:
    frame = make_frame()
    assert check_determinism([frame, frame.copy()]).passed

    drifted = frame.copy()
    drifted.loc[drifted.index[0], "num_0"] += 1e-6
    assert not check_determinism([frame, drifted]).passed


def test_permutation_guard() -> None:
    assert check_permutation(permuted_score=0.27, chance_score=0.265, tolerance=0.02).passed
    assert not check_permutation(permuted_score=0.45, chance_score=0.265, tolerance=0.02).passed


def test_gauntlet_stops_at_first_failure() -> None:
    outcomes = run_static_gauntlet("def broken(:\n    pass\n")
    assert len(outcomes) == 1
    assert outcomes[0].guard is GuardId.SYNTAX


def test_import_allowlist_blocks_io() -> None:
    code = "import os\n\n\nclass FeatureStep:\n    def fit(self, X, y):\n        return self\n\n    def transform(self, X):\n        return X\n"
    failure = first_failure(run_static_gauntlet(code))
    assert failure is not None and failure.guard is GuardId.IMPORTS


def test_transform_may_not_accept_target() -> None:
    code = "class FeatureStep:\n    def fit(self, X, y):\n        return self\n\n    def transform(self, X, y):\n        return X\n"
    failure = first_failure(run_static_gauntlet(code))
    assert failure is not None and failure.guard is GuardId.CLASS_SHAPE


def test_batch_invariance_clears_a_clean_cell() -> None:
    frame = make_frame()
    cell = instantiate_cell((FIXTURES / "clean_reference.py").read_text())
    cell.fit(frame, pd.Series(np.zeros(len(frame))))

    positions = tuple(range(len(frame) // 2))
    assert check_batch_invariance(
        cell.transform(frame), cell.transform(frame.iloc[list(positions)]), positions
    ).passed


def test_batch_invariance_requires_aligned_input() -> None:
    frame = make_frame()
    with pytest.raises(ValueError):
        check_batch_invariance(frame, frame.head(5), (0, 1, 2))