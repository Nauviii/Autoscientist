"""Complexity is the tie-breaker for the deliverable, so it must not reward verbosity."""

from __future__ import annotations

import pytest

from dsar.core.complexity import (
    Candidate,
    ast_node_count,
    build_complexity,
    complexity_report,
    feature_expansion,
    nesting_depth,
    pareto_front,
    simplest,
)
from dsar.core.contracts import ComplexityVector

FLAT = '''
class FeatureStep:
    def fit(self, X, y):
        self.median_ = X["num_1"].median()
        return self

    def transform(self, X):
        out = X.copy()
        out["num_1"] = out["num_1"].fillna(self.median_)
        return out
'''

VERBOSE = '''
class FeatureStep:
    """A long docstring that says nothing the code does not already say."""

    def fit(self, X, y):
        training_frame_median_value = X["num_1"].median()
        self.training_frame_median_value = training_frame_median_value
        return self

    def transform(self, X):
        transformed_output_frame = X.copy()
        transformed_output_frame["num_1"] = transformed_output_frame["num_1"].fillna(
            self.training_frame_median_value
        )
        return transformed_output_frame
'''

NESTED = '''
class FeatureStep:
    def fit(self, X, y):
        self.cols_ = [c for c in X.columns if c.startswith("num_")]
        return self

    def transform(self, X):
        out = X.copy()
        for col in self.cols_:
            if out[col].isna().any():
                for other in self.cols_:
                    out[f"{col}_{other}"] = out[col] * out[other]
        return out
'''


def vector(features: int, cells: int, depth: int, seconds: float, nodes: int) -> ComplexityVector:
    return ComplexityVector(
        n_features=features,
        n_cells=cells,
        pipeline_depth=depth,
        fit_seconds=seconds,
        ast_nodes=nodes,
    )


def test_verbosity_does_not_inflate_the_node_count() -> None:
    """Measured on the canonical form, so a wordy rewrite draws instead of losing."""
    assert ast_node_count(FLAT) == pytest.approx(ast_node_count(VERBOSE), rel=0.25)


def test_branching_raises_the_depth() -> None:
    assert nesting_depth(FLAT) == 0
    assert nesting_depth(NESTED) >= 3


def test_pipeline_depth_takes_the_deepest_cell() -> None:
    complexity = build_complexity([FLAT, NESTED], n_features=40, fit_seconds=2.0)
    assert complexity.n_cells == 2
    assert complexity.pipeline_depth == nesting_depth(NESTED)
    assert complexity.ast_nodes == ast_node_count(FLAT) + ast_node_count(NESTED)


def test_feature_expansion_is_relative_to_the_raw_columns() -> None:
    assert feature_expansion(20, 60) == 3.0
    assert feature_expansion(0, 5) == 5.0


def test_dominance_needs_a_strict_improvement_somewhere() -> None:
    base = vector(40, 2, 1, 1.0, 200)
    assert vector(30, 2, 1, 1.0, 200).dominates(base)
    assert not base.dominates(base)
    assert not vector(30, 3, 1, 1.0, 200).dominates(base)


def test_pareto_front_keeps_both_ends_of_the_trade_off() -> None:
    """A higher score and a simpler pipeline are both defensible answers."""
    candidates = [
        Candidate("accurate_complex", 0.72, vector(180, 5, 3, 4.0, 900)),
        Candidate("balanced", 0.70, vector(60, 3, 1, 2.0, 400)),
        Candidate("simple", 0.66, vector(20, 1, 0, 0.5, 120)),
        Candidate("dominated", 0.66, vector(90, 4, 2, 3.0, 700)),
    ]
    front = pareto_front(candidates)
    assert "dominated" not in front
    assert {"accurate_complex", "simple"} <= set(front)


def test_epsilon_drops_candidates_kept_alive_by_noise() -> None:
    """A gap far below what the data resolves should not protect a complex pipeline."""
    candidates = [
        Candidate("complex", 0.7010, vector(180, 5, 3, 4.0, 900)),
        Candidate("simple", 0.7000, vector(20, 1, 0, 0.5, 120)),
    ]
    assert set(pareto_front(candidates)) == {"complex", "simple"}
    assert pareto_front(candidates, epsilon=0.005) == ("simple",)


def test_simplest_picks_the_lightest_pipeline() -> None:
    reference = vector(20, 1, 0, 0.5, 120)
    candidates = [
        Candidate("heavy", 0.72, vector(180, 5, 3, 4.0, 900)),
        Candidate("light", 0.71, vector(30, 2, 1, 0.8, 180)),
    ]
    assert simplest(candidates, reference) == "light"


def test_simplest_breaks_ties_deterministically() -> None:
    reference = vector(20, 1, 0, 0.5, 120)
    shared = vector(30, 2, 1, 0.8, 180)
    candidates = [Candidate("b", 0.70, shared), Candidate("a", 0.70, shared)]
    assert simplest(candidates, reference) == "a"


def test_simplest_rejects_an_empty_set() -> None:
    with pytest.raises(ValueError):
        simplest([], vector(20, 1, 0, 0.5, 120))


def test_complexity_report_is_expressed_as_ratios() -> None:
    reference = vector(20, 1, 1, 1.0, 100)
    report = complexity_report(vector(40, 2, 1, 2.0, 300), reference)
    assert report["n_features"] == 2.0
    assert report["ast_nodes"] == 3.0
