"""Complexity measurement for pipelines. Pure: static analysis and set operations.

Complexity is the tie-breaker for the deliverable, so it is measured on the
canonical form. Otherwise a verbose rewrite of the same transform would look more
complex than a terse one and lose a comparison it should have drawn.

Components stay separate rather than being collapsed into one number. Scalarising
demands a weight vector nobody can defend; the Pareto front shows the trade-off
instead and lets the reader decide.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import ComplexityVector
from .fingerprint import canonical_source

# Statements that add a branch or a loop, which is what makes a cell hard to read.
NESTING_NODES = (
    ast.If,
    ast.For,
    ast.While,
    ast.With,
    ast.Try,
    ast.ListComp,
    ast.DictComp,
    ast.SetComp,
    ast.GeneratorExp,
)

DEFAULT_WEIGHTS: Mapping[str, float] = {
    "n_features": 0.3,
    "n_cells": 0.3,
    "pipeline_depth": 0.2,
    "fit_seconds": 0.1,
    "ast_nodes": 0.1,
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One pipeline positioned on the score and complexity axes."""

    candidate_id: str
    score: float
    complexity: ComplexityVector


def ast_node_count(code: str) -> int:
    """Node count of the canonical form, so cosmetic verbosity does not inflate it."""
    return sum(1 for _ in ast.walk(ast.parse(canonical_source(code))))


def nesting_depth(code: str) -> int:
    """Deepest branch or loop nesting in a cell."""

    def depth(node: ast.AST, current: int = 0) -> int:
        deepest = current
        for child in ast.iter_child_nodes(node):
            step = current + 1 if isinstance(child, NESTING_NODES) else current
            deepest = max(deepest, depth(child, step))
        return deepest

    return depth(ast.parse(code))


def build_complexity(
    cells: Sequence[str], n_features: int, fit_seconds: float
) -> ComplexityVector:
    """Assemble the complexity vector for a pipeline that has already been run."""
    return ComplexityVector(
        n_features=n_features,
        n_cells=len(cells),
        pipeline_depth=max((nesting_depth(cell) for cell in cells), default=0),
        fit_seconds=fit_seconds,
        ast_nodes=sum(ast_node_count(cell) for cell in cells),
    )


def feature_expansion(before: int, after: int) -> float:
    """How far a pipeline widened the matrix, relative to the raw columns."""
    return after / max(before, 1)


def pareto_front(candidates: Sequence[Candidate], epsilon: float = 0.0) -> tuple[str, ...]:
    """Candidates no other candidate beats on score while being no more complex.

    epsilon lets a marginally lower score still count as matched, so a pipeline is
    not kept alive by a difference far below what the data can resolve.
    """
    front: list[str] = []
    for candidate in candidates:
        dominated = any(
            other.candidate_id != candidate.candidate_id
            and other.score >= candidate.score - epsilon
            and (
                other.complexity.dominates(candidate.complexity)
                or (other.complexity == candidate.complexity and other.score > candidate.score)
            )
            for other in candidates
        )
        if not dominated:
            front.append(candidate.candidate_id)
    return tuple(sorted(front))


def simplest(
    candidates: Sequence[Candidate],
    reference: ComplexityVector,
    weights: Mapping[str, float] | None = None,
) -> str:
    """Least complex candidate, with the id as a deterministic tie-break."""
    if not candidates:
        raise ValueError("no candidates to choose from")
    active = weights or DEFAULT_WEIGHTS
    return min(
        candidates,
        key=lambda c: (c.complexity.scalar(active, reference), c.candidate_id),
    ).candidate_id


def complexity_report(
    candidate: ComplexityVector, reference: ComplexityVector
) -> Mapping[str, float]:
    """Per-component ratio against the reference pipeline, for the final report."""
    return {
        "n_features": candidate.n_features / max(reference.n_features, 1),
        "n_cells": candidate.n_cells / max(reference.n_cells, 1),
        "pipeline_depth": candidate.pipeline_depth / max(reference.pipeline_depth, 1),
        "fit_seconds": candidate.fit_seconds / max(reference.fit_seconds, 1e-6),
        "ast_nodes": candidate.ast_nodes / max(reference.ast_nodes, 1),
    }