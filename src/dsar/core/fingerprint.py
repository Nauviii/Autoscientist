"""Canonical hashing for duplicate detection and cache keys. Pure: no I/O.

Two independent layers, because they fail in different ways:

Structural. Agents restate the same transform with different variable names,
docstrings and annotations. Canonicalising the AST collapses those into one hash,
so the same idea is not paid for twice.

Behavioural. Different code can produce identical features. output_signature hashes
the resulting matrix instead of the source, catching what the AST cannot.

Neither proves semantic equivalence. Reordering independent statements yields a
different structural hash, which is deliberate: a false miss costs one experiment,
a false hit silently discards a real result.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .contracts import ModelConfig, ValidationSpec, canonical_hash

# Parameter names kept verbatim so the cell contract stays readable after renaming.
CONTRACT_NAMES = frozenset({"self", "X", "y"})

# Floats are rounded before hashing so 0.1 and 0.10000000000000001 agree.
PARAM_SIGNIFICANT_DIGITS = 6

# Column statistics are rounded before hashing for the same reason.
SIGNATURE_PRECISION = 8


def _hash(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def _round_significant(value: float, digits: int) -> float:
    """Round to a fixed number of significant digits rather than decimal places."""
    if value == 0.0 or value != value or value in (float("inf"), float("-inf")):
        return value
    from math import floor, log10

    exponent = floor(log10(abs(value)))
    return round(value, digits - 1 - exponent)


class _Canonicaliser(ast.NodeTransformer):
    """Rename locally bound names and self-attributes to positional placeholders.

    Only names bound inside the tree are renamed; imported and builtin names must
    survive, or two different libraries would collapse to the same hash.
    """

    def __init__(self) -> None:
        self._locals: dict[str, str] = {}
        self._attributes: dict[str, str] = {}
        self._self_alias: str | None = None

    def _local(self, name: str) -> str:
        if name in CONTRACT_NAMES:
            return name
        if name not in self._locals:
            self._locals[name] = f"v{len(self._locals)}"
        return self._locals[name]

    def _bind(self, node: ast.AST) -> None:
        """Register every name bound in this subtree before rewriting references."""
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                self._local(child.id)
            elif isinstance(child, ast.arg):
                self._local(child.arg)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for arg in child.args.args + child.args.kwonlyargs:
                    self._local(arg.arg)

    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = _strip_docstring(node.body)
        self._bind(node)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.body = _strip_docstring(node.body)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node.body = _strip_docstring(node.body)
        node.returns = None
        if node.args.args:
            self._self_alias = node.args.args[0].arg
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None
            arg.arg = self._local(arg.arg)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self._locals or isinstance(node.ctx, (ast.Store, ast.Del)):
            node.id = self._local(node.id)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        """Rename attributes hung on the instance; leave library attributes alone."""
        node.value = self.visit(node.value)
        target = node.value
        is_self = isinstance(target, ast.Name) and target.id in (
            self._self_alias,
            self._locals.get(self._self_alias or ""),
            "self",
        )
        if is_self and not node.attr.startswith("__"):
            if node.attr not in self._attributes:
                self._attributes[node.attr] = f"a{len(self._attributes)}"
            node.attr = self._attributes[node.attr]
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Drop the annotation, keeping the assignment when one exists."""
        if node.value is None:
            return ast.Pass()
        assign = ast.Assign(targets=[node.target], value=node.value)
        return self.visit(ast.copy_location(assign, node))


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:] or [ast.Pass()]
    return body


def canonical_source(code: str) -> str:
    """Rewrite a cell into canonical form: no docstrings, annotations or local names."""
    tree = _Canonicaliser().visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def cell_fingerprint(code: str) -> str:
    """Structural hash of one cell, invariant to cosmetic rewrites."""
    return _hash(canonical_source(code))


def pipeline_fingerprint(cells: Sequence[str]) -> str:
    """Hash of an ordered pipeline. Order is significant; cells are not commutative."""
    return _hash("|".join(cell_fingerprint(cell) for cell in cells))


def pipeline_prefixes(cells: Sequence[str]) -> tuple[str, ...]:
    """Fingerprint of every prefix, so an unchanged head can be reused from cache."""
    fingerprints = [cell_fingerprint(cell) for cell in cells]
    return tuple(_hash("|".join(fingerprints[: i + 1])) for i in range(len(fingerprints)))


def shared_prefix_length(left: Sequence[str], right: Sequence[str]) -> int:
    """How many leading cells two pipelines have in common."""
    a, b = pipeline_prefixes(left), pipeline_prefixes(right)
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def quantise_params(
    params: Mapping[str, object], digits: int = PARAM_SIGNIFICANT_DIGITS
) -> dict[str, object]:
    """Round float hyperparameters onto a grid so equivalent configs agree."""
    return {
        key: _round_significant(value, digits) if isinstance(value, float) else value
        for key, value in sorted(params.items())
    }


def model_fingerprint(model: ModelConfig) -> str:
    """Hash of the model family, quantised parameters and thread count.

    Thread count is included because reduction order changes results in the last
    digits, which would otherwise make a cached score irreproducible.
    """
    return canonical_hash(
        {
            "family": model.family,
            "params": quantise_params(model.params),
            "n_threads": model.n_threads,
        }
    )


def validation_fingerprint(validation: ValidationSpec) -> str:
    """Hash of the resampling protocol, including the seed that fixes the folds."""
    return canonical_hash(
        {
            "kind": validation.kind,
            "k": validation.k,
            "repeats": validation.repeats,
            "seed": validation.seed,
            "group_column": validation.group_column,
            "time_column": validation.time_column,
        }
    )


def environment_fingerprint(
    lockfile: bytes, python_version: str, library_versions: Mapping[str, str] | None = None
) -> str:
    """Hash of the resolved dependency set, so a library upgrade invalidates results."""
    return canonical_hash(
        {
            "lock": hashlib.sha256(lockfile).hexdigest()[:32],
            "python": python_version,
            "libraries": dict(sorted((library_versions or {}).items())),
        }
    )


def prompt_fingerprint(templates: Mapping[str, str]) -> str:
    """Hash of prompt templates; editing one makes later experiments incomparable."""
    return canonical_hash({name: _hash(text) for name, text in sorted(templates.items())})


def experiment_fingerprint(
    *,
    cells: Sequence[str],
    model: ModelConfig,
    validation: ValidationSpec,
    data_hash: str,
    environment: str,
    seed: int,
) -> str:
    """Identity of a runnable experiment. Equal fingerprints must yield equal scores."""
    return canonical_hash(
        {
            "pipeline": pipeline_fingerprint(cells),
            "model": model_fingerprint(model),
            "validation": validation_fingerprint(validation),
            "data": data_hash,
            "environment": environment,
            "seed": seed,
        }
    )


def output_signature(frame: pd.DataFrame, precision: int = SIGNATURE_PRECISION) -> str:
    """Behavioural hash of a feature matrix, invariant to column order and naming.

    Two pipelines written differently that yield the same features collapse here,
    which the structural hash cannot detect.
    """
    columns: list[tuple[str, ...]] = []
    for name in frame.columns:
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series):
            stats = series.describe()
            summary = tuple(
                f"{round(float(stats[key]), precision)}"
                for key in ("count", "mean", "std", "min", "50%", "max")
            )
        else:
            counts = series.astype("string").value_counts().sort_index()
            summary = (str(series.nunique(dropna=False)), _hash(counts.to_json(), 12))
        columns.append((str(series.dtype), str(int(series.isna().sum())), *summary))
    return _hash("|".join(sorted(",".join(col) for col in columns)))


def is_duplicate(
    candidate: str, seen: Iterable[str]
) -> bool:
    """Membership test kept explicit so callers record why an experiment was skipped."""
    return candidate in set(seen)