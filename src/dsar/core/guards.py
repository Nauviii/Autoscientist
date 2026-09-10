"""Correctness guards for agent-written cells. Pure: analysis and adjudication only.

This is a correctness sandbox, not a security one. Agent code is assumed careless,
not adversarial; the guards catch methodological faults (leakage, nondeterminism,
broken contracts) that produce plausible numbers rather than errors.

Static guards inspect the AST and never execute anything. Runtime guards receive
artifacts already produced by adapters.sandbox and only adjudicate them, which keeps
this module free of subprocess and resource imports.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import pandas as pd

CELL_CLASS = "FeatureStep"

DEFAULT_IMPORT_ALLOWLIST = frozenset(
    {"pandas", "numpy", "sklearn", "scipy", "math", "itertools", "collections", "typing"}
)

FORBIDDEN_CALLS = frozenset(
    {"exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars", "input"}
)

FORBIDDEN_ATTRIBUTES = frozenset(
    {"__class__", "__globals__", "__subclasses__", "__bases__", "__dict__", "__builtins__"}
)

# Unseeded generators. Seeded equivalents (default_rng(seed), RandomState(seed)) are
# allowed and handled separately.
RANDOM_ROOTS = frozenset({"random", "np.random", "numpy.random"})

FIT_METHODS = frozenset({"fit", "fit_transform", "fit_predict", "fit_resample"})


class GuardId(str, Enum):
    """Ordered by cost: cheap static checks run before anything is executed."""

    SYNTAX = "syntax"
    IMPORTS = "imports"
    FORBIDDEN_NODES = "forbidden_nodes"
    CLASS_SHAPE = "class_shape"
    TARGET_RETENTION = "target_retention"
    FIT_IN_TRANSFORM = "fit_in_transform"
    RANDOMNESS = "randomness"
    SMOKE = "smoke"
    SCHEMA = "schema"
    DETERMINISM = "determinism"
    ROW_INDEPENDENCE = "row_independence"
    PERMUTATION = "permutation"


REPAIR_HINTS: Mapping[GuardId, str] = {
    GuardId.IMPORTS: "Only pandas, numpy, sklearn and scipy may be imported. Remove any file, network or OS access.",
    GuardId.FORBIDDEN_NODES: "Remove exec, eval, open, __import__ and dunder attribute access.",
    GuardId.CLASS_SHAPE: f"Define exactly one class named {CELL_CLASS} with fit(self, X, y) and transform(self, X). transform must not take y.",
    GuardId.TARGET_RETENTION: "Do not store the target on the instance. Derive statistics from y inside fit and keep only those statistics.",
    GuardId.FIT_IN_TRANSFORM: "transform must not fit anything. Move every fit or fit_transform call into fit and store the fitted object as an attribute.",
    GuardId.RANDOMNESS: "Use a seeded generator, for example numpy.random.default_rng(0), so the pipeline is reproducible.",
    GuardId.SCHEMA: "Return a DataFrame with the same index, no all-null columns, and within the feature budget.",
    GuardId.DETERMINISM: "Two identical calls returned different output. Remove any dependence on unordered iteration or unseeded randomness.",
    GuardId.ROW_INDEPENDENCE: "transform computed a statistic from the batch it was given. Every statistic must be learned in fit and stored as an attribute, so a row transforms identically alone or in a batch.",
    GuardId.PERMUTATION: "The pipeline scores well above chance on a shuffled target, which indicates target leakage.",
}


@dataclass(frozen=True, slots=True)
class GuardOutcome:
    """Verdict for one guard. guard_id drives repair far more than the message does."""

    guard: GuardId
    passed: bool
    detail: str = ""

    @property
    def repair_hint(self) -> str:
        return "" if self.passed else REPAIR_HINTS.get(self.guard, "")


def _ok(guard: GuardId, detail: str = "") -> GuardOutcome:
    return GuardOutcome(guard, True, detail)


def _fail(guard: GuardId, detail: str) -> GuardOutcome:
    return GuardOutcome(guard, False, detail)


def _dotted(node: ast.AST) -> str:
    """Render an attribute chain such as np.random.rand back into dotted form."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _find_class(tree: ast.Module) -> ast.ClassDef | None:
    return next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CELL_CLASS), None
    )


def _find_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef | None:
    return next(
        (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name), None
    )


def check_syntax(code: str) -> tuple[GuardOutcome, ast.Module | None]:
    """Parse the cell. Every later static guard depends on this succeeding."""
    try:
        return _ok(GuardId.SYNTAX), ast.parse(code)
    except SyntaxError as exc:
        return _fail(GuardId.SYNTAX, f"line {exc.lineno}: {exc.msg}"), None


def check_imports(tree: ast.Module, allowlist: frozenset[str] = DEFAULT_IMPORT_ALLOWLIST) -> GuardOutcome:
    """Reject any import outside the numeric stack."""
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names if a.name.split(".")[0] not in allowlist]
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] not in allowlist:
                bad.append(node.module)
    return _ok(GuardId.IMPORTS) if not bad else _fail(GuardId.IMPORTS, f"disallowed: {sorted(set(bad))}")


def check_forbidden_nodes(tree: ast.Module) -> GuardOutcome:
    """Reject dynamic execution, file access and introspection escape hatches."""
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                bad.append(node.func.id)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            bad.append(node.attr)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad.append("global/nonlocal")
    return _ok(GuardId.FORBIDDEN_NODES) if not bad else _fail(
        GuardId.FORBIDDEN_NODES, f"forbidden: {sorted(set(bad))}"
    )


def check_class_shape(tree: ast.Module) -> GuardOutcome:
    """Enforce the cell contract, including that transform never receives the target."""
    cls = _find_class(tree)
    if cls is None:
        return _fail(GuardId.CLASS_SHAPE, f"no class named {CELL_CLASS}")

    fit, transform = _find_method(cls, "fit"), _find_method(cls, "transform")
    if fit is None or transform is None:
        return _fail(GuardId.CLASS_SHAPE, "both fit and transform are required")

    fit_args = [a.arg for a in fit.args.args]
    tr_args = [a.arg for a in transform.args.args]
    if fit_args[:3] != ["self", "X", "y"]:
        return _fail(GuardId.CLASS_SHAPE, f"fit signature is {fit_args}, expected (self, X, y)")
    if tr_args[:2] != ["self", "X"]:
        return _fail(GuardId.CLASS_SHAPE, f"transform signature is {tr_args}, expected (self, X)")
    if "y" in tr_args:
        return _fail(GuardId.CLASS_SHAPE, "transform must not accept y")
    return _ok(GuardId.CLASS_SHAPE)


def check_target_retention(tree: ast.Module) -> GuardOutcome:
    """Forbid stashing the raw target on the instance for later use in transform."""
    cls = _find_class(tree)
    fit = _find_method(cls, "fit") if cls else None
    if fit is None:
        return _ok(GuardId.TARGET_RETENTION)

    target = fit.args.args[2].arg if len(fit.args.args) > 2 else "y"
    for node in ast.walk(fit):
        if not isinstance(node, ast.Assign):
            continue
        stores_target = isinstance(node.value, ast.Name) and node.value.id == target
        to_attribute = any(
            isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
            for t in node.targets
        )
        if stores_target and to_attribute:
            return _fail(GuardId.TARGET_RETENTION, f"fit assigns {target} to an instance attribute")
    return _ok(GuardId.TARGET_RETENTION)


def check_fit_in_transform(tree: ast.Module) -> GuardOutcome:
    """Catch estimators refitted inside transform, the classic fold-safety break."""
    cls = _find_class(tree)
    transform = _find_method(cls, "transform") if cls else None
    if transform is None:
        return _ok(GuardId.FIT_IN_TRANSFORM)

    found = [
        node.func.attr
        for node in ast.walk(transform)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in FIT_METHODS
    ]
    return _ok(GuardId.FIT_IN_TRANSFORM) if not found else _fail(
        GuardId.FIT_IN_TRANSFORM, f"transform calls {sorted(set(found))}"
    )


def check_randomness(tree: ast.Module) -> GuardOutcome:
    """Flag unseeded generators, which break reproducibility without raising."""
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        dotted = _dotted(node.func)
        root = dotted.rsplit(".", 1)[0]
        if root in RANDOM_ROOTS and node.func.attr not in {"default_rng", "RandomState"}:
            bad.append(dotted)
        elif node.func.attr in {"default_rng", "RandomState"} and not (node.args or node.keywords):
            bad.append(f"{dotted}() without a seed")
    return _ok(GuardId.RANDOMNESS) if not bad else _fail(
        GuardId.RANDOMNESS, f"unseeded: {sorted(set(bad))}"
    )


def run_static_gauntlet(
    code: str, allowlist: frozenset[str] = DEFAULT_IMPORT_ALLOWLIST
) -> tuple[GuardOutcome, ...]:
    """Run every static guard, stopping at the first failure. Nothing is executed."""
    syntax, tree = check_syntax(code)
    if not syntax.passed or tree is None:
        return (syntax,)

    outcomes = [syntax]
    for outcome in (
        check_imports(tree, allowlist),
        check_forbidden_nodes(tree),
        check_class_shape(tree),
        check_target_retention(tree),
        check_fit_in_transform(tree),
        check_randomness(tree),
    ):
        outcomes.append(outcome)
        if not outcome.passed:
            break
    return tuple(outcomes)


def frame_signature(frame: pd.DataFrame, precision: int = 10) -> str:
    """Order-stable content hash of a DataFrame, used for determinism comparison."""
    payload = (
        ",".join(map(str, frame.columns))
        + "|"
        + ",".join(map(str, frame.dtypes))
        + "|"
        + frame.round(precision).to_csv(index=True)
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def check_schema(
    before: pd.DataFrame, after: pd.DataFrame, max_features: int
) -> GuardOutcome:
    """Validate the transform output before it reaches the model."""
    problems: list[str] = []
    if not isinstance(after, pd.DataFrame):
        return _fail(GuardId.SCHEMA, f"returned {type(after).__name__}, expected DataFrame")
    if len(after) != len(before):
        problems.append(f"row count {len(before)} -> {len(after)}")
    if not after.index.equals(before.index):
        problems.append("index not preserved")
    if after.shape[1] > max_features:
        problems.append(f"{after.shape[1]} features exceeds budget {max_features}")
    all_null = [c for c in after.columns if after[c].isna().all()]
    if all_null:
        problems.append(f"all-null columns: {all_null[:5]}")
    if after.columns.duplicated().any():
        problems.append("duplicate column names")
    return _ok(GuardId.SCHEMA) if not problems else _fail(GuardId.SCHEMA, "; ".join(problems))


def check_determinism(outputs: Sequence[pd.DataFrame]) -> GuardOutcome:
    """Require identical output across repeated calls on identical input."""
    if len(outputs) < 2:
        raise ValueError("determinism needs at least two runs")
    signatures = {frame_signature(frame) for frame in outputs}
    return _ok(GuardId.DETERMINISM) if len(signatures) == 1 else _fail(
        GuardId.DETERMINISM, f"{len(signatures)} distinct signatures across {len(outputs)} runs"
    )


def check_row_independence(
    full: pd.DataFrame, singles: Mapping[int, pd.DataFrame], tolerance: float = 1e-9
) -> GuardOutcome:
    """Require each row to transform identically alone and inside a batch.

    The cheapest guard that catches transductive leakage: a transform deriving any
    statistic from the batch it receives fails here while raising no error and
    inflating the score.

    Empirical, not a proof. It can only observe behaviour the sampled rows provoke,
    so a leaky operation that happens to be a no-op on those rows passes. Sampling
    across the fold rather than from one end reduces the blind spot without
    removing it.
    """
    if not singles:
        raise ValueError("row independence needs at least one sampled row")

    mismatched: list[int] = []
    for position, single in singles.items():
        expected = full.iloc[[position]].reset_index(drop=True)
        actual = single.reset_index(drop=True)
        if list(expected.columns) != list(actual.columns):
            mismatched.append(position)
            continue
        numeric = expected.select_dtypes("number").columns
        other = [c for c in expected.columns if c not in numeric]
        numeric_ok = (
            (expected[numeric] - actual[numeric]).abs().le(tolerance).all().all()
            if len(numeric)
            else True
        )
        other_ok = expected[other].equals(actual[other]) if other else True
        if not (numeric_ok and other_ok):
            mismatched.append(position)

    return _ok(GuardId.ROW_INDEPENDENCE) if not mismatched else _fail(
        GuardId.ROW_INDEPENDENCE, f"rows differ when transformed alone: {sorted(mismatched)}"
    )


def check_batch_invariance(
    full: pd.DataFrame,
    subset: pd.DataFrame,
    positions: Sequence[int],
    tolerance: float = 1e-9,
) -> GuardOutcome:
    """Require a slice of rows to transform identically inside and outside the batch.

    Strictly stronger than sampling individual rows: every row in the slice is
    checked at once, so a leaky statistic cannot hide in the rows that happened not
    to be sampled. Any statistic derived from the batch shifts when the batch does.
    """
    if len(subset) != len(positions):
        raise ValueError("subset and positions must align")
    if list(full.columns) != list(subset.columns):
        return _fail(GuardId.ROW_INDEPENDENCE, "column set changed with the batch")

    expected = full.iloc[list(positions)].reset_index(drop=True)
    actual = subset.reset_index(drop=True)
    numeric = expected.select_dtypes("number").columns
    other = [c for c in expected.columns if c not in numeric]

    numeric_ok = (
        (expected[numeric] - actual[numeric]).abs().le(tolerance).all()
        if len(numeric)
        else pd.Series(dtype=bool)
    )
    drifted = [str(c) for c in numeric if not bool(numeric_ok[c])]
    drifted += [c for c in other if not expected[c].equals(actual[c])]

    return _ok(GuardId.ROW_INDEPENDENCE) if not drifted else _fail(
        GuardId.ROW_INDEPENDENCE,
        f"columns depend on the batch they are transformed with: {sorted(set(drifted))[:5]}",
    )


def check_permutation(
    permuted_score: float, chance_score: float, tolerance: float
) -> GuardOutcome:
    """Require near-chance performance once the target has been shuffled."""
    excess = permuted_score - chance_score
    return _ok(GuardId.PERMUTATION, f"excess {excess:+.4f}") if excess <= tolerance else _fail(
        GuardId.PERMUTATION,
        f"scored {permuted_score:.4f} on a shuffled target, {excess:+.4f} above chance {chance_score:.4f}",
    )


def first_failure(outcomes: Sequence[GuardOutcome]) -> GuardOutcome | None:
    """The guard that should drive the repair prompt, if any failed."""
    return next((o for o in outcomes if not o.passed), None)