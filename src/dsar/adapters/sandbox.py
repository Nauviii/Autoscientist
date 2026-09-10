"""Execute agent-written cells under resource limits, then adjudicate the guards.

Threat model: careless, not adversarial. Agent code is assumed to contain honest
mistakes that produce plausible numbers, so what is enforced here is correctness
and containment, not security. A determined escape is out of scope; the container
boundary handles that.

Forking rather than spawning is deliberate. A spawned worker re-imports pandas,
numpy and sklearn on every fold, costing roughly a second against fits that take a
quarter of one. Forking inherits the loaded modules and the dataset through
copy-on-write, so the per-fold overhead falls to tens of milliseconds.

Known limitation. Forking a process that already has threads is unsafe in general:
a lock held by another thread at fork time stays locked forever in the child. The
boosters here start OpenMP thread pools, so the parent is multi-threaded once a fit
has run. In practice children only execute pandas transforms and have not deadlocked
across the test suite, but the risk is real rather than theoretical. If it ever
surfaces, the fix is a forkserver started before any model is loaded, trading the
copy-on-write dataset for a clean fork ancestor.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from ..core.contracts import ProbeResult
from ..core.guards import (
    GuardId,
    GuardOutcome,
    check_batch_invariance,
    check_determinism,
    check_row_independence,
    check_schema,
    first_failure,
    run_static_gauntlet,
)
from ..ports import FoldSpec, ResourceLimits, TransformArtifacts

CELL_CLASS = "FeatureStep"

# Rows sampled for the row-independence check. Batch statistics affect every row,
# so a handful is enough to expose one.
INDEPENDENCE_SAMPLES = 5

# Builtins exposed to agent code. __import__ is present because an import
# statement compiles to a call on it; the static allowlist has already vetted
# which modules may be named.
_SAFE_BUILTINS: dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
        "format", "frozenset", "getattr", "hasattr", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "print", "range", "repr",
        "reversed", "round", "set", "setattr", "slice", "sorted", "str", "sum", "super",
        "tuple", "type", "zip", "__import__", "__build_class__", "__name__",
        "Exception", "ValueError", "KeyError", "TypeError",
    )
}

_FRAME: pd.DataFrame | None = None
_TARGET: pd.Series | None = None


def install_dataset(frame: pd.DataFrame, target: pd.Series) -> None:
    """Load the dataset into the parent once; children inherit it through fork."""
    global _FRAME, _TARGET
    _FRAME, _TARGET = frame, target


POSIX_ONLY_MESSAGE = (
    "Execution requires Linux. This sandbox depends on fork and setrlimit, neither "
    "of which exists on Windows, and pinning the platform is what lets an "
    "environment fingerprint guarantee a reproducible score.\n\n"
    "  WSL2:    wsl --install -d Ubuntu-24.04, then run from inside ~/projects\n"
    "  Docker:  docker compose run --rm shell"
)


def require_posix() -> None:
    """Fail loudly rather than degrade the isolation guarantees in silence.

    The resource module is imported here rather than at module scope so that a
    Windows user gets this explanation instead of a bare ModuleNotFoundError.
    """
    if os.name != "posix":
        raise RuntimeError(POSIX_ONLY_MESSAGE)


def _apply_limits(limits: ResourceLimits) -> None:
    """Cap address space and CPU time in the child, never in the parent."""
    import resource

    memory = limits.memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))


def instantiate(code: str) -> Any:
    """Build a cell from source inside a namespace with restricted builtins."""
    namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    exec(compile(code, "<cell>", "exec"), namespace)
    if CELL_CLASS not in namespace:
        raise ValueError(f"no class named {CELL_CLASS}")
    return namespace[CELL_CLASS]()


def _fit_pipeline(cells: Sequence[str], train: pd.DataFrame, y: pd.Series):
    """Fit every cell in order, returning the fitted cells and the transformed train."""
    fitted = []
    current = train
    for code in cells:
        cell = instantiate(code)
        cell.fit(current, y)
        current = cell.transform(current)
        fitted.append(cell)
    return fitted, current


def _apply(fitted: Sequence[Any], frame: pd.DataFrame) -> pd.DataFrame:
    for cell in fitted:
        frame = cell.transform(frame)
    return frame


def _child(connection, cells, fold, limits, collect) -> None:
    """Body of the forked worker. Never raises into the parent."""
    started = time.monotonic()
    try:
        _apply_limits(limits)
        assert _FRAME is not None and _TARGET is not None

        train = _FRAME.iloc[list(fold.train)]
        validate = _FRAME.iloc[list(fold.validate)]
        y = _TARGET.iloc[list(fold.train)]

        fitted, train_out = _fit_pipeline(cells, train, y)
        validate_out = _apply(fitted, validate)

        singles: dict[int, pd.DataFrame] = {}
        subset = None
        positions: tuple[int, ...] = ()
        replay = None
        if collect:
            count = min(INDEPENDENCE_SAMPLES, len(validate))
            for offset in range(count):
                position = round(offset * (len(validate) - 1) / max(count - 1, 1))
                singles[position] = _apply(fitted, validate.iloc[[position]])
            # Half the fold at once, which no sampling scheme can miss.
            positions = tuple(range(len(validate) // 2))
            if positions:
                subset = _apply(fitted, validate.iloc[list(positions)])
            replay = _apply(fitted, validate)

        connection.send(
            TransformArtifacts(
                train=train_out,
                validate=validate_out,
                singles=singles,
                subset=subset,
                subset_positions=positions,
                replay=replay,
                seconds=time.monotonic() - started,
            )
        )
    except MemoryError:
        connection.send(_failure("MemoryError: cell exceeded the address space limit", started))
    except BaseException as exc:  # noqa: BLE001 - the child must never propagate
        connection.send(_failure(f"{type(exc).__name__}: {exc}", started))
    finally:
        connection.close()


def _failure(message: str, started: float) -> TransformArtifacts:
    return TransformArtifacts(
        train=None, validate=None, error=message, seconds=time.monotonic() - started
    )


@dataclass
class ForkExecutor:
    """ExecutorPort backed by forked workers. Requires a POSIX host."""

    frame: pd.DataFrame
    target: pd.Series
    _context: Any = field(init=False)

    def __post_init__(self) -> None:
        require_posix()
        install_dataset(self.frame, self.target)
        self._context = mp.get_context("fork")

    def _run(self, target, args, wall_seconds: int) -> TransformArtifacts:
        """Run a child with a wall-clock deadline separate from the CPU limit.

        RLIMIT_CPU only counts processor time, so a cell blocked on something other
        than computation would otherwise hang the loop indefinitely.
        """
        started = time.monotonic()
        parent, child = self._context.Pipe(duplex=False)
        process = self._context.Process(target=target, args=(child, *args))
        process.start()
        child.close()

        reported = False
        try:
            if parent.poll(wall_seconds):
                result = parent.recv()
                reported = True
            else:
                result = _failure(f"timeout: exceeded {wall_seconds}s wall clock", started)
        except EOFError:
            result = _failure("worker died without reporting; likely a resource limit", started)
        finally:
            parent.close()
            if process.is_alive():
                process.terminate()
                process.join(2)
                if process.is_alive():
                    process.kill()
            process.join()

        # The child is terminated once it has reported, so its exit code is only
        # informative when nothing came back.
        if not reported and process.exitcode not in (0, None):
            result = _failure(f"{result.error} (exit code {process.exitcode})", started)
        return result

    def run_cells(
        self,
        cells: Sequence[str],
        fold: FoldSpec,
        limits: ResourceLimits,
        collect_guard_evidence: bool = False,
    ) -> TransformArtifacts:
        return self._run(
            _child, (list(cells), fold, limits, collect_guard_evidence), limits.wall_seconds
        )

    def run_probe(
        self, code: str, kind: str, params: Mapping[str, object], limits: ResourceLimits
    ) -> ProbeResult:
        artifacts = self._run(_probe_child, (code, params, limits), limits.wall_seconds)
        payload: Mapping[str, object] = (
            {"error": artifacts.error}
            if artifacts.failed
            else {"rows": len(artifacts.validate or []), "result": "ok"}
        )
        return ProbeResult(probe_id=kind, kind=kind, params=dict(params), payload=payload)


def _probe_child(connection, code, params, limits) -> None:
    """Read-only probe over the exploration split. May see the target."""
    started = time.monotonic()
    try:
        _apply_limits(limits)
        namespace: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
        exec(compile(code, "<probe>", "exec"), namespace)
        result = namespace["probe"](_FRAME, _TARGET, **dict(params))
        connection.send(
            TransformArtifacts(
                train=None, validate=pd.DataFrame(result), seconds=time.monotonic() - started
            )
        )
    except BaseException as exc:  # noqa: BLE001
        connection.send(_failure(f"{type(exc).__name__}: {exc}", started))
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class GauntletOutcome:
    """Verdict of the full guard sequence, plus the artifacts if it survived."""

    outcomes: tuple[GuardOutcome, ...]
    artifacts: TransformArtifacts | None

    @property
    def passed(self) -> bool:
        return all(o.passed for o in self.outcomes) and self.artifacts is not None

    @property
    def failure(self) -> GuardOutcome | None:
        return first_failure(self.outcomes)


def run_gauntlet(
    cells: Sequence[str],
    executor: Any,
    fold: FoldSpec,
    limits: ResourceLimits,
) -> GauntletOutcome:
    """Static guards first, then one execution that feeds every runtime guard.

    Ordering is by cost. The static pass costs a few milliseconds and rejects
    roughly half of what an agent proposes, so paying for a fit before running it
    would waste most of that budget.
    """
    outcomes: list[GuardOutcome] = []
    for code in cells:
        static = run_static_gauntlet(code)
        outcomes.extend(static)
        if first_failure(static) is not None:
            return GauntletOutcome(tuple(outcomes), None)

    artifacts = executor.run_cells(cells, fold, limits, collect_guard_evidence=True)
    if artifacts.failed:
        outcomes.append(GuardOutcome(GuardId.SMOKE, False, artifacts.error or "execution failed"))
        return GauntletOutcome(tuple(outcomes), None)

    assert artifacts.validate is not None
    original = executor.frame.iloc[list(fold.validate)]

    outcomes.append(check_schema(original, artifacts.validate, limits.max_features))
    if artifacts.replay is not None:
        outcomes.append(check_determinism([artifacts.validate, artifacts.replay]))
    if artifacts.singles:
        outcomes.append(check_row_independence(artifacts.validate, artifacts.singles))
    if artifacts.subset is not None:
        outcomes.append(
            check_batch_invariance(
                artifacts.validate, artifacts.subset, artifacts.subset_positions
            )
        )

    failed = first_failure(outcomes)
    return GauntletOutcome(tuple(outcomes), None if failed else artifacts)