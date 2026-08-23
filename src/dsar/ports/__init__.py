"""Boundaries between pure logic and the outside world.

Every side effect the system can have passes through one of these five protocols.
Nothing here performs one: these are shapes, and adapters supply the behaviour.

The point is testability. Because the orchestrator only ever holds a port, a whole
session can run against fakes with no API key, no subprocess and no disk, which is
what makes golden replay possible. It also keeps the dependency arrow one-way:
ports import from core, core imports nothing from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

from ..core.contracts import Evidence, ExperimentResult, ProbeResult
from ..core.projection import Context


# --------------------------------------------------------------------------- LLM


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One completion. The cache breakpoint comes from the context block ordering."""

    context: Context
    model: str
    temperature: float
    max_tokens: int
    purpose: Literal["propose", "repair", "probe", "interpret"]

    @property
    def cache_breakpoint(self) -> int:
        return self.context.cache_breakpoint


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Completion plus the accounting the budget and the audit trail both need."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    stop_reason: str = "end_turn"
    from_cache: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_ratio(self) -> float:
        return self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0


@runtime_checkable
class LLMPort(Protocol):
    """Text completion. The only component allowed to be nondeterministic."""

    def complete(self, request: LLMRequest) -> LLMResponse: ...


# ---------------------------------------------------------------------- execution


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Ceilings enforced per child process."""

    memory_mb: int = 2048
    cpu_seconds: int = 120
    wall_seconds: int = 180
    max_features: int = 200


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """One train/validate split. Indices are generated once and reused everywhere."""

    fold_index: int
    repeat: int
    train: tuple[int, ...]
    validate: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TransformArtifacts:
    """Everything the runtime guards need, gathered in a single execution.

    singles, subset and replay exist only so guards can be adjudicated in core:
    producing them here avoids a second round trip into the sandbox per check.
    """

    train: pd.DataFrame | None
    validate: pd.DataFrame | None
    singles: Mapping[int, pd.DataFrame] = field(default_factory=dict)
    subset: pd.DataFrame | None = None
    subset_positions: tuple[int, ...] = ()
    replay: pd.DataFrame | None = None
    seconds: float = 0.0
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@runtime_checkable
class ExecutorPort(Protocol):
    """Runs agent-written code under resource limits, in isolation from the parent."""

    def run_cells(
        self,
        cells: Sequence[str],
        fold: FoldSpec,
        limits: ResourceLimits,
        collect_guard_evidence: bool = False,
    ) -> TransformArtifacts: ...

    def run_probe(
        self, code: str, kind: str, params: Mapping[str, object], limits: ResourceLimits
    ) -> ProbeResult: ...


# --------------------------------------------------------------------------- store


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """Content-addressed handle. Artifacts are immutable once written."""

    artifact_id: str
    kind: str
    bytes_written: int


@runtime_checkable
class StorePort(Protocol):
    """Append-only record of the session. Nothing is ever updated in place."""

    def put_artifact(self, kind: str, payload: bytes) -> ArtifactRef: ...

    def get_artifact(self, artifact_id: str) -> bytes: ...

    def record_experiment(self, result: ExperimentResult) -> None: ...

    def record_evidence(self, evidence: Evidence) -> None: ...

    def experiments(self) -> tuple[ExperimentResult, ...]: ...

    def evidences(self) -> tuple[Evidence, ...]: ...

    def fingerprints(self) -> frozenset[str]: ...

    def find_by_fingerprint(self, fingerprint: str) -> ExperimentResult | None: ...


# --------------------------------------------------------------------------- clock


@runtime_checkable
class ClockPort(Protocol):
    """Abstracted so a recorded session replays to an identical transcript."""

    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...


# ------------------------------------------------------------------------ approval


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """A decision point, rendered as a diff against what the system would do alone."""

    gate: Literal["data_contract", "direction_change", "open_holdout", "invalid_rate"]
    summary: str
    proposed: Mapping[str, str]
    defaults: Mapping[str, str] = field(default_factory=dict)
    blocking: bool = True

    @property
    def diff(self) -> tuple[tuple[str, str, str], ...]:
        """Fields where the proposal departs from the system default."""
        return tuple(
            (key, self.defaults.get(key, "-"), value)
            for key, value in sorted(self.proposed.items())
            if self.defaults.get(key) != value
        )


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Outcome of a gate. auto_approved records that no human actually saw it."""

    approved: bool
    edits: Mapping[str, str] = field(default_factory=dict)
    note: str = ""
    auto_approved: bool = False


@runtime_checkable
class ApprovalPort(Protocol):
    """Human gate. In auto mode the adapter approves but still records the request."""

    def request(self, approval: ApprovalRequest) -> ApprovalDecision: ...


__all__ = [
    "ApprovalDecision",
    "ApprovalPort",
    "ApprovalRequest",
    "ArtifactRef",
    "ClockPort",
    "ExecutorPort",
    "FoldSpec",
    "LLMPort",
    "LLMRequest",
    "LLMResponse",
    "ResourceLimits",
    "StorePort",
    "TransformArtifacts",
]