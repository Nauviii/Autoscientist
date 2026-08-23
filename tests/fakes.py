"""In-memory implementations of every port, so a session can run with no side effects.

These are not mocks. Each one behaves like the real adapter within its contract:
the store enforces immutability, the executor actually runs the cells, the clock
advances. What they drop is the part that touches the outside world.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

import pandas as pd

from dsar.core.contracts import Evidence, ExperimentResult, ProbeResult
from dsar.ports import (
    ApprovalDecision,
    ApprovalRequest,
    ArtifactRef,
    FoldSpec,
    LLMRequest,
    LLMResponse,
    ResourceLimits,
    TransformArtifacts,
)


@dataclass
class ScriptedLLM:
    """Returns queued replies in order, recording every request for assertions."""

    replies: list[str] = field(default_factory=list)
    requests: list[LLMRequest] = field(default_factory=list)
    default: str = ""

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = self.replies.pop(0) if self.replies else self.default
        cacheable = request.context.cacheable_tokens
        return LLMResponse(
            text=text,
            model=request.model,
            input_tokens=request.context.total_tokens,
            output_tokens=max(1, len(text) // 4),
            cached_input_tokens=cacheable if len(self.requests) > 1 else 0,
        )


@dataclass
class InProcessExecutor:
    """Runs cells in this process. Real behaviour, no isolation.

    Suitable for tests only: a runaway cell takes the test runner down with it,
    which is exactly why the production adapter forks.
    """

    frame: pd.DataFrame
    target: pd.Series
    single_positions: tuple[int, ...] = (0, 1, 2)

    def _instantiate(self, code: str) -> Any:
        namespace: dict[str, Any] = {}
        exec(compile(code, "<cell>", "exec"), namespace)
        return namespace["FeatureStep"]()

    def run_cells(
        self,
        cells: Sequence[str],
        fold: FoldSpec,
        limits: ResourceLimits,
        collect_guard_evidence: bool = False,
    ) -> TransformArtifacts:
        try:
            train = self.frame.iloc[list(fold.train)]
            validate = self.frame.iloc[list(fold.validate)]
            y = self.target.iloc[list(fold.train)]

            fitted = []
            for code in cells:
                cell = self._instantiate(code)
                cell.fit(train, y)
                train = cell.transform(train)
                fitted.append(cell)

            out = validate
            for cell in fitted:
                out = cell.transform(out)

            singles: dict[int, pd.DataFrame] = {}
            replay = None
            if collect_guard_evidence:
                for position in self.single_positions:
                    if position >= len(validate):
                        continue
                    row = validate.iloc[[position]]
                    for cell in fitted:
                        row = cell.transform(row)
                    singles[position] = row
                replay = validate
                for cell in fitted:
                    replay = cell.transform(replay)

            return TransformArtifacts(
                train=train, validate=out, singles=singles, replay=replay, seconds=0.01
            )
        except Exception as exc:
            return TransformArtifacts(
                train=None, validate=None, error=f"{type(exc).__name__}: {exc}"
            )

    def run_probe(
        self, code: str, kind: str, params: Mapping[str, object], limits: ResourceLimits
    ) -> ProbeResult:
        return ProbeResult(
            probe_id=hashlib.sha256(code.encode()).hexdigest()[:12],
            kind=kind,
            params=dict(params),
            payload={"rows": len(self.frame), "columns": list(self.frame.columns)},
        )


@dataclass
class MemoryStore:
    """Append-only in-memory store that refuses to overwrite an existing artifact."""

    _artifacts: dict[str, bytes] = field(default_factory=dict)
    _experiments: list[ExperimentResult] = field(default_factory=list)
    _evidences: list[Evidence] = field(default_factory=list)

    def put_artifact(self, kind: str, payload: bytes) -> ArtifactRef:
        artifact_id = f"{kind}_{hashlib.sha256(payload).hexdigest()[:16]}"
        self._artifacts.setdefault(artifact_id, payload)
        return ArtifactRef(artifact_id, kind, len(payload))

    def get_artifact(self, artifact_id: str) -> bytes:
        if artifact_id not in self._artifacts:
            raise KeyError(artifact_id)
        return self._artifacts[artifact_id]

    def record_experiment(self, result: ExperimentResult) -> None:
        if any(e.experiment_id == result.experiment_id for e in self._experiments):
            raise ValueError(f"experiment {result.experiment_id} already recorded")
        self._experiments.append(result)

    def record_evidence(self, evidence: Evidence) -> None:
        self._evidences.append(evidence)

    def experiments(self) -> tuple[ExperimentResult, ...]:
        return tuple(self._experiments)

    def evidences(self) -> tuple[Evidence, ...]:
        return tuple(self._evidences)

    def fingerprints(self) -> frozenset[str]:
        return frozenset(e.fingerprint for e in self._experiments)

    def find_by_fingerprint(self, fingerprint: str) -> ExperimentResult | None:
        return next((e for e in self._experiments if e.fingerprint == fingerprint), None)


@dataclass
class FrozenClock:
    """Deterministic clock. Required for a golden session to replay byte-identically."""

    start: datetime = datetime(2026, 1, 1, 0, 0, 0)
    step_seconds: float = 1.0
    _ticks: int = 0

    def now(self) -> datetime:
        value = self.start + timedelta(seconds=self.step_seconds * self._ticks)
        self._ticks += 1
        return value

    def monotonic(self) -> float:
        value = self.step_seconds * self._ticks
        self._ticks += 1
        return value


@dataclass
class AutoApprover:
    """Approves everything while still recording what a human would have been asked."""

    requests: list[ApprovalRequest] = field(default_factory=list)
    approve: bool = True

    def request(self, approval: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(approval)
        return ApprovalDecision(
            approved=self.approve,
            note="auto mode: no human reviewed this gate",
            auto_approved=True,
        )
