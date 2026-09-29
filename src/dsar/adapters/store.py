"""Durable session record: SQLite for the registry, content-addressed files for blobs.

Append-only by construction. Nothing here updates a row, because the research state
is a fold over what happened rather than a mutable summary of it. An experiment that
could be edited after the fact would make the ladder unreplayable, and a rerun could
then disagree with the report it produced.

The split is by size and by access. Metrics, verdicts and fingerprints are small,
queried constantly and belong in a table. Cell source, predictions and EDA reports
are large, read rarely and belong on disk under the hash of their own contents, so
the same bytes are stored once however many experiments refer to them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..core.contracts import (
    ComplexityVector,
    ControlRole,
    Evidence,
    ExperimentResult,
    ExperimentStatus,
    HypothesisStatus,
    ModelConfig,
    ProbeResult,
)
from ..ports import ArtifactRef

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    fingerprint     TEXT NOT NULL,
    hypothesis_id   TEXT,
    parent_id       TEXT,
    slot_id         TEXT,
    status          TEXT NOT NULL,
    runtime_s       REAL NOT NULL,
    token_cost      INTEGER NOT NULL,
    guard_failure   TEXT,
    payload         TEXT NOT NULL,
    position        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_fingerprint ON experiments (fingerprint);
CREATE INDEX IF NOT EXISTS experiments_position ON experiments (position);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id     TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    hypothesis_id   TEXT,
    control_id      TEXT NOT NULL,
    control_role    TEXT NOT NULL,
    status          TEXT NOT NULL,
    mean_delta      REAL NOT NULL,
    payload         TEXT NOT NULL,
    position        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_experiment ON evidence (experiment_id);

CREATE TABLE IF NOT EXISTS probes (
    probe_id        TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    payload         TEXT NOT NULL,
    position        INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id     TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    bytes_written   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);
"""


class StoreError(RuntimeError):
    """Raised when a write would overwrite something already recorded."""


def _dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def experiment_to_json(result: ExperimentResult) -> str:
    """Serialise an experiment. fold_scores keeps every fold, never the mean."""
    return _dumps(
        {
            "experiment_id": result.experiment_id,
            "fingerprint": result.fingerprint,
            "hypothesis_id": result.hypothesis_id,
            "parent_id": result.parent_id,
            "slot_id": result.slot_id,
            "model": {
                "family": result.model.family,
                "params": dict(result.model.params),
                "n_threads": result.model.n_threads,
            },
            "fold_scores": {k: list(v) for k, v in result.fold_scores.items()},
            "complexity": {
                "n_features": result.complexity.n_features,
                "n_cells": result.complexity.n_cells,
                "pipeline_depth": result.complexity.pipeline_depth,
                "fit_seconds": result.complexity.fit_seconds,
                "ast_nodes": result.complexity.ast_nodes,
            },
            "runtime_s": result.runtime_s,
            "token_cost": result.token_cost,
            "status": result.status.value,
            "artifact_ids": dict(result.artifact_ids),
            "guard_failure": result.guard_failure,
        }
    )


def experiment_from_json(blob: str) -> ExperimentResult:
    """Rebuild an experiment. Round-trip fidelity is what makes replay meaningful."""
    raw = json.loads(blob)
    return ExperimentResult(
        experiment_id=raw["experiment_id"],
        fingerprint=raw["fingerprint"],
        hypothesis_id=raw["hypothesis_id"],
        parent_id=raw["parent_id"],
        slot_id=raw["slot_id"],
        model=ModelConfig(
            family=raw["model"]["family"],
            params=raw["model"]["params"],
            n_threads=raw["model"]["n_threads"],
        ),
        fold_scores={k: tuple(v) for k, v in raw["fold_scores"].items()},
        complexity=ComplexityVector(**raw["complexity"]),
        runtime_s=raw["runtime_s"],
        token_cost=raw["token_cost"],
        status=ExperimentStatus(raw["status"]),
        artifact_ids=raw["artifact_ids"],
        guard_failure=raw["guard_failure"],
    )


def evidence_to_json(evidence: Evidence) -> str:
    return _dumps(
        {
            "evidence_id": evidence.evidence_id,
            "experiment_id": evidence.experiment_id,
            "hypothesis_id": evidence.hypothesis_id,
            "control_experiment_id": evidence.control_experiment_id,
            "control_fingerprint": evidence.control_fingerprint,
            "control_role": evidence.control_role.value,
            "metric": evidence.metric,
            "mean_delta": evidence.mean_delta,
            "ci_low": evidence.ci_low,
            "ci_high": evidence.ci_high,
            "delta_min": evidence.delta_min,
            "status": evidence.status.value,
            "underpowered": evidence.underpowered,
            "gate_violations": list(evidence.gate_violations),
            "reason": evidence.reason,
        }
    )


def evidence_from_json(blob: str) -> Evidence:
    raw = json.loads(blob)
    return Evidence(
        evidence_id=raw["evidence_id"],
        experiment_id=raw["experiment_id"],
        hypothesis_id=raw["hypothesis_id"],
        control_experiment_id=raw["control_experiment_id"],
        control_fingerprint=raw["control_fingerprint"],
        control_role=ControlRole(raw["control_role"]),
        metric=raw["metric"],
        mean_delta=raw["mean_delta"],
        ci_low=raw["ci_low"],
        ci_high=raw["ci_high"],
        delta_min=raw["delta_min"],
        status=HypothesisStatus(raw["status"]),
        underpowered=raw["underpowered"],
        gate_violations=tuple(raw["gate_violations"]),
        reason=raw["reason"],
    )


@dataclass
class SqliteStore:
    """StorePort backed by one SQLite file and a content-addressed object directory.

    A session owns a directory. Keeping sessions apart means a benchmark sweep can
    run many at once without any of them seeing another's experiments.
    """

    root: Path
    session_id: str = "default"
    _connection: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.session_dir / "registry.db")
        # WAL lets a reader, such as a dashboard, run without blocking the writer.
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    @property
    def session_dir(self) -> Path:
        return Path(self.root) / self.session_id

    @property
    def objects_dir(self) -> Path:
        return self.session_dir / "objects"

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _next_position(self, table: str) -> int:
        """Insertion order, which is what the promotion ladder is replayed against."""
        row = self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0])

    # ------------------------------------------------------------------ artifacts

    def _object_path(self, artifact_id: str) -> Path:
        return self.objects_dir / artifact_id[:2] / artifact_id[2:]

    def put_artifact(self, kind: str, payload: bytes) -> ArtifactRef:
        """Store bytes under their own hash. Identical content is written once."""
        artifact_id = hashlib.sha256(payload).hexdigest()[:32]
        path = self._object_path(artifact_id)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self._connection.execute(
                "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?)",
                (artifact_id, kind, len(payload)),
            )
            self._connection.commit()
        return ArtifactRef(artifact_id, kind, len(payload))

    def get_artifact(self, artifact_id: str) -> bytes:
        path = self._object_path(artifact_id)
        if not path.exists():
            raise KeyError(artifact_id)
        return path.read_bytes()

    def has_artifact(self, artifact_id: str) -> bool:
        return self._object_path(artifact_id).exists()

    # ---------------------------------------------------------------- experiments

    def record_experiment(self, result: ExperimentResult) -> None:
        """Insert once. A second write under the same id is an error, not an update."""
        try:
            self._connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.experiment_id,
                    result.fingerprint,
                    result.hypothesis_id,
                    result.parent_id,
                    result.slot_id,
                    result.status.value,
                    result.runtime_s,
                    result.token_cost,
                    result.guard_failure,
                    experiment_to_json(result),
                    self._next_position("experiments"),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StoreError(
                f"experiment {result.experiment_id} is already recorded; "
                "the store is append-only"
            ) from exc
        self._connection.commit()

    def experiments(self) -> tuple[ExperimentResult, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM experiments ORDER BY position"
        ).fetchall()
        return tuple(experiment_from_json(row[0]) for row in rows)

    def fingerprints(self) -> frozenset[str]:
        rows = self._connection.execute("SELECT DISTINCT fingerprint FROM experiments")
        return frozenset(row[0] for row in rows)

    def find_by_fingerprint(self, fingerprint: str) -> ExperimentResult | None:
        row = self._connection.execute(
            "SELECT payload FROM experiments WHERE fingerprint = ? ORDER BY position LIMIT 1",
            (fingerprint,),
        ).fetchone()
        return experiment_from_json(row[0]) if row else None

    # ------------------------------------------------------------------- evidence

    def record_evidence(self, evidence: Evidence) -> None:
        try:
            self._connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    evidence.evidence_id,
                    evidence.experiment_id,
                    evidence.hypothesis_id,
                    evidence.control_experiment_id,
                    evidence.control_role.value,
                    evidence.status.value,
                    evidence.mean_delta,
                    evidence_to_json(evidence),
                    self._next_position("evidence"),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise StoreError(
                f"evidence {evidence.evidence_id} is already recorded; "
                "the store is append-only"
            ) from exc
        self._connection.commit()

    def evidences(self) -> tuple[Evidence, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM evidence ORDER BY position"
        ).fetchall()
        return tuple(evidence_from_json(row[0]) for row in rows)

    # --------------------------------------------------------------------- probes

    def record_probe(self, probe: ProbeResult) -> None:
        """Probes are part of the reasoning trail, not scratch work.

        Recording them is what lets a report say which observation a hypothesis
        followed from, rather than asserting that one did.
        """
        payload = _dumps(
            {
                "probe_id": probe.probe_id,
                "kind": probe.kind,
                "params": dict(probe.params),
                "payload": dict(probe.payload),
            }
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO probes VALUES (?, ?, ?, ?)",
            (probe.probe_id, probe.kind, payload, self._next_position("probes")),
        )
        self._connection.commit()

    def probes(self) -> tuple[ProbeResult, ...]:
        rows = self._connection.execute(
            "SELECT payload FROM probes ORDER BY position"
        ).fetchall()
        out = []
        for (blob,) in rows:
            raw = json.loads(blob)
            out.append(
                ProbeResult(
                    probe_id=raw["probe_id"],
                    kind=raw["kind"],
                    params=raw["params"],
                    payload=raw["payload"],
                )
            )
        return tuple(out)

    # ----------------------------------------------------------------------- meta

    def set_meta(self, key: str, value: str) -> None:
        """Session-level facts such as the contract fingerprint and the environment."""
        self._connection.execute(
            "INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value)
        )
        self._connection.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def meta(self) -> Mapping[str, str]:
        return dict(self._connection.execute("SELECT key, value FROM meta").fetchall())

    # ---------------------------------------------------------------------- stats

    def summary(self) -> Mapping[str, int]:
        """Row counts, for a run header or a health check."""
        return {
            table: int(
                self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in ("experiments", "evidence", "probes", "artifacts")
        }


def open_store(root: Path | str, session_id: str = "default") -> SqliteStore:
    """Open or create a session store."""
    return SqliteStore(Path(root), session_id)


def list_sessions(root: Path | str) -> tuple[str, ...]:
    """Session ids present under a root, for resuming or comparing runs."""
    base = Path(root)
    if not base.exists():
        return ()
    return tuple(
        sorted(p.name for p in base.iterdir() if (p / "registry.db").exists())
    )