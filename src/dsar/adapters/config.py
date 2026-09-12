"""Load a research configuration from YAML. The only place that reads config files.

Parsing and validation stay in core.dossier; this module does the I/O and nothing
else, so a configuration can be built in memory for tests without touching disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from ..core.dossier import Dossier, DossierError
from ..core.state import SessionBudget

DEFAULT_K = 10
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42
DEFAULT_DELTA_PRACTICAL = 0.01


@dataclass(frozen=True, slots=True)
class Evaluation:
    """How results are judged. delta_practical is a floor, never a ceiling."""

    primary_metric: str | None = None
    delta_practical: float = DEFAULT_DELTA_PRACTICAL
    secondary_gates: Mapping[str, float] = field(default_factory=dict)
    k: int = DEFAULT_K
    repeats: int = DEFAULT_REPEATS


@dataclass(frozen=True, slots=True)
class Constraints:
    """Ceilings on what a proposed pipeline may do."""

    max_features: int = 200
    max_pipeline_depth: int = 5
    max_cells: int = 5


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """One session's inputs: what to model, what is known, and what is allowed."""

    dataset: Path
    target: str
    dossier: Dossier = field(default_factory=Dossier)
    evaluation: Evaluation = field(default_factory=Evaluation)
    constraints: Constraints = field(default_factory=Constraints)
    budget: SessionBudget = field(
        default_factory=lambda: SessionBudget(experiments=50, tokens=1_500_000, seconds=5400.0)
    )
    approval_mode: Literal["auto", "interactive", "strict"] = "interactive"
    seed: int = DEFAULT_SEED


def _dossier_from(raw: Mapping[str, Any]) -> Dossier:
    """Build the dossier, rejecting keys that look like typos of real ones."""
    known = {f for f in Dossier.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise DossierError(f"unknown dossier keys: {unknown}; expected one of {sorted(known)}")

    return Dossier(
        objective=str(raw.get("objective", "")),
        prediction_time=str(raw.get("prediction_time", "")),
        grain=str(raw.get("grain", "")),
        notes=str(raw.get("notes", "")),
        columns={str(k): str(v) for k, v in (raw.get("columns") or {}).items()},
        exclude=tuple(raw.get("exclude") or ()),
        protected=tuple(raw.get("protected") or ()),
        protected_policy=raw.get("protected_policy", "audit"),
        group_column=raw.get("group_column"),
        time_column=raw.get("time_column"),
        grouping_confirmed=bool(raw.get("grouping_confirmed", False)),
        temporal_confirmed=bool(raw.get("temporal_confirmed", False)),
    )


def config_from_mapping(raw: Mapping[str, Any], base: Path = Path(".")) -> ResearchConfig:
    """Build a configuration from an already-parsed mapping."""
    for required in ("dataset", "target"):
        if required not in raw:
            raise DossierError(f"config is missing the required key: {required}")

    evaluation_raw = raw.get("evaluation") or {}
    constraints_raw = raw.get("constraints") or {}
    budget_raw = raw.get("budget") or {}

    dataset = Path(str(raw["dataset"]))
    if not dataset.is_absolute():
        dataset = base / dataset

    return ResearchConfig(
        dataset=dataset,
        target=str(raw["target"]),
        dossier=_dossier_from(raw.get("dossier") or {}),
        evaluation=Evaluation(
            primary_metric=evaluation_raw.get("primary_metric"),
            delta_practical=float(
                evaluation_raw.get("delta_practical", DEFAULT_DELTA_PRACTICAL)
            ),
            secondary_gates={
                str(k): float(v)
                for k, v in (evaluation_raw.get("secondary_gates") or {}).items()
            },
            k=int(evaluation_raw.get("k", DEFAULT_K)),
            repeats=int(evaluation_raw.get("repeats", DEFAULT_REPEATS)),
        ),
        constraints=Constraints(
            max_features=int(constraints_raw.get("max_features", 200)),
            max_pipeline_depth=int(constraints_raw.get("max_pipeline_depth", 5)),
            max_cells=int(constraints_raw.get("max_cells", 5)),
        ),
        budget=SessionBudget(
            experiments=int(budget_raw.get("max_experiments", 50)),
            tokens=int(budget_raw.get("max_tokens", 1_500_000)),
            seconds=float(budget_raw.get("max_wall_time_min", 90)) * 60.0,
        ),
        approval_mode=raw.get("governance", {}).get("approval_mode", "interactive"),
        seed=int(raw.get("seed", DEFAULT_SEED)),
    )


def load_research_config(path: Path) -> ResearchConfig:
    """Read a YAML configuration. Paths inside it resolve against the repository root."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, Mapping):
        raise DossierError(f"{path} does not contain a mapping at the top level")
    return config_from_mapping(raw, base=path.parent.parent)