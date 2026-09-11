"""Architectural invariants. Fast, offline, no API key, no dataset required."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.conftest import CORE_DIR as CORE
from tests.conftest import PROMPTS_DIR as PROMPTS
from tests.conftest import SRC_DIR as SRC

SIDE_EFFECT_MODULES = ("adapters", "stages", "anthropic", "sqlite3", "requests", "httpx", "openai")

# Column names from the development dataset. Their presence anywhere in src/ means
# a decision is hardcoded instead of flowing from DataContract.schema, which would
# silently break the moment a different tabular dataset is used.
DEV_DATASET_LITERALS = {
    "customerID",
    "TotalCharges",
    "MonthlyCharges",
    "SeniorCitizen",
    "PaperlessBilling",
    "PaymentMethod",
    "InternetService",
    "Churn",
    "telco",
}


def _imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _string_constants(path: Path) -> set[str]:
    return {
        node.value
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")), ids=lambda p: p.name)
def test_core_stays_pure(path: Path) -> None:
    """core must not reach adapters, network, storage or LLM clients."""
    for name in _imported_modules(path):
        for forbidden in SIDE_EFFECT_MODULES:
            assert forbidden not in name, f"{path.name} imports {name}"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC)))
def test_no_dev_dataset_column_names(path: Path) -> None:
    """Behaviour must derive from the contract, never from hardcoded column names."""
    found = {s for s in _string_constants(path) if s in DEV_DATASET_LITERALS}
    assert not found, f"{path.relative_to(SRC)} hardcodes dataset literals: {sorted(found)}"


@pytest.mark.parametrize("path", sorted(PROMPTS.glob("*.md")), ids=lambda p: p.name)
def test_prompts_are_dataset_agnostic(path: Path) -> None:
    """Few-shot examples must not bias the agent toward the development dataset."""
    text = path.read_text()
    found = sorted(lit for lit in DEV_DATASET_LITERALS if lit in text)
    assert not found, f"{path.name} mentions dataset literals: {found}"


def test_config_carries_no_metric_constants() -> None:
    """Noise assumptions belong in core.noise, expressed as functions of shape."""
    source = (CORE / "statistics.py").read_text()
    for banned in ("_PR_AUC_INFLATION", "1.7"):
        assert banned not in source, f"statistics.py still hardcodes {banned}"