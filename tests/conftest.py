"""Shared paths and fixtures. Every test here runs offline, with no API key."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src" / "dsar"
CORE_DIR = SRC_DIR / "core"
PROMPTS_DIR = SRC_DIR / "prompts"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
LEAKY_CELLS_DIR = FIXTURES_DIR / "leaky_cells"


def make_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Generic frame whose column names carry no dataset identity."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "num_0": rng.normal(size=n),
            "num_1": rng.normal(size=n),
            "cat_0": rng.integers(0, 4, size=n).astype(str),
        }
    )
    # Spread the gaps across the frame: clustering them at the start would leave
    # later folds without any, and a guard cannot observe what never occurs.
    frame.loc[frame.index[::7], "num_1"] = np.nan
    return frame


def instantiate_cell(source: str, name: str = "<cell>") -> Any:
    """Instantiate a FeatureStep from source. Test fixtures are trusted input."""
    namespace: dict[str, Any] = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace["FeatureStep"]()


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def target(frame: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.integers(0, 2, size=len(frame)), index=frame.index)


@pytest.fixture
def load_cell():
    """Load a leaky-cell fixture by filename."""

    def _load(filename: str) -> Any:
        path = LEAKY_CELLS_DIR / filename
        return instantiate_cell(path.read_text(), filename)

    return _load