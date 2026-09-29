"""Parsing cases the curated benchmarks never produce.

Every dataset used during development is a published benchmark: UTF-8, dot decimals,
ISO dates, one missing marker. Business data is not, and the column classifier is
where that difference lands. Each case below isolates one way a real file departs
from a clean one, so the question stops being whether the system is robust and
becomes how many of ten it survives.

Some of these are expected to fail today. A test that records a known limitation is
worth more than one that quietly avoids it, so the failures are marked rather than
removed, and the marker comes off as each is fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dsar.stages.eda import (
    classify_column,
    coerce_datetime,
    coerce_numeric,
    has_case_variants,
    looks_decorated,
    run_eda,
)

N = 800


def frame_with(column: str, values, seed: int = 0) -> pd.DataFrame:
    """A small clean frame with one awkward column bolted on."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {"num_0": rng.normal(size=N), "num_1": rng.normal(size=N)}
    )
    frame[column] = values
    frame["y"] = (frame["num_0"] + rng.normal(size=N) > 0).astype(int)
    return frame


def test_padded_values_do_not_become_distinct_categories() -> None:
    """Trailing spaces would otherwise triple the cardinality of a clean column."""
    rng = np.random.default_rng(1)
    cities = rng.choice(["Jakarta", "Jakarta ", " Jakarta", "Bandung"], N)
    spec = classify_column("city", pd.Series(cities))
    assert spec.cardinality == 2


def test_case_variants_are_reported_rather_than_folded() -> None:
    """Folding them would be a guess: case separates product codes as often as not."""
    rng = np.random.default_rng(2)
    flags = pd.Series(rng.choice(["Ya", "ya", "YA", "Tidak", "tidak"], N))
    assert has_case_variants(flags)

    frame = frame_with("subscribed", flags)
    artifact, _ = run_eda(frame=frame, target_column="y")
    assert any("letter case" in w for w in artifact.warnings)


def test_several_missing_markers_are_all_recognised() -> None:
    """One file routinely carries N/A, a dash and an empty string at once."""
    rng = np.random.default_rng(3)
    values = rng.normal(size=N).round(2).astype(str)
    for marker, position in (("N/A", 0), ("-", 1), ("", 2), ("null", 3), ("NULL", 4)):
        values[position] = marker

    spec = classify_column("amount", pd.Series(values))
    assert spec.role == "numeric"
    assert spec.missing_rate == pytest.approx(5 / N, abs=1e-6)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([f"{v:,.2f}" for v in np.random.default_rng(4).gamma(2, 50000, N)],
                     id="thousand_separator"),
        pytest.param([f"{v:.2f}".replace(".", ",")
                      for v in np.random.default_rng(5).gamma(2, 500, N)],
                     id="comma_decimal"),
        pytest.param([f"Rp{v:,.0f}" for v in np.random.default_rng(6).gamma(2, 100000, N)],
                     id="currency_prefix"),
        pytest.param([f"{v:.1f}%" for v in np.random.default_rng(7).random(N) * 100],
                     id="percent_suffix"),
    ],
)
def test_decorated_numbers_are_flagged_not_silently_parsed(values: list[str]) -> None:
    """The separator convention is a fact about the source, not about the characters.

    "1,234" is a thousand under one convention and one point two under another, and
    stripping it either way would be a guess dressed as parsing. The deterministic
    layer reports what it sees and leaves the reading to a stage that can weigh it.
    """
    column = pd.Series(values)
    assert coerce_numeric(column) is None
    assert looks_decorated(column)

    artifact, _ = run_eda(frame=frame_with("amount", column), target_column="y")
    assert any("symbols or separators" in w for w in artifact.warnings)


def test_day_first_dates_are_detected() -> None:
    stamps = pd.date_range("2024-01-01", periods=N, freq="D")
    written = stamps.strftime("%d/%m/%Y")
    assert coerce_datetime(pd.Series(written)) is not None


def test_free_text_stays_categorical_and_is_not_coerced() -> None:
    """A notes column has near-unique values but is neither a key nor a number."""
    rng = np.random.default_rng(8)
    notes = [f"customer called on {rng.integers(1, 28)} about billing" for _ in range(N)]
    spec = classify_column("notes", pd.Series(notes))
    assert spec.role in ("categorical", "id")
    assert coerce_numeric(pd.Series(notes)) is None


def test_duplicate_column_names_do_not_silently_merge() -> None:
    """pandas keeps both, and selecting one returns a frame rather than a series."""
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(N, 3)), columns=["a", "b", "a"])
    frame["y"] = (frame.iloc[:, 0] > 0).astype(int)

    with pytest.raises(Exception):
        run_eda(frame=frame, target_column="y")


def test_a_class_too_rare_to_stratify_is_handled() -> None:
    """Fewer members than folds makes stratification impossible."""
    rng = np.random.default_rng(10)
    frame = pd.DataFrame({"num_0": rng.normal(size=N), "num_1": rng.normal(size=N)})
    labels = np.zeros(N, dtype=int)
    labels[:3] = 1
    frame["y"] = labels

    artifact, contract = run_eda(frame=frame, target_column="y", k=10, repeats=1)
    assert contract.power.underpowered
    assert artifact.warnings


def test_an_all_missing_column_is_dropped_rather_than_profiled() -> None:
    frame = frame_with("empty", np.nan)
    artifact, contract = run_eda(frame=frame, target_column="y")
    assert "empty" in artifact.integrity.empty_columns
    assert "empty" not in contract.schema


def test_a_constant_column_is_dropped() -> None:
    frame = frame_with("constant", 7)
    _, contract = run_eda(frame=frame, target_column="y")
    assert "constant" not in contract.schema
