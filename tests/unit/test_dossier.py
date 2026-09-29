"""The dossier carries the facts no statistic can reach, so it is tested directly."""

from __future__ import annotations

import pytest

from dsar.core.contracts import (
    ColumnSpec,
    DataContract,
    TargetDistribution,
    ValidationSpec,
)
from dsar.core.dossier import (
    Dossier,
    DossierError,
    apply_dossier,
    open_questions,
    resolve_validation,
    settles_grouping,
)
from dsar.core.statistics import build_power_profile


def column(name: str, role: str = "numeric") -> ColumnSpec:
    return ColumnSpec(
        name=name,
        dtype="float64",
        role=role,  # type: ignore[arg-type]
        missing_rate=0.0,
        cardinality=50,
        unique_ratio=0.02,
    )


def make_contract(
    task: str = "binary",
    kind: str = "stratified_kfold",
    columns: tuple[str, ...] = ("num_0", "num_1", "cat_0"),
    excluded: tuple[str, ...] = (),
) -> DataContract:
    validation = ValidationSpec(kind=kind, k=5, repeats=2, seed=42)  # type: ignore[arg-type]
    distribution = TargetDistribution(
        task=task,  # type: ignore[arg-type]
        n_rows=2000,
        n_positive=500 if task == "binary" else None,
    )
    family = "pr_auc" if task == "binary" else "r2"
    power = build_power_profile(
        n_rows=2000,
        n_pos=distribution.n_positive,
        validation=validation,
        metric_family=family,  # type: ignore[arg-type]
        expected_score=0.85 if task == "binary" else 0.6,
        expected_auc=0.85 if task == "binary" else None,
        delta_practical=0.01,
    )
    return DataContract(
        dataset_hash="test",
        task=task,  # type: ignore[arg-type]
        target="y",
        target_distribution=distribution,
        schema={name: column(name) for name in columns},
        validation=validation,
        primary_metric=family,
        metric_family=family,
        secondary_metrics=(),
        secondary_gates={},
        delta_practical=0.01,
        power=power,
        leakage_flags=(),
        excluded_columns=excluded,
    )


def test_an_empty_dossier_changes_nothing() -> None:
    """The offline path has to keep working exactly as before."""
    contract = make_contract()
    assert Dossier().is_empty
    assert apply_dossier(contract, Dossier()) == contract


def test_text_alone_does_not_make_a_dossier_active() -> None:
    """Prose is for the proposal stage; it must not alter any decision."""
    contract = make_contract()
    described = Dossier(objective="predict churn", notes="see the data dictionary")
    assert not described.is_empty
    assert apply_dossier(contract, described).schema == contract.schema


def test_a_misspelled_column_is_rejected_loudly() -> None:
    """A silent miss is the worst outcome: the column stays in and nobody is told."""
    contract = make_contract()
    with pytest.raises(DossierError) as caught:
        apply_dossier(contract, Dossier(exclude=("num_zero",)))
    assert "num_zero" in str(caught.value)


def test_the_target_may_be_described() -> None:
    """Explaining what the target means is exactly the context proposals need."""
    contract = make_contract()
    apply_dossier(contract, Dossier(columns={"y": "whether the customer left"}))


def test_stated_exclusions_join_the_screened_ones() -> None:
    """Replacing them would undo a finding the user never saw."""
    contract = make_contract(excluded=("record_id",))
    merged = apply_dossier(contract, Dossier(exclude=("num_1",)))

    assert set(merged.excluded_columns) == {"record_id", "num_1"}
    assert "num_1" not in merged.schema
    assert "num_0" in merged.schema


def test_excluding_everything_is_an_error_not_an_empty_run() -> None:
    contract = make_contract()
    with pytest.raises(DossierError):
        apply_dossier(contract, Dossier(exclude=("num_0", "num_1", "cat_0")))


def test_protected_columns_are_audited_by_default() -> None:
    contract = make_contract()
    merged = apply_dossier(contract, Dossier(protected=("cat_0",)))

    assert merged.protected_columns == ("cat_0",)
    assert merged.protected_policy == "audit"
    assert "cat_0" in merged.schema


def test_the_exclude_policy_removes_protected_columns() -> None:
    contract = make_contract()
    merged = apply_dossier(
        contract, Dossier(protected=("cat_0",), protected_policy="exclude")
    )
    assert "cat_0" not in merged.schema
    assert "cat_0" in merged.excluded_columns


def test_naming_a_time_column_switches_to_forward_chaining() -> None:
    contract = make_contract()
    resolved = resolve_validation(contract, Dossier(time_column="event_day"))

    assert resolved.kind == "time_series"
    assert resolved.time_column == "event_day"
    assert resolved.group_column is None


def test_naming_a_group_column_switches_to_grouped_folds() -> None:
    contract = make_contract()
    resolved = resolve_validation(contract, Dossier(group_column="entity"))

    assert resolved.kind == "group_kfold"
    assert resolved.group_column == "entity"
    assert resolved.time_column is None


@pytest.mark.parametrize(
    "task,fallback", [("binary", "stratified_kfold"), ("regression", "kfold")]
)
def test_confirming_without_naming_is_the_opposite_answer(task: str, fallback: str) -> None:
    """Setting the flag with no column means the user looked and found nothing."""
    contract = make_contract(task=task, kind="time_series")
    resolved = resolve_validation(contract, Dossier(temporal_confirmed=True))

    assert resolved.kind == fallback
    assert resolved.time_column is None


def test_denying_grouping_falls_back_the_same_way() -> None:
    contract = make_contract(kind="group_kfold")
    resolved = resolve_validation(contract, Dossier(grouping_confirmed=True))

    assert resolved.kind == "stratified_kfold"
    assert resolved.group_column is None


def test_a_named_column_outranks_the_confirmation_flag() -> None:
    """Naming one is a stronger statement than declaring the question settled."""
    contract = make_contract(kind="time_series")
    resolved = resolve_validation(
        contract, Dossier(time_column="event_day", temporal_confirmed=True)
    )
    assert resolved.kind == "time_series"


@pytest.mark.parametrize(
    "dossier,settled",
    [
        (None, False),
        (Dossier(), False),
        (Dossier(grouping_confirmed=True), True),
        (Dossier(group_column="entity"), True),
        (Dossier(temporal_confirmed=True), False),
    ],
)
def test_settles_grouping_covers_both_kinds_of_answer(dossier, settled: bool) -> None:
    assert settles_grouping(dossier) is settled


def test_open_questions_name_what_is_still_unanswered() -> None:
    """Each one is cheap to answer and expensive to get wrong."""
    contract = make_contract()
    assert open_questions(contract, Dossier()) != ()


def test_a_complete_dossier_leaves_fewer_questions() -> None:
    contract = make_contract()
    complete = Dossier(
        objective="predict churn",
        prediction_time="start of the billing month",
        grain="one row per customer",
        grouping_confirmed=True,
        temporal_confirmed=True,
        columns={"num_0": "a", "num_1": "b", "cat_0": "c"},
    )
    assert len(open_questions(contract, complete)) < len(open_questions(contract, Dossier()))
