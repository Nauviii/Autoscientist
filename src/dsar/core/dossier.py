"""Domain knowledge the data cannot supply. Pure: validation and merging only.

Three of the four questions that decide whether a session is valid have no
statistical answer. Whether a column will exist at prediction time, whether a
repeated key groups rows that must stay together, whether an attribute is protected
are all facts about the world, not about the table. A column filled in after the
outcome looks perfectly ordinary; a sampling weight looks like a feature.

So they are stated here rather than guessed. Everything is optional: with no
dossier the pipeline falls back to its heuristics exactly as before, which keeps
the offline path and the test suite intact.

The text fields are not read by any calculation yet. They exist so that the
proposal stage, once it arrives, starts from what the user knows rather than from
the column names alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Literal, Mapping

from .contracts import DataContract, ValidationSpec


class DossierError(ValueError):
    """Raised when the dossier names a column the dataset does not have.

    A silent miss here is the worst failure mode available: a typo in an exclusion
    means the column stays in, and nothing anywhere reports it.
    """


@dataclass(frozen=True, slots=True)
class Dossier:
    """What the user knows that the data cannot show."""

    objective: str = ""
    prediction_time: str = ""
    grain: str = ""
    notes: str = ""
    columns: Mapping[str, str] = field(default_factory=dict)
    exclude: tuple[str, ...] = ()
    protected: tuple[str, ...] = ()
    protected_policy: Literal["exclude", "audit", "allow"] = "audit"
    group_column: str | None = None
    time_column: str | None = None
    grouping_confirmed: bool = False
    temporal_confirmed: bool = False

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.objective, self.prediction_time, self.grain, self.notes,
             self.columns, self.exclude, self.protected,
             self.group_column, self.time_column)
        )

    def named_columns(self) -> tuple[str, ...]:
        """Every column the dossier refers to, for validation against the dataset."""
        names = set(self.columns) | set(self.exclude) | set(self.protected)
        for column in (self.group_column, self.time_column):
            if column:
                names.add(column)
        return tuple(sorted(names))

    def validate(self, available: Iterable[str]) -> None:
        """Reject references to columns the dataset does not contain."""
        known = set(available)
        missing = [name for name in self.named_columns() if name not in known]
        if missing:
            raise DossierError(
                f"dossier names columns absent from the dataset: {missing}. "
                "Check for typos; an unmatched exclusion would silently do nothing."
            )


def resolve_validation(contract: DataContract, dossier: Dossier) -> ValidationSpec:
    """Apply the user's answers to the questions EDA could only guess at.

    A confirmed grouping key switches the scheme; an explicit denial settles it the
    other way. Both are better than the heuristic, which cannot tell an identifier
    from a discretised measurement.
    """
    validation = contract.validation

    if dossier.time_column:
        return replace(
            validation, kind="time_series", time_column=dossier.time_column, group_column=None
        )
    if dossier.group_column:
        return replace(
            validation, kind="group_kfold", group_column=dossier.group_column, time_column=None
        )
    if dossier.temporal_confirmed and validation.kind == "time_series":
        # The user says the data is not ordered in time after all.
        fallback = "kfold" if contract.task == "regression" else "stratified_kfold"
        return replace(validation, kind=fallback, time_column=None)  # type: ignore[arg-type]
    return validation


def apply_dossier(contract: DataContract, dossier: Dossier) -> DataContract:
    """Merge stated knowledge into a contract EDA built from the data alone.

    Exclusions are unioned rather than replaced: the screen may have blocked a
    column the user did not think to mention, and dropping that would undo a
    finding the user never saw.
    """
    if dossier.is_empty:
        return contract

    # The target belongs to the dataset even though it is not a feature, and
    # describing what it means is exactly the kind of context the proposal stage
    # will need.
    dossier.validate(
        set(contract.schema) | set(contract.excluded_columns) | {contract.target}
    )

    excluded = tuple(sorted(set(contract.excluded_columns) | set(dossier.exclude)))
    schema = {name: spec for name, spec in contract.schema.items() if name not in excluded}
    if not schema:
        raise DossierError("every column was excluded; nothing is left to model")

    protected = tuple(sorted(set(contract.protected_columns) | set(dossier.protected)))
    if dossier.protected_policy == "exclude":
        excluded = tuple(sorted(set(excluded) | set(protected)))
        schema = {name: spec for name, spec in schema.items() if name not in excluded}

    return replace(
        contract,
        schema=schema,
        excluded_columns=excluded,
        protected_columns=protected,
        protected_policy=dossier.protected_policy,
        validation=resolve_validation(contract, dossier),
    )


def open_questions(contract: DataContract, dossier: Dossier) -> tuple[str, ...]:
    """What the dossier left unanswered that a human still needs to settle.

    Surfacing these is the point of the gate: each one is cheap to answer and
    expensive to get wrong, because every later experiment inherits it.
    """
    questions: list[str] = []

    if not dossier.prediction_time:
        questions.append(
            "Which columns are unavailable at prediction time? No test can detect this."
        )
    undescribed = [name for name in contract.schema if name not in dossier.columns]
    if undescribed:
        preview = ", ".join(undescribed[:5])
        suffix = f" and {len(undescribed) - 5} more" if len(undescribed) > 5 else ""
        questions.append(f"Columns with no description: {preview}{suffix}.")
    if not dossier.group_column and not dossier.grouping_confirmed:
        questions.append("Is there a key whose rows must stay in the same fold?")
    if not dossier.protected and dossier.protected_policy != "allow":
        questions.append("Are any attributes protected and in need of a subgroup audit?")
    return tuple(questions)
