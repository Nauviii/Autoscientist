"""Render the research state as bounded text for the model. Pure: no I/O, no LLM.

Two rules shape everything here.

Ordering. Blocks run from least to most volatile, because a cache prefix ends at
the first byte that changes. Putting the turn-by-turn state ahead of the data
profile would make every block behind it uncacheable.

Density. Tables, never prose. A rendered table of twenty columns costs a fraction
of the same content written out, and the model reads it more reliably. Nothing
here is generated; every line is a template filled with numbers already computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .contracts import DataContract, Finding, ResearchState, SlotStatus

# Average characters per token for English prose and tables. Only used for budget
# guards, so a rough figure is enough; the adapter reports exact counts.
CHARS_PER_TOKEN = 4

MAX_SCHEMA_ROWS = 25
MAX_RECENT = 5
MAX_FINDINGS = 12


@dataclass(frozen=True, slots=True)
class ContextBlock:
    """One prompt block. cacheable marks content that is stable within a session."""

    name: str
    text: str
    cacheable: bool

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass(frozen=True, slots=True)
class Context:
    """Ordered blocks ready for the adapter to turn into an API request."""

    blocks: tuple[ContextBlock, ...]

    @property
    def cache_breakpoint(self) -> int:
        """Index after the last cacheable block; everything before it is stable."""
        for index, block in enumerate(self.blocks):
            if not block.cacheable:
                return index
        return len(self.blocks)

    @property
    def cacheable_tokens(self) -> int:
        return sum(b.tokens for b in self.blocks[: self.cache_breakpoint])

    @property
    def total_tokens(self) -> int:
        return sum(b.tokens for b in self.blocks)

    def render(self) -> str:
        return "\n\n".join(block.text for block in self.blocks)


def estimate_tokens(text: str) -> int:
    """Rough token count used only for budget guards."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Fixed-width table. Deterministic given the same rows."""
    widths = [len(h) for h in headers]
    text_rows = [[str(cell) for cell in row] for row in rows]
    for row in text_rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()]
    lines += ["  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip() for row in text_rows]
    return "\n".join(lines)


def render_schema(contract: DataContract, max_rows: int = MAX_SCHEMA_ROWS) -> str:
    """Column profile, collapsed by role once listing every column gets expensive."""
    specs = sorted(contract.schema.values(), key=lambda s: s.name)
    if not specs:
        return "columns: none profiled"

    if len(specs) <= max_rows:
        rows = [
            [s.name, s.role, f"{s.missing_rate:.3f}", s.cardinality, f"{s.unique_ratio:.3f}"]
            for s in specs
        ]
        return "columns\n" + _table(["name", "role", "missing", "card", "unique"], rows)

    grouped: dict[str, list] = {}
    for spec in specs:
        grouped.setdefault(spec.role, []).append(spec)
    rows = [
        [
            role,
            len(items),
            f"{sum(s.missing_rate for s in items) / len(items):.3f}",
            f"{min(s.cardinality for s in items)}-{max(s.cardinality for s in items)}",
        ]
        for role, items in sorted(grouped.items())
    ]
    return (
        f"columns ({len(specs)} total, grouped by role)\n"
        + _table(["role", "count", "mean_missing", "cardinality"], rows)
    )


def render_contract(contract: DataContract) -> str:
    """Session-stable facts: task shape, protocol, thresholds, exclusions."""
    distribution = contract.target_distribution
    lines = [
        "data contract",
        f"task            {contract.task}",
        f"rows            {contract.n_rows}",
    ]
    if distribution.prevalence is not None:
        lines.append(
            f"positives       {distribution.n_positive} ({distribution.prevalence:.1%})"
        )
    lines += [
        f"primary metric  {contract.primary_metric}",
        f"validation      {contract.validation.kind} k={contract.validation.k} "
        f"repeats={contract.validation.repeats} seed={contract.validation.seed}",
        f"delta_min       {contract.delta_min:.4f}  "
        f"(practical {contract.delta_practical:.4f}, detectable {contract.power.mde:.4f})",
        f"selection rule  {contract.selection_rule}",
    ]
    if contract.secondary_gates:
        gates = ", ".join(f"{k} <= {v:.4f}" for k, v in sorted(contract.secondary_gates.items()))
        lines.append(f"gates           {gates}")
    if contract.excluded_columns:
        lines.append(f"excluded        {', '.join(sorted(contract.excluded_columns))}")
    if contract.protected_columns:
        lines.append(
            f"protected       {', '.join(sorted(contract.protected_columns))} "
            f"(policy: {contract.protected_policy})"
        )
    if contract.power.underpowered:
        lines.append("warning         effects below delta_min cannot be resolved on this dataset")
    return "\n".join(lines)


def render_slots(state: ResearchState) -> str:
    """The checklist, which is what routing is driven by."""
    rows = [
        [
            slot.slot_id,
            slot.status.value,
            len(slot.experiment_ids),
            slot.resolution or "-",
        ]
        for slot in state.slots
    ]
    header = f"decision checklist ({state.coverage:.0%} closed)"
    return header + "\n" + _table(["slot", "status", "tested", "resolution"], rows)


def render_progress(state: ResearchState) -> str:
    """Reference against incumbent, so the model sees both anchors at once."""
    delta = state.incumbent_score - state.reference_score
    return "\n".join(
        [
            "progress",
            f"reference   {state.reference_experiment_id}  {state.reference_score:.4f}",
            f"incumbent   {state.incumbent_experiment_id}  {state.incumbent_score:.4f}  "
            f"({delta:+.4f}, {state.incumbent_promotions} promotions)",
            f"budget      {state.experiments_used}/{state.experiments_budget} experiments, "
            f"{state.tokens_used:,}/{state.tokens_budget:,} tokens, "
            f"{state.seconds_used:.0f}/{state.seconds_budget:.0f} seconds",
        ]
    )


def render_recent(state: ResearchState, metric: str, limit: int = MAX_RECENT) -> str:
    """Only the last few experiments; the store holds the rest."""
    experiments = state.recent_experiments[-limit:]
    if not experiments:
        return "recent experiments: none"
    rows = []
    for experiment in experiments:
        score = f"{experiment.mean(metric):.4f}" if metric in experiment.fold_scores else "-"
        rows.append(
            [
                experiment.experiment_id,
                experiment.slot_id or "-",
                experiment.status.value,
                score,
                experiment.complexity.n_features,
                experiment.guard_failure or "-",
            ]
        )
    return "recent experiments\n" + _table(
        ["id", "slot", "status", metric, "features", "guard"], rows
    )


def render_findings(findings: Sequence[Finding], limit: int = MAX_FINDINGS) -> str:
    """Conclusions so far, so the model does not re-propose a settled decision."""
    if not findings:
        return "findings: none yet"
    rows = [
        [f.slot_id, f.status.value, f"{f.effect:+.4f}", f"[{f.ci_low:+.4f},{f.ci_high:+.4f}]"]
        for f in findings[:limit]
    ]
    return "findings\n" + _table(["slot", "status", "effect", "ci"], rows)


def render_task(
    action: str,
    slot_id: str | None,
    rationale: str,
    extra: Mapping[str, str] | None = None,
) -> str:
    """The narrow question for this turn. Narrow questions get better answers."""
    lines = ["task", f"action    {action}"]
    if slot_id:
        lines.append(f"slot      {slot_id}")
    lines.append(f"reason    {rationale}")
    for key, value in sorted((extra or {}).items()):
        lines.append(f"{key:<9} {value}")
    return "\n".join(lines)


def build_context(
    *,
    system_prompt: str,
    contract: DataContract,
    state: ResearchState,
    findings: Sequence[Finding] = (),
    action: str = "propose",
    slot_id: str | None = None,
    rationale: str = "",
    extra: Mapping[str, str] | None = None,
    token_budget: int | None = None,
) -> Context:
    """Assemble the prompt, ordered from stable to volatile for cache reuse."""
    blocks = [
        ContextBlock("system", system_prompt, cacheable=True),
        ContextBlock("contract", render_contract(contract), cacheable=True),
        ContextBlock("schema", render_schema(contract), cacheable=True),
        ContextBlock("progress", render_progress(state), cacheable=False),
        ContextBlock("checklist", render_slots(state), cacheable=False),
        ContextBlock("findings", render_findings(findings), cacheable=False),
        ContextBlock(
            "recent", render_recent(state, contract.primary_metric), cacheable=False
        ),
        ContextBlock("task", render_task(action, slot_id, rationale, extra), cacheable=False),
    ]
    context = Context(tuple(blocks))
    if token_budget is not None and context.total_tokens > token_budget:
        context = trim(context, token_budget)
    return context


def trim(context: Context, token_budget: int) -> Context:
    """Drop volatile blocks, least important first, until the budget is met.

    The system prompt, contract and task always survive: without them the model has
    no instruction, no thresholds and no question.
    """
    droppable = ("findings", "recent", "checklist", "progress")
    blocks = list(context.blocks)
    for name in droppable:
        if Context(tuple(blocks)).total_tokens <= token_budget:
            break
        blocks = [b for b in blocks if b.name != name]
    return Context(tuple(blocks))