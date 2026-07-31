"""
Shared workflow state (§16) — the system's spine.

Agents do not pass chat histories to each other. They read and write named channels on one
typed state object, and each channel declares who may write it. That is what §16 means by
"design a structured state model" rather than threading transcripts around.

**The parallel-safety rule.** LangGraph merges the partial states returned by concurrently
running nodes. A channel written by two branches in the same superstep needs a *reducer* saying
how to combine them; without one, LangGraph raises ``InvalidUpdateError`` at runtime. Every
channel below that a fan-out researcher can touch is therefore annotated additive, and every
channel a researcher must not touch is left plain — deliberately, so that an accidental write
from inside the fan-out fails loudly instead of silently dropping a sibling's work.

Read/write permissions are declared in :data:`FIELD_PERMISSIONS` and enforced in tests, not just
documented — §27 asks for the table, and a table that drifts from the code is worse than none.
"""
from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from app.schemas.common import AgentId, TaskStatus, WorkflowStatus
from app.schemas.evidence import Evidence
from app.schemas.handoffs import (
    AnalysisHandoff,
    CriticVerdict,
    FactCheckReport,
    ResearchHandoff,
)
from app.schemas.reports import ErrorRecord, FinalReport, TraceEvent
from app.schemas.request import Clarification, RequestBrief
from app.schemas.tasks import TaskPlan


# --------------------------------------------------------------------------- #
# Reducers
# --------------------------------------------------------------------------- #
def merge_status(left: dict[str, TaskStatus], right: dict[str, TaskStatus]) -> dict[str, TaskStatus]:
    """Merge per-task status maps from concurrent researchers.

    Each parallel branch reports only its own task, so keys never collide in practice. Last
    write wins on the rare overlap, which is correct here: a later status for the same task is
    a later observation of it.
    """
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def keep_last(left: Any, right: Any) -> Any:
    """Explicit last-write-wins for single-value channels a fan-out may touch."""
    return right if right is not None else left


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
class WorkflowState(TypedDict, total=False):
    """One research run.

    Channels are grouped by who writes them. ``Annotated[..., operator.add]`` marks a channel
    as append-only and therefore safe for the parallel researcher fan-out.
    """

    # --- identity (written once at intake) ---
    run_id: str
    user_request: str
    started_at: str

    # --- supervisor-owned ---
    brief: RequestBrief | None
    clarifications: Annotated[list[Clarification], operator.add]
    plan: TaskPlan | None
    plan_approved: bool
    research_round: int

    # --- researcher-owned (PARALLEL — every one of these must be additive) ---
    evidence: Annotated[list[Evidence], operator.add]
    research_handoffs: Annotated[list[ResearchHandoff], operator.add]
    task_status: Annotated[dict[str, TaskStatus], merge_status]

    # --- analyst / fact-checker / critic (sequential, single writer each) ---
    analysis: AnalysisHandoff | None
    fact_check: FactCheckReport | None
    critic_verdicts: Annotated[list[CriticVerdict], operator.add]
    revision_count: int

    # --- writer ---
    report: FinalReport | None

    # --- human checkpoints (§20) ---
    awaiting: str                       # which gate, "" when running
    human_decisions: Annotated[list[dict], operator.add]

    # --- observability (§23) — written by everything, so always additive ---
    trace: Annotated[list[TraceEvent], operator.add]
    errors: Annotated[list[ErrorRecord], operator.add]

    # --- control ---
    status: WorkflowStatus
    agent_calls: int
    abort_reason: str


def initial_state(user_request: str, run_id: str | None = None) -> WorkflowState:
    """A fresh run. Every additive channel starts as an empty list, never None."""
    return WorkflowState(
        run_id=run_id or uuid.uuid4().hex[:12],
        user_request=user_request.strip(),
        started_at=datetime.now(timezone.utc).isoformat(),
        brief=None,
        clarifications=[],
        plan=None,
        plan_approved=False,
        research_round=0,
        evidence=[],
        research_handoffs=[],
        task_status={},
        analysis=None,
        fact_check=None,
        critic_verdicts=[],
        revision_count=0,
        report=None,
        awaiting="",
        human_decisions=[],
        trace=[],
        errors=[],
        status=WorkflowStatus.PENDING,
        agent_calls=0,
        abort_reason="",
    )


# --------------------------------------------------------------------------- #
# Evidence id allocation — the subtle parallel bug
# --------------------------------------------------------------------------- #
def evidence_id_for(task_id: str, sequence: int) -> str:
    """Allocate a globally unique evidence id **without** a shared counter.

    Parallel researchers cannot share a mutable counter — each branch sees its own copy of
    state, so two branches both reading "next id is E4" would each mint E4 and one finding
    would be silently overwritten by id collision downstream.

    Instead the id is derived from the owning task, which is unique by construction: task R2's
    third finding is always ``E203``. No coordination, no collisions, and the id itself says
    which task produced it, which makes the trace readable.

    >>> evidence_id_for("R2", 3)
    'E203'
    """
    digits = "".join(ch for ch in task_id if ch.isdigit()) or "0"
    return f"E{int(digits)}{sequence:02d}"


# --------------------------------------------------------------------------- #
# §27 state specification — the single source of truth for the docs table
# --------------------------------------------------------------------------- #
ALL = frozenset(AgentId)
NOBODY: frozenset[AgentId] = frozenset()

FIELD_PERMISSIONS: dict[str, dict[str, Any]] = {
    "run_id":            {"type": "str",                "read": ALL, "write": {AgentId.SYSTEM}},
    "user_request":      {"type": "str",                "read": ALL, "write": {AgentId.SYSTEM}},
    "started_at":        {"type": "str",                "read": ALL, "write": {AgentId.SYSTEM}},
    "brief":             {"type": "RequestBrief",       "read": ALL, "write": {AgentId.SUPERVISOR}},
    "clarifications":    {"type": "list[Clarification]", "read": ALL, "write": {AgentId.SUPERVISOR}},
    "plan":              {"type": "TaskPlan",           "read": ALL, "write": {AgentId.SUPERVISOR}},
    "plan_approved":     {"type": "bool",               "read": ALL, "write": {AgentId.SYSTEM}},
    "research_round":    {"type": "int",                "read": ALL, "write": {AgentId.SUPERVISOR}},
    "evidence":          {"type": "list[Evidence]",
                          "read": {AgentId.ANALYST, AgentId.FACT_CHECKER, AgentId.CRITIC,
                                   AgentId.WRITER, AgentId.SUPERVISOR},
                          "write": {AgentId.RESEARCHER}},
    "research_handoffs": {"type": "list[ResearchHandoff]",
                          "read": {AgentId.ANALYST, AgentId.SUPERVISOR, AgentId.CRITIC},
                          "write": {AgentId.RESEARCHER}},
    "task_status":       {"type": "dict[str, TaskStatus]", "read": ALL,
                          "write": {AgentId.RESEARCHER, AgentId.SYSTEM}},
    "analysis":          {"type": "AnalysisHandoff",
                          "read": {AgentId.CRITIC, AgentId.FACT_CHECKER, AgentId.WRITER,
                                   AgentId.SUPERVISOR},
                          "write": {AgentId.ANALYST}},
    "fact_check":        {"type": "FactCheckReport",
                          "read": {AgentId.CRITIC, AgentId.SUPERVISOR},
                          "write": {AgentId.FACT_CHECKER}},
    "critic_verdicts":   {"type": "list[CriticVerdict]",
                          "read": {AgentId.SUPERVISOR, AgentId.ANALYST, AgentId.WRITER},
                          "write": {AgentId.CRITIC}},
    "revision_count":    {"type": "int",                "read": ALL, "write": {AgentId.SYSTEM}},
    "report":            {"type": "FinalReport",        "read": ALL, "write": {AgentId.WRITER}},
    "awaiting":          {"type": "str",                "read": ALL, "write": {AgentId.SYSTEM}},
    "human_decisions":   {"type": "list[dict]",         "read": ALL, "write": {AgentId.SYSTEM}},
    "trace":             {"type": "list[TraceEvent]",   "read": ALL, "write": ALL},
    "errors":            {"type": "list[ErrorRecord]",  "read": ALL, "write": ALL},
    "status":            {"type": "WorkflowStatus",     "read": ALL, "write": {AgentId.SYSTEM}},
    "agent_calls":       {"type": "int",                "read": ALL, "write": {AgentId.SYSTEM}},
    "abort_reason":      {"type": "str",                "read": ALL, "write": {AgentId.SYSTEM}},
}

# Channels the parallel researcher fan-out writes. Each MUST have a reducer in the Annotated
# type above, or LangGraph raises InvalidUpdateError the first time two researchers finish in
# the same superstep. Asserted in tests/test_state.py.
PARALLEL_WRITE_CHANNELS = frozenset({
    "evidence", "research_handoffs", "task_status", "trace", "errors",
})


def permissions_table() -> str:
    """Render the §27 table from the code, so docs cannot drift from behaviour."""
    rows = ["| State Field | Type | Read | Write |", "|---|---|---|---|"]
    for name, spec in FIELD_PERMISSIONS.items():
        read = "All" if spec["read"] == ALL else ", ".join(
            sorted(a.value for a in spec["read"])
        )
        write = "All" if spec["write"] == ALL else ", ".join(
            sorted(a.value for a in spec["write"])
        )
        rows.append(f"| `{name}` | {spec['type']} | {read} | {write} |")
    return "\n".join(rows)
