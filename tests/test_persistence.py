"""
Persistence tests for the graph's ``finalise`` node.

This closes a real gap. Every live workflow run so far passed ``store=None``, so the code that
writes evidence, traces, errors and the report to Postgres had never actually executed — it was
covered only by the "disabled store no-ops" tests, which exercise the opposite branch.

Two properties matter and are tested separately:

1. **It writes what it claims to write.** Against a real database when one is configured.
2. **It cannot fail the run.** A completed workflow whose database is unreachable must still be
   reported as completed. Persistence is durability, not a dependency — and a finalise node that
   raised would turn every transient network blip into a lost run.
"""
from __future__ import annotations

import pytest

from app.graph.nodes import WorkflowDeps, make_finalise, make_intake
from app.graph.state import initial_state
from app.schemas.common import AgentId, ClaimType, Confidence, WorkflowStatus
from app.schemas.evidence import Evidence
from app.schemas.reports import ErrorRecord, FinalReport, Recommendation, TraceEvent
from app.services.usage import UsageTracker
from app.storage.evidence_store import EvidenceStore

RUN_ID = "test-finalise-node"


def _state() -> dict:
    state = initial_state("Compare two frameworks", run_id=RUN_ID)
    state["evidence"] = [
        Evidence(
            evidence_id="E101", claim="LangGraph uses typed state channels",
            supporting_text="State is declared as a TypedDict whose channels may carry reducers.",
            source_id="fw-langgraph-docs", source_title="LangGraph Documentation",
            research_question="What state management does LangGraph provide?",
            confidence=Confidence.HIGH, claim_type=ClaimType.FACT,
            agent_id="researcher", task_id="R1",
        )
    ]
    state["trace"] = [
        TraceEvent(agent_id=AgentId.SUPERVISOR, event="handoff", node="supervisor_plan",
                   detail="3 research tasks", duration_seconds=1.2),
        TraceEvent(agent_id=AgentId.CRITIC, event="handoff", node="critic",
                   detail="approved", duration_seconds=2.4),
    ]
    state["errors"] = [
        ErrorRecord(agent_id=AgentId.RESEARCHER, node="research", task_id="R2",
                    kind="api_failure", message="rate limited", recovered=True,
                    action_taken="fallback_provider")
    ]
    state["report"] = FinalReport(
        title="Framework Comparison", executive_summary="Summary.",
        research_objective="Compare frameworks", methodology="Corpus research.",
        key_findings=["LangGraph has typed state"],
        recommendation=Recommendation(statement="Adopt LangGraph", evidence_ids=["E101"]),
        evidence_used=state["evidence"],
    )
    state["revision_count"] = 1
    return state


def _usage() -> UsageTracker:
    usage = UsageTracker(run_id=RUN_ID)
    usage.record(agent_id="supervisor", provider="groq", model="m",
                 input_tokens=1200, output_tokens=300, seconds=1.2)
    usage.record(agent_id="critic", provider="groq", model="m",
                 input_tokens=800, output_tokens=150, seconds=2.4)
    return usage


# --------------------------------------------------------------------------- #
# Must never fail the run
# --------------------------------------------------------------------------- #
class BrokenStore:
    """A store whose every write raises. Simulates an unreachable database."""

    enabled = True

    def _boom(self, *a, **k):
        raise RuntimeError("connection refused")

    start_run = save_evidence = save_trace = save_errors = _boom
    save_report = finish_run = _boom


def test_finalise_completes_even_when_the_database_is_unreachable():
    """A completed workflow must not be reported as failed because persistence broke."""
    deps = WorkflowDeps(index=None, store=BrokenStore(), usage=_usage(), run_id=RUN_ID)
    update = make_finalise(deps)(_state())

    assert update["status"] is WorkflowStatus.COMPLETED
    assert update["agent_calls"] == 2


def test_intake_survives_an_unreachable_database():
    deps = WorkflowDeps(index=None, store=BrokenStore(), usage=UsageTracker(), run_id=RUN_ID)
    update = make_intake(deps)(initial_state("Compare things", run_id=RUN_ID))
    assert update["status"] is WorkflowStatus.ANALYSING_REQUEST


def test_finalise_without_a_store_still_completes():
    deps = WorkflowDeps(index=None, store=None, usage=_usage(), run_id=RUN_ID)
    update = make_finalise(deps)(_state())
    assert update["status"] is WorkflowStatus.COMPLETED


def test_finalise_reports_measured_usage_not_estimates():
    deps = WorkflowDeps(index=None, store=None, usage=_usage(), run_id=RUN_ID)
    update = make_finalise(deps)(_state())
    detail = update["trace"][0].detail
    assert "2 model calls" in detail
    assert "2450 tokens" in detail          # 1200+300+800+150, summed not guessed


# --------------------------------------------------------------------------- #
# Real database round trip
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not EvidenceStore().enabled, reason="no DATABASE_URL configured")
def test_finalise_writes_everything_to_the_database():
    """The write path the live runs never exercised, end to end against real Postgres."""
    store = EvidenceStore()
    deps = WorkflowDeps(index=None, store=store, usage=_usage(), run_id=RUN_ID)
    state = _state()

    # finalise assumes intake already registered the run (foreign keys depend on it).
    make_intake(deps)(state)
    update = make_finalise(deps)(state)
    assert update["status"] is WorkflowStatus.COMPLETED

    try:
        evidence = store.load_evidence(RUN_ID)
        assert len(evidence) == 1
        assert evidence[0].evidence_id == "E101"
        assert evidence[0].confidence is Confidence.HIGH
        assert evidence[0].claim_type is ClaimType.FACT

        markdown = store.load_report(RUN_ID)
        assert markdown and "## Executive Summary" in markdown
        assert "E101" in markdown

        with store._conn() as conn:
            traces = conn.execute(
                "SELECT agent_id, event, node FROM w4_trace WHERE run_id = %s ORDER BY at",
                (RUN_ID,)).fetchall()
            errors = conn.execute(
                "SELECT kind, recovered, action_taken FROM w4_errors WHERE run_id = %s",
                (RUN_ID,)).fetchall()
            run = conn.execute(
                "SELECT status, agent_calls, input_tokens, output_tokens, revision_count "
                "FROM w4_runs WHERE run_id = %s", (RUN_ID,)).fetchone()

        assert len(traces) == 2 and traces[0][0] == "supervisor"
        assert len(errors) == 1 and errors[0] == ("api_failure", True, "fallback_provider")
        assert run[0] == "completed"
        assert run[1] == 2                     # agent_calls
        assert run[2] == 2000 and run[3] == 450  # tokens, summed from the tracker
        assert run[4] == 1                     # revision_count
    finally:
        with store._conn() as conn:
            conn.execute("DELETE FROM w4_runs WHERE run_id = %s", (RUN_ID,))
            conn.commit()


@pytest.mark.skipif(not EvidenceStore().enabled, reason="no DATABASE_URL configured")
def test_cascade_delete_removes_every_child_row():
    """Deleting a run must not strand evidence, traces or reports behind it."""
    store = EvidenceStore()
    deps = WorkflowDeps(index=None, store=store, usage=_usage(), run_id=RUN_ID)
    state = _state()
    make_intake(deps)(state)
    make_finalise(deps)(state)

    with store._conn() as conn:
        conn.execute("DELETE FROM w4_runs WHERE run_id = %s", (RUN_ID,))
        conn.commit()
        for table in ("w4_evidence", "w4_trace", "w4_errors", "w4_reports"):
            left = conn.execute(
                f"SELECT count(*) FROM {table} WHERE run_id = %s", (RUN_ID,)).fetchone()[0]
            assert left == 0, f"{table} kept {left} orphaned row(s)"
