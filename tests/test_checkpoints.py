"""
Phase 8 tests: the two human checkpoints (§20).

The §29 metric these support is **human approval compliance, target 100%**. A checkpoint that
can be bypassed makes that metric meaningless, so these assert three separate things:

1. The gate actually pauses — the workflow stops *before* the expensive stage, not after it.
2. Rejection is honoured — a rejected plan does not quietly proceed to research.
3. An unattended run cannot be mistaken for an approved one — it records that it skipped review.

Point 3 matters most. The evaluation runner disables the checkpoints because it cannot sit at a
prompt; if that were silent, every unattended eval run would score as "approval respected".
"""
from __future__ import annotations

import pytest

from app.graph.workflow import WorkflowSession, run_workflow
from app.schemas.common import WorkflowStatus
from app.schemas.request import HumanCheckpointResponse
from app.schemas.common import HumanDecision
from tests.test_workflow import full_script


def _session(fake_llm_factory, corpus_index, **kw):
    fake_llm_factory(full_script(n_research=2))
    return WorkflowSession("Compare LangGraph and CrewAI", index=corpus_index, store=None,
                           human_in_the_loop=True, **kw)


# --------------------------------------------------------------------------- #
# Checkpoint 1 — plan approval
# --------------------------------------------------------------------------- #
def test_run_pauses_at_plan_approval(fake_llm_factory, corpus_index):
    session = _session(fake_llm_factory, corpus_index)
    session.start()

    pending = session.pending_interrupt()
    assert pending is not None, "the workflow did not pause for plan approval"
    assert pending["gate"] == "plan_approval"
    assert pending["research_task_count"] == 2
    assert "plan" in pending and pending["options"] == ["approve", "edit", "reject"]


def test_the_pause_happens_before_research_is_paid_for(fake_llm_factory, corpus_index):
    """§20's whole point: stop before the expensive stage, not after it."""
    created = fake_llm_factory(full_script(n_research=2))
    session = WorkflowSession("Compare LangGraph and CrewAI", index=corpus_index, store=None,
                              human_in_the_loop=True)
    session.start()

    state = session.snapshot()
    assert state["evidence"] == [], "research ran before the human approved the plan"
    # Exactly the two Supervisor calls (analyse + plan). FakeLLM does not write to the usage
    # tracker, so the assertion is on its own call log.
    assert len(created["supervisor"].calls) == 2
    assert "researcher" not in created, "a researcher was constructed before approval"


def test_approving_continues_the_run(fake_llm_factory, corpus_index):
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    result = session.resume("approve")

    assert session.pending_interrupt() is not None, "should now pause at the final review"
    assert session.pending_interrupt()["gate"] == "final_review"
    assert len(result.state["evidence"]) == 2


def test_rejecting_the_plan_aborts_without_researching(fake_llm_factory, corpus_index):
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    result = session.resume({"decision": "reject", "note": "wrong framing"})

    assert result.status is WorkflowStatus.ABORTED
    assert result.state["evidence"] == [], "rejected plan still ran research"
    assert result.state["report"] is None
    assert "wrong framing" in result.state["abort_reason"]
    assert not result.state["plan_approved"]


def test_editing_the_plan_replaces_it(fake_llm_factory, corpus_index):
    from app.schemas.common import AgentId
    from app.schemas.tasks import Task

    session = _session(fake_llm_factory, corpus_index)
    session.start()

    edited = [
        Task(task_id="R1", description="Only this one",
             assigned_agent=AgentId.RESEARCHER,
             research_question="What state management does LangGraph provide?"),
        Task(task_id="A1", description="Analyse", assigned_agent=AgentId.ANALYST,
             depends_on=["R1"]),
    ]
    result = session.resume({"decision": "edit",
                             "edited_payload": {"tasks": [t.model_dump() for t in edited]}})

    plan = result.state["plan"]
    assert len(plan.research_tasks()) == 1, "the user's edit was ignored"
    assert plan.revision == 1


def test_an_invalid_edit_keeps_the_original_plan(fake_llm_factory, corpus_index):
    """A human may decide, but may not bypass the DAG invariants."""
    from app.schemas.common import AgentId
    from app.schemas.tasks import Task

    session = _session(fake_llm_factory, corpus_index)
    session.start()
    original = len(session.snapshot()["plan"].tasks)

    cyclic = [
        Task(task_id="R1", description="a", assigned_agent=AgentId.RESEARCHER,
             research_question="q?", depends_on=["A1"]),
        Task(task_id="A1", description="b", assigned_agent=AgentId.ANALYST,
             depends_on=["R1"]),
    ]
    result = session.resume({"decision": "edit",
                             "edited_payload": {"tasks": [t.model_dump() for t in cyclic]}})

    assert len(result.state["plan"].tasks) == original, "a cyclic plan was accepted"
    assert any(e.kind == "invalid_output" and e.recovered for e in result.state["errors"])


# --------------------------------------------------------------------------- #
# Checkpoint 2 — final review
# --------------------------------------------------------------------------- #
def test_run_pauses_at_final_review_with_the_recommendation(fake_llm_factory, corpus_index):
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    session.resume("approve")

    pending = session.pending_interrupt()
    assert pending["gate"] == "final_review"
    assert pending["recommendation"] == "Adopt LangGraph"
    assert pending["markdown"].startswith("# ")
    assert "options" in pending


def test_approving_the_report_completes_the_run(fake_llm_factory, corpus_index):
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    session.resume("approve")
    result = session.resume("approve")

    assert result.completed
    assert result.state["report"] is not None
    assert session.pending_interrupt() is None


def test_rejecting_the_report_keeps_it_and_records_the_objection(fake_llm_factory, corpus_index):
    """The artefact survives the disagreement — discarding it would lose the evidence of it."""
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    session.resume("approve")
    result = session.resume({"decision": "reject", "note": "conclusion is too strong"})

    assert result.completed
    assert result.state["report"] is not None
    assert any("too strong" in lim for lim in result.state["report"].risks_and_limitations)


# --------------------------------------------------------------------------- #
# Both gates, and the compliance metric
# --------------------------------------------------------------------------- #
def test_both_gates_are_recorded(fake_llm_factory, corpus_index):
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    session.resume("approve")
    result = session.resume({"decision": "approve", "note": "looks right"})

    gates = [d["gate"] for d in result.state["human_decisions"]]
    assert gates == ["plan_approval", "final_review"]
    assert all(d["decision"] == "approve" for d in result.state["human_decisions"])


def test_unattended_runs_declare_that_review_was_skipped(fake_llm_factory, corpus_index):
    """§29 compliance depends on this: silence would score as approval."""
    fake_llm_factory(full_script(n_research=2))
    result = run_workflow("Compare", index=corpus_index, store=None)

    assert result.completed
    decisions = result.state["human_decisions"]
    assert [d["gate"] for d in decisions] == ["plan_approval", "final_review"]
    assert all(d["decision"] == "auto_approved" for d in decisions)
    assert all("disabled" in d["note"].lower() for d in decisions)


def test_unattended_runs_never_pause(fake_llm_factory, corpus_index):
    """The evaluation runner cannot sit at a prompt."""
    fake_llm_factory(full_script(n_research=2))
    result = run_workflow("Compare", index=corpus_index, store=None)
    assert result.status is WorkflowStatus.COMPLETED
    assert result.state["awaiting"] == ""


def test_checkpoint_decisions_normalise_from_several_shapes(fake_llm_factory, corpus_index):
    """The UI sends a dict, tests send a string, the schema sends an object."""
    from app.graph.nodes import _decision_of

    assert _decision_of("approve") == "approve"
    assert _decision_of({"decision": "reject"}) == "reject"
    assert _decision_of(HumanCheckpointResponse(decision=HumanDecision.EDIT,
                                                note="x")) == "edit"
    assert _decision_of(True) == "approve"
    assert _decision_of(False) == "reject"
    assert _decision_of("nonsense") == "approve"   # never silently abort on a bad value


def test_resuming_is_idempotent_across_the_interrupt(fake_llm_factory, corpus_index):
    """`interrupt()` re-runs its node from the top on resume.

    Anything mutating state above the call would apply twice. This asserts the plan is not
    duplicated or re-planned by the replay.
    """
    session = _session(fake_llm_factory, corpus_index)
    session.start()
    before = session.snapshot()["plan"]
    result = session.resume("approve")

    assert len(result.state["plan"].tasks) == len(before.tasks)
    assert result.state["research_round"] == 1, "the replay re-planned"
