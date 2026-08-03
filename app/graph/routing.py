"""
Conditional edges — every routing decision in the workflow.

All of them are pure functions of state. None calls a model. That is the point: routing is where
a multi-agent system either stays predictable or becomes impossible to reason about, and a
router that asks an LLM "what should happen next?" cannot be tested, cannot be proven to
terminate, and fails differently every run.

The termination argument for the whole graph rests on three counters, each compared here:

- ``revision_count``  vs the revision cap    — bounds the critic loop
- ``research_round``  vs the research cap    — bounds re-planning
- ``agent_calls``     vs the call budget     — bounds everything else

Because each is monotonically increasing and each comparison routes forward when the cap is met,
no cycle can repeat indefinitely. ``tests/test_workflow.py`` asserts this against a critic that
rejects forever.
"""
from __future__ import annotations

from langgraph.graph import END
from langgraph.types import Send

from app.config import settings
from app.graph.nodes import WorkflowDeps
from app.graph.state import WorkflowState
from app.schemas.common import TaskStatus, WorkflowStatus

# Node names, in one place so a typo is an import error rather than a silent dead edge.
INTAKE = "intake"
ANALYSE = "supervisor_analyse"
PLAN = "supervisor_plan"
PLAN_APPROVAL = "plan_approval"
RESEARCH = "research_dispatch"      # sequential arm (Experiment 3 control)
RESEARCH_TASK = "research_task"      # Send() target, one per sub-question
EVIDENCE_GATE = "evidence_gate"
ANALYST = "analyst"
FACT_CHECKER = "fact_checker"
CRITIC = "critic"
REVISION = "revision"
WRITER = "writer"
FINALISE = "finalise"


def _dead(state: WorkflowState) -> bool:
    status = state.get("status")
    return isinstance(status, WorkflowStatus) and status.is_terminal()


def route_after_intake(state: WorkflowState) -> str:
    return END if _dead(state) else ANALYSE


def route_after_analyse(state: WorkflowState) -> str:
    """Clarification is a terminal pause, not a branch.

    When the request is ambiguous the run stops and returns to the caller with
    ``AWAITING_CLARIFICATION``. It resumes as a fresh invocation carrying the user's answers in
    ``clarifications`` — which is why :func:`make_analyse` suppresses a second clarification
    request once answers exist, and why this cannot loop.
    """
    if _dead(state):
        return END
    if state.get("status") is WorkflowStatus.AWAITING_CLARIFICATION:
        return END
    return PLAN


def route_after_plan(state: WorkflowState) -> str:
    return END if _dead(state) else PLAN_APPROVAL


def make_route_after_plan_approval(deps: WorkflowDeps):
    """Dispatch research — sequentially, or fanned out across concurrent branches (§19).

    A conditional edge may return either a node name or a list of ``Send`` objects, so the same
    edge serves both arms of Experiment 3 without duplicating the graph. That matters: the
    control arm has to be the *same program* under a different flag, or the comparison measures
    two codebases rather than one variable.

    Each ``Send`` carries only its own task, so a researcher receives its sub-question and
    nothing else. Results merge back through the state reducers.
    """
    def route_after_plan_approval(state: WorkflowState):
        if _dead(state) or not state.get("plan_approved"):
            return END

        plan = state.get("plan")
        if not deps.parallel_research or plan is None:
            return RESEARCH

        objective = state["brief"].objective if state.get("brief") else ""
        done = state.get("task_status", {})
        pending = [
            t for t in plan.research_tasks()
            if done.get(t.task_id) is not TaskStatus.COMPLETED
        ][: settings.max_parallel_researchers]

        if not pending:
            return RESEARCH      # nothing to fan out; the sequential node reports the emptiness
        return [
            Send(RESEARCH_TASK, {
                "task_id": t.task_id,
                "research_question": t.research_question or "",
                "objective": objective,
            })
            for t in pending
        ]

    return route_after_plan_approval


def make_route_after_gate(deps: WorkflowDeps):
    def route_after_gate(state: WorkflowState) -> str:
        """Back to planning when evidence is thin, forward otherwise.

        The round cap is enforced in :func:`app.agents.supervisor.evidence_gate`, which sets
        ``proceed=True`` once it is reached. This function only reads the resulting status, so
        the cap cannot be bypassed by a routing change.
        """
        if _dead(state):
            return END
        if state.get("status") is WorkflowStatus.PLANNING:
            if state.get("research_round", 0) >= settings.max_research_rounds:
                return ANALYST      # belt and braces: never re-plan past the cap
            return PLAN
        return ANALYST

    return route_after_gate


def route_after_analyst(state: WorkflowState) -> str:
    return END if _dead(state) else FACT_CHECKER


def route_after_fact_check(state: WorkflowState) -> str:
    return END if _dead(state) else CRITIC


def make_route_after_critic(deps: WorkflowDeps):
    def route_after_critic(state: WorkflowState) -> str:
        """The quality-control loop, and the guarantee that it ends.

        §18 requires the workflow to terminate even if the Critic never approves. That guarantee
        is this comparison — a model asked to "stop after two revisions" will eventually not.
        """
        if _dead(state):
            return END

        verdicts = state.get("critic_verdicts", [])
        if not verdicts:
            return WRITER                        # no verdict: do not stall the run
        verdict = verdicts[-1]

        if verdict.approved:
            return WRITER

        revisions = state.get("revision_count", 0)
        if revisions >= deps.revision_cap():
            return WRITER                        # cap reached; objections go into the report

        if state.get("agent_calls", 0) >= settings.max_agent_calls_per_run:
            return WRITER                        # budget backstop

        if (verdict.needs_more_research
                and state.get("research_round", 0) < settings.max_research_rounds):
            return REVISION                      # counted, then routed back through planning

        return REVISION

    return route_after_critic


def make_route_after_revision(deps: WorkflowDeps):
    def route_after_revision(state: WorkflowState) -> str:
        """A revision either needs new evidence or only re-analysis."""
        if _dead(state):
            return END
        verdicts = state.get("critic_verdicts", [])
        if (verdicts and verdicts[-1].needs_more_research
                and state.get("research_round", 0) < settings.max_research_rounds):
            return PLAN
        return ANALYST

    return route_after_revision


def route_after_writer(state: WorkflowState) -> str:
    """Phase 8 inserts the second human checkpoint between here and finalise."""
    return END if _dead(state) else FINALISE
