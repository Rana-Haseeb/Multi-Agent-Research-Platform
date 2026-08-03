"""
The workflow graph — assembly, compilation, and the public entry point.

Topology::

    START -> intake -> supervisor_analyse -+-> END  (awaiting clarification)
                                           |
                                           +-> supervisor_plan -> plan_approval
                                                                       |
                                                                       v
                                                              research_dispatch
                                                                       |
                                                                       v
                                                                evidence_gate
                                                                  |         |
                                                       (thin) supervisor_plan
                                                                            |
                                                                            v
                                                                         analyst
                                                                            |
                                                                            v
                                                                     fact_checker
                                                                            |
                                                                            v
                                                                          critic
                                                                    |            |
                                                            (reject) revision  (approve)
                                                                    |            |
                                                          analyst / plan       writer
                                                                                 |
                                                                                 v
                                                                             finalise -> END

Every edge out of an agent node is conditional on ``status`` being non-terminal, so a failure
anywhere short-circuits to END with the state intact rather than cascading into nodes that would
read a field the failed node never wrote.

The compiled graph takes a checkpointer, which does nothing in this phase and is what makes the
Phase 8 ``interrupt()`` calls resumable. Wiring it now means Phase 8 adds pauses to a graph that
is already durable, instead of retrofitting durability around pauses.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.graph import routing as R
from app.graph.nodes import (
    WorkflowDeps,
    make_analyse,
    make_analyst,
    make_critic,
    make_evidence_gate,
    make_fact_checker,
    make_final_review,
    make_finalise,
    make_intake,
    make_plan,
    make_plan_approval,
    make_research_dispatch,
    make_research_task,
    make_revision,
    make_writer,
)
from app.graph.state import WorkflowState, initial_state
from app.schemas.common import WorkflowStatus
from app.services.usage import UsageTracker
from app.storage.corpus import get_index
from app.storage.evidence_store import EvidenceStore


def build_workflow(deps: WorkflowDeps, checkpointer: Any = None):
    """Compile the graph. Dependencies are injected, never global."""
    g = StateGraph(WorkflowState)

    g.add_node(R.INTAKE, make_intake(deps))
    g.add_node(R.ANALYSE, make_analyse(deps))
    g.add_node(R.PLAN, make_plan(deps))
    g.add_node(R.PLAN_APPROVAL, make_plan_approval(deps))
    g.add_node(R.RESEARCH, make_research_dispatch(deps))
    g.add_node(R.RESEARCH_TASK, make_research_task(deps))
    g.add_node(R.EVIDENCE_GATE, make_evidence_gate(deps))
    g.add_node(R.ANALYST, make_analyst(deps))
    g.add_node(R.FACT_CHECKER, make_fact_checker(deps))
    g.add_node(R.CRITIC, make_critic(deps))
    g.add_node(R.REVISION, make_revision(deps))
    g.add_node(R.WRITER, make_writer(deps))
    g.add_node(R.FINAL_REVIEW, make_final_review(deps))
    g.add_node(R.FINALISE, make_finalise(deps))

    g.add_edge(START, R.INTAKE)
    g.add_conditional_edges(R.INTAKE, R.route_after_intake, [R.ANALYSE, END])
    g.add_conditional_edges(R.ANALYSE, R.route_after_analyse, [R.PLAN, END])
    g.add_conditional_edges(R.PLAN, R.route_after_plan, [R.PLAN_APPROVAL, END])
    # One edge, two arms: a node name for the sequential path, a Send list for the fan-out.
    g.add_conditional_edges(R.PLAN_APPROVAL, R.make_route_after_plan_approval(deps),
                            [R.RESEARCH, R.RESEARCH_TASK, END])
    g.add_edge(R.RESEARCH, R.EVIDENCE_GATE)
    g.add_edge(R.RESEARCH_TASK, R.EVIDENCE_GATE)
    g.add_conditional_edges(R.EVIDENCE_GATE, R.make_route_after_gate(deps),
                            [R.PLAN, R.ANALYST, END])
    g.add_conditional_edges(R.ANALYST, R.route_after_analyst, [R.FACT_CHECKER, END])
    g.add_conditional_edges(R.FACT_CHECKER, R.route_after_fact_check, [R.CRITIC, END])
    g.add_conditional_edges(R.CRITIC, R.make_route_after_critic(deps),
                            [R.REVISION, R.WRITER, END])
    g.add_conditional_edges(R.REVISION, R.make_route_after_revision(deps),
                            [R.ANALYST, R.PLAN, END])
    g.add_conditional_edges(R.WRITER, R.route_after_writer, [R.FINAL_REVIEW, END])
    g.add_conditional_edges(R.FINAL_REVIEW, R.route_after_final_review, [R.FINALISE, END])
    g.add_edge(R.FINALISE, END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
@dataclass
class WorkflowResult:
    """Everything a caller — the dashboard, the eval runner, an experiment — needs."""

    run_id: str
    state: WorkflowState
    deps: WorkflowDeps
    wall_seconds: float
    thread_id: str = ""

    @property
    def status(self) -> WorkflowStatus:
        return self.state.get("status", WorkflowStatus.PENDING)

    @property
    def completed(self) -> bool:
        return self.status is WorkflowStatus.COMPLETED

    @property
    def report_markdown(self) -> str:
        report = self.state.get("report")
        return report.to_markdown() if report else ""

    def summary(self) -> dict:
        """One row per run. This is what the §29 metrics are computed from."""
        state = self.state
        verdicts = state.get("critic_verdicts", [])
        evidence = state.get("evidence", [])
        usage = self.deps.usage.summary()
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "wall_seconds": round(self.wall_seconds, 2),
            "agent_calls": usage["billable_calls"],
            "attempted_calls": usage["total_calls"],
            "refused_calls": usage["refused_calls"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost_usd": usage["cost_usd"],
            "by_agent": usage["by_agent"],
            "tasks": len(state["plan"].tasks) if state.get("plan") else 0,
            "research_tasks": len(state["plan"].research_tasks()) if state.get("plan") else 0,
            "evidence_count": len(evidence),
            "revision_count": state.get("revision_count", 0),
            "research_rounds": state.get("research_round", 0),
            "critic_approved": verdicts[-1].approved if verdicts else None,
            "critic_problems": sum(len(v.problems) for v in verdicts),
            "fabricated_citations": (
                list(state["fact_check"].fabricated) if state.get("fact_check") else []
            ),
            "errors": [
                {"kind": e.kind, "agent": e.agent_id.value, "message": e.message[:160]}
                for e in state.get("errors", [])
            ],
            "trace_events": len(state.get("trace", [])),
            "has_report": state.get("report") is not None,
            "abort_reason": state.get("abort_reason", ""),
        }

    def trace_lines(self) -> list[str]:
        return [e.line() for e in self.state.get("trace", [])]


# Sentinel distinguishing "caller did not specify a store" from "caller explicitly wants none".
# Previously both were ``None``, so ``store=None`` — which every test passes, meaning to disable
# persistence — silently constructed a real EvidenceStore and wrote to the live database. The
# test suite had been creating rows in the production tables, and those rows displaced a real
# run from ``recent_runs()``, which is how this was noticed.
_UNSET = object()


def run_workflow(
    user_request: str,
    *,
    run_id: str | None = None,
    index: Any = None,
    store: Any = _UNSET,
    clarifications: list | None = None,
    parallel_research: bool = False,
    critic_enabled: bool = True,
    max_revisions: int | None = None,
    full_context: bool = False,
    checkpointer: Any = None,
    thread_id: str | None = None,
) -> WorkflowResult:
    """Run one request end to end.

    The keyword flags are the experiment levers (§30) — Experiments 2, 3, 4 and 5 differ from the
    default run by exactly one of them, so each comparison isolates a single variable rather
    than a different code path.

    ``recursion_limit`` is a backstop only. The counters in ``routing.py`` normally stop the graph
    well before it, and a run that hits this limit indicates a routing bug rather than a
    legitimately long workflow.
    """
    rid = run_id or uuid.uuid4().hex[:12]
    deps = WorkflowDeps(
        index=index if index is not None else get_index(),
        store=EvidenceStore() if store is _UNSET else store,
        usage=UsageTracker(run_id=rid),
        run_id=rid,
        parallel_research=parallel_research,
        critic_enabled=critic_enabled,
        max_revisions=max_revisions,
        full_context=full_context,
    )

    graph = build_workflow(deps, checkpointer=checkpointer)
    state = initial_state(user_request, run_id=rid)
    if clarifications:
        state["clarifications"] = list(clarifications)

    tid = thread_id or rid
    config = {
        "configurable": {"thread_id": tid},
        "recursion_limit": 4 * (settings.max_revision_cycles + settings.max_research_rounds) + 30,
    }

    started = time.perf_counter()
    try:
        final = graph.invoke(state, config=config)
    except Exception as e:  # noqa: BLE001
        # The graph itself failed — recursion limit, or a node that raised despite the
        # convention. Return a coherent failed result rather than propagating, so the eval
        # runner records the failure instead of dying on it.
        from app.schemas.common import AgentId
        from app.schemas.reports import ErrorRecord

        state["status"] = WorkflowStatus.FAILED
        state["abort_reason"] = f"Graph execution failed: {type(e).__name__}: {e}"[:300]
        state["errors"] = [*state.get("errors", []), ErrorRecord(
            agent_id=AgentId.SYSTEM, node="graph", kind="graph_failure",
            message=str(e)[:400], action_taken="abort")]
        final = state

    return WorkflowResult(run_id=rid, state=final, deps=deps,
                          wall_seconds=time.perf_counter() - started, thread_id=tid)


# Single source of truth for the topology. Both the architecture doc and its mermaid diagram are
# generated from this list, so a diagram cannot drift from the graph that actually runs. Kept
# next to build_workflow so an edge added there without a matching entry here is visible in
# review — and asserted by tests/test_workflow.py::test_documented_edges_match_the_graph.
EDGES: list[tuple[str, str, str]] = [
        (START, R.INTAKE, ""),
        (R.INTAKE, R.ANALYSE, "ok"),
        (R.ANALYSE, R.PLAN, "clear"),
        (R.ANALYSE, END, "clarification required"),
        (R.PLAN, R.PLAN_APPROVAL, "valid plan"),
        (R.PLAN_APPROVAL, R.RESEARCH, "approved, sequential"),
        (R.PLAN_APPROVAL, R.RESEARCH_TASK, "approved, parallel fan-out"),
        (R.RESEARCH_TASK, R.EVIDENCE_GATE, ""),
        (R.RESEARCH, R.EVIDENCE_GATE, ""),
        (R.EVIDENCE_GATE, R.PLAN, "coverage thin, round < cap"),
        (R.EVIDENCE_GATE, R.ANALYST, "coverage sufficient"),
        (R.ANALYST, R.FACT_CHECKER, ""),
        (R.FACT_CHECKER, R.CRITIC, ""),
        (R.CRITIC, R.WRITER, "approved, or revision cap reached"),
        (R.CRITIC, R.REVISION, "rejected, under cap"),
        (R.REVISION, R.ANALYST, "re-analysis only"),
        (R.REVISION, R.PLAN, "needs more research"),
        (R.WRITER, R.FINAL_REVIEW, ""),
        (R.FINAL_REVIEW, R.FINALISE, "reviewed"),
        (R.FINALISE, END, ""),
]


class WorkflowSession:
    """A resumable run. Holds the graph, checkpointer and usage tracker across interrupts.

    ``run_workflow`` builds a fresh graph per call, which is correct for unattended runs but
    cannot support ``interrupt()``: resuming needs the *same* compiled graph, the *same*
    checkpointer and the *same* usage tracker, or the resumed half of the run is metered
    separately and the checkpoint cannot be found at all.

    Week 3's lesson applies directly — the session object must be held in
    ``st.session_state`` so it survives Streamlit's top-to-bottom rerun on every interaction.
    A session rebuilt per rerun loses the checkpointer and the pause becomes unresumable.

    Usage::

        session = WorkflowSession("Compare X and Y", human_in_the_loop=True)
        result = session.start()
        while session.pending_interrupt():
            result = session.resume({"decision": "approve"})
    """

    def __init__(
        self,
        user_request: str,
        *,
        run_id: str | None = None,
        index: Any = None,
        store: Any = _UNSET,
        clarifications: list | None = None,
        human_in_the_loop: bool = True,
        parallel_research: bool = False,
        critic_enabled: bool = True,
        max_revisions: int | None = None,
        full_context: bool = False,
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.user_request = user_request
        self.deps = WorkflowDeps(
            index=index if index is not None else get_index(),
            store=EvidenceStore() if store is _UNSET else store,
            usage=UsageTracker(run_id=self.run_id),
            run_id=self.run_id,
            parallel_research=parallel_research,
            critic_enabled=critic_enabled,
            max_revisions=max_revisions,
            full_context=full_context,
            human_in_the_loop=human_in_the_loop,
        )
        self.checkpointer = MemorySaver()
        self.graph = build_workflow(self.deps, checkpointer=self.checkpointer)
        self.thread_id = self.run_id
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": (
                4 * (settings.max_revision_cycles + settings.max_research_rounds) + 30
            ),
        }
        self._initial = initial_state(user_request, run_id=self.run_id)
        if clarifications:
            self._initial["clarifications"] = list(clarifications)
        self._started = 0.0
        self._elapsed = 0.0

    # ------------------------------------------------------------------ drive
    def start(self) -> WorkflowResult:
        self._started = time.perf_counter()
        return self._invoke(self._initial)

    def resume(self, decision: Any) -> WorkflowResult:
        """Continue a paused run with the human's decision.

        ``decision`` may be a string ("approve"), a dict, or a ``HumanCheckpointResponse``;
        the nodes normalise all three.
        """
        from langgraph.types import Command

        if not self._started:
            self._started = time.perf_counter()
        return self._invoke(Command(resume=decision))

    def _invoke(self, payload: Any) -> WorkflowResult:
        try:
            state = self.graph.invoke(payload, config=self.config)
        except Exception as e:  # noqa: BLE001
            state = dict(self.snapshot() or self._initial)
            state["status"] = WorkflowStatus.FAILED
            state["abort_reason"] = f"Graph execution failed: {type(e).__name__}: {e}"[:300]
        self._elapsed = time.perf_counter() - self._started
        return WorkflowResult(run_id=self.run_id, state=state, deps=self.deps,
                              wall_seconds=self._elapsed, thread_id=self.thread_id)

    # ------------------------------------------------------------------ state
    def snapshot(self) -> dict | None:
        snap = self.graph.get_state(self.config)
        return dict(snap.values) if snap and snap.values else None

    def pending_interrupt(self) -> dict | None:
        """The payload the graph is paused on, or None if it is not waiting for a human.

        This is what the §29 approval-compliance metric reads: a workflow that reached the
        Writer without ever surfacing a checkpoint did not respect the gate.
        """
        snap = self.graph.get_state(self.config)
        if not getattr(snap, "next", None):
            return None
        for task in getattr(snap, "tasks", []) or []:
            interrupts = getattr(task, "interrupts", None)
            if interrupts:
                value = interrupts[0].value
                return value if isinstance(value, dict) else {"payload": value}
        return None

    def is_waiting(self) -> bool:
        return self.pending_interrupt() is not None


def describe_topology() -> str:
    """Node and edge listing, for the generated architecture doc."""
    return "\n".join(
        f"  {src:<20} -> {dst:<20} {('[' + label + ']') if label else ''}"
        for src, dst, label in EDGES
    )
