"""
Graph nodes: thin adapters between the agents and the shared state.

Every node has the same shape — read the slice of state it needs, call one agent, return a
partial state update. Nodes contain no reasoning of their own; anything resembling a decision
lives either in an agent (judgement) or in ``routing.py`` (deterministic).

Two conventions make the graph safe:

1. **Nodes never raise.** Agents return ``AgentOutcome`` rather than throwing, and nodes turn a
   failed outcome into an ``ErrorRecord`` on state. A raised exception would abort the whole
   graph and lose the evidence gathered so far.

2. **Nodes only write channels they own.** The §27 permission table is not decoration — the
   research node writes ``evidence`` and ``task_status`` because those carry reducers, and never
   touches ``analysis``, which does not.

``research_dispatch`` runs research tasks sequentially in this phase. It is written so the
per-task work already lives in :func:`research_one`, which is what Phase 7 will dispatch through
``Send()`` — the fan-out is a routing change, not a rewrite.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.agents import researcher as researcher_agent
from app.agents import reviewers, supervisor
from app.config import settings
from app.graph.state import WorkflowState
from app.schemas.common import AgentId, TaskStatus, WorkflowStatus
from app.schemas.reports import ErrorRecord, TraceEvent
from app.services.usage import BudgetExceeded, UsageTracker


@dataclass
class WorkflowDeps:
    """Injected dependencies. Held outside state because they are not serialisable.

    The three flags exist so the experiments in §30 are configuration changes rather than code
    branches. An experiment that requires editing the graph is an experiment whose control arm
    is a different program.
    """

    index: Any                                   # CorpusIndex
    store: Any = None                            # EvidenceStore | None
    usage: UsageTracker = field(default_factory=UsageTracker)
    run_id: str = ""

    parallel_research: bool = False              # Experiment 3
    critic_enabled: bool = True                  # Experiment 2
    max_revisions: int | None = None             # Experiment 5
    full_context: bool = False                   # Experiment 4

    def revision_cap(self) -> int:
        return settings.max_revision_cycles if self.max_revisions is None else self.max_revisions


def _trace(agent: AgentId, event: str, node: str, detail: str = "", task_id: str = "",
           seconds: float = 0.0) -> TraceEvent:
    return TraceEvent(agent_id=agent, event=event, node=node, detail=detail,
                      task_id=task_id, duration_seconds=round(seconds, 3))


def _fail(state_update: dict, agent: AgentId, node: str, error: ErrorRecord | None,
          reason: str) -> dict:
    """Terminate the run cleanly, preserving whatever was produced before the failure."""
    update = dict(state_update)
    update["status"] = WorkflowStatus.FAILED
    update["abort_reason"] = reason
    update.setdefault("errors", [])
    if error:
        update["errors"] = [*update["errors"], error]
    update.setdefault("trace", [])
    update["trace"] = [*update["trace"], _trace(agent, "error", node, reason)]
    return update


# --------------------------------------------------------------------------- #
# 1. intake — deterministic
# --------------------------------------------------------------------------- #
def make_intake(deps: WorkflowDeps):
    def intake(state: WorkflowState) -> dict:
        request = (state.get("user_request") or "").strip()
        if not request:
            return _fail({}, AgentId.SYSTEM, "intake", None, "Empty request.")
        if deps.store is not None and getattr(deps.store, "enabled", False):
            try:
                deps.store.start_run(state["run_id"], request, status="running")
            except Exception:  # noqa: BLE001 — persistence is optional, never fatal
                pass
        return {
            "status": WorkflowStatus.ANALYSING_REQUEST,
            "trace": [_trace(AgentId.SYSTEM, "node_end", "intake",
                             f"run {state['run_id']} started")],
        }

    return intake


# --------------------------------------------------------------------------- #
# 2. supervisor — analyse
# --------------------------------------------------------------------------- #
def make_analyse(deps: WorkflowDeps):
    def analyse(state: WorkflowState) -> dict:
        answers = "\n".join(
            f"Q: {c.question}\nA: {c.answer}" for c in state.get("clarifications", [])
        )
        outcome = supervisor.analyse_request(
            state["user_request"], usage=deps.usage, clarification_answers=answers
        )
        if outcome.failed or outcome.output is None:
            return _fail({"trace": outcome.trace}, AgentId.SUPERVISOR, "supervisor_analyse",
                         outcome.error, "Could not analyse the request.")

        brief = outcome.output
        needs = brief.needs_clarification and not state.get("clarifications")
        return {
            "brief": brief,
            "status": (WorkflowStatus.AWAITING_CLARIFICATION if needs
                       else WorkflowStatus.PLANNING),
            "awaiting": "clarification" if needs else "",
            "trace": [*outcome.trace, _trace(
                AgentId.SUPERVISOR, "handoff", "supervisor_analyse",
                f"brief: {len(brief.sub_questions)} sub-questions"
                + (", clarification required" if needs else ""),
                seconds=outcome.duration_seconds)],
        }

    return analyse


# --------------------------------------------------------------------------- #
# 3. supervisor — plan
# --------------------------------------------------------------------------- #
def make_plan(deps: WorkflowDeps):
    def plan(state: WorkflowState) -> dict:
        brief = state["brief"]
        verdicts = state.get("critic_verdicts", [])
        note = ""
        if verdicts and verdicts[-1].missing_evidence:
            note = ("The reviewer requires evidence that was not gathered:\n"
                    + "\n".join(f"  - {m}" for m in verdicts[-1].missing_evidence)
                    + "\nPlan research tasks that close these gaps.")

        outcome = supervisor.build_plan(brief, usage=deps.usage, revision_note=note)
        if outcome.failed or outcome.output is None:
            return _fail({"trace": outcome.trace}, AgentId.SUPERVISOR, "supervisor_plan",
                         outcome.error, "Could not produce a valid task plan.")

        task_plan = outcome.output
        if not task_plan.research_tasks():
            return _fail({"trace": outcome.trace}, AgentId.SUPERVISOR, "supervisor_plan", None,
                         "Plan contained no research tasks.")

        # Reset status for every task in the new plan. A re-plan reuses ids (R1, R2, ...), and
        # those ids are still marked COMPLETED from the previous round — so without this the
        # dispatcher skips every task and the extra research round does nothing at all while
        # still costing a supervisor call. Found by running the graph end to end.
        reset = {t.task_id: TaskStatus.PENDING for t in task_plan.tasks}

        return {
            "plan": task_plan,
            "task_status": reset,
            "research_round": state.get("research_round", 0) + 1,
            "status": WorkflowStatus.AWAITING_PLAN_APPROVAL,
            "trace": [*outcome.trace, _trace(
                AgentId.SUPERVISOR, "handoff", "supervisor_plan",
                f"{len(task_plan.tasks)} tasks, "
                f"{len(task_plan.research_tasks())} research",
                seconds=outcome.duration_seconds)],
        }

    return plan


# --------------------------------------------------------------------------- #
# 4. plan approval — the §20 checkpoint (interrupt arrives in Phase 8)
# --------------------------------------------------------------------------- #
def make_plan_approval(deps: WorkflowDeps):
    def plan_approval(state: WorkflowState) -> dict:
        """Auto-approves in this phase. Phase 8 replaces the body with ``interrupt()``.

        It exists as a node now rather than being added later so the graph topology — and every
        routing decision around it — is already correct when the interrupt is dropped in.
        """
        return {
            "plan_approved": True,
            "awaiting": "",
            "status": WorkflowStatus.RESEARCHING,
            "human_decisions": [{"gate": "plan_approval", "decision": "auto_approved",
                                 "note": "Human checkpoint is wired in Phase 8."}],
            "trace": [_trace(AgentId.SYSTEM, "checkpoint", "plan_approval", "auto-approved")],
        }

    return plan_approval


# --------------------------------------------------------------------------- #
# 5. research — one task (the unit Phase 7 will Send() to)
# --------------------------------------------------------------------------- #
def research_one(deps: WorkflowDeps, task_id: str, research_question: str,
                 objective: str) -> dict:
    """Run a single research task and return its state update.

    Written to be callable both from the sequential dispatcher below and, unchanged, from a
    ``Send()`` payload in Phase 7. Everything it writes carries a reducer.
    """
    started = time.perf_counter()
    try:
        outcome, evidence = researcher_agent.research(
            task_id=task_id, research_question=research_question, objective=objective,
            index=deps.index, usage=deps.usage, store=deps.store, run_id=deps.run_id,
        )
    except BudgetExceeded as e:
        return {
            "task_status": {task_id: TaskStatus.FAILED},
            "errors": [ErrorRecord(agent_id=AgentId.RESEARCHER, node="research",
                                   task_id=task_id, kind="budget", message=str(e),
                                   action_taken="abort")],
            "trace": [_trace(AgentId.RESEARCHER, "error", "research", str(e)[:80], task_id)],
        }

    update: dict = {
        "evidence": evidence,
        "trace": outcome.trace,
        "task_status": {task_id: TaskStatus.COMPLETED if outcome.ok else TaskStatus.FAILED},
    }
    if outcome.ok and outcome.output is not None:
        update["research_handoffs"] = [outcome.output]
    if outcome.error:
        update["errors"] = [outcome.error]

    update["trace"] = [*update["trace"], _trace(
        AgentId.RESEARCHER, "handoff", "research",
        f"{len(evidence)} evidence" if outcome.ok else "failed",
        task_id, time.perf_counter() - started)]
    return update


def make_research_task(deps: WorkflowDeps):
    """The ``Send()`` target: one researcher, one sub-question, executed concurrently.

    LangGraph delivers the ``Send`` payload as this node's input rather than the full workflow
    state, which is exactly the isolation §21 asks for — a researcher physically cannot read a
    sibling's findings, so parallel branches stay independent rather than merely being told to.

    Everything returned here carries a reducer (``evidence``, ``research_handoffs``,
    ``task_status``, ``trace``, ``errors``), so concurrent completions merge instead of
    overwriting. That property was designed in Phase 1 and asserted before the fan-out existed.
    """
    def research_task(payload: dict) -> dict:
        return research_one(
            deps,
            payload["task_id"],
            payload.get("research_question", ""),
            payload.get("objective", ""),
        )

    return research_task


def make_research_dispatch(deps: WorkflowDeps):
    def research_dispatch(state: WorkflowState) -> dict:
        """Run every pending research task, sequentially.

        Phase 7 replaces this node with a ``Send()`` fan-out over the same
        :func:`research_one`. Keeping the sequential path afterwards is not dead code — it is
        the control arm for Experiment 3, which measures what the fan-out actually saves.
        """
        plan = state["plan"]
        objective = state["brief"].objective
        tasks = [t for t in plan.research_tasks()
                 if state.get("task_status", {}).get(t.task_id) is not TaskStatus.COMPLETED]

        merged: dict = {"evidence": [], "research_handoffs": [], "errors": [],
                        "trace": [], "task_status": {}}
        for task in tasks[: settings.max_parallel_researchers]:
            update = research_one(deps, task.task_id, task.research_question or "", objective)
            for key in ("evidence", "research_handoffs", "errors", "trace"):
                merged[key] = [*merged[key], *update.get(key, [])]
            merged["task_status"].update(update.get("task_status", {}))

        merged["status"] = WorkflowStatus.ANALYSING_EVIDENCE
        merged["trace"] = [*merged["trace"], _trace(
            AgentId.SYSTEM, "node_end", "research_dispatch",
            f"{len(tasks)} task(s) sequential, {len(merged['evidence'])} evidence")]
        return merged

    return research_dispatch


# --------------------------------------------------------------------------- #
# 6. evidence gate — deterministic
# --------------------------------------------------------------------------- #
def make_evidence_gate(deps: WorkflowDeps):
    def evidence_gate(state: WorkflowState) -> dict:
        plan = supervisor.apply_task_status(state["plan"], state.get("task_status", {}))
        # Score coverage against the questions actually assigned, not every sub-question in the
        # brief — the plan is capped at max_parallel_researchers, so the brief is usually wider.
        assigned = [t.research_question for t in plan.research_tasks() if t.research_question]
        decision = supervisor.evidence_gate(
            state["brief"], state.get("evidence", []), state.get("research_handoffs", []),
            state.get("research_round", 1), questions=assigned,
        )
        return {
            "plan": plan,
            "status": (WorkflowStatus.ANALYSING_EVIDENCE if decision.proceed
                       else WorkflowStatus.PLANNING),
            "trace": [_trace(AgentId.SYSTEM, "node_end", "evidence_gate", decision.reason)],
        }

    return evidence_gate


# --------------------------------------------------------------------------- #
# 7. analyst
# --------------------------------------------------------------------------- #
def make_analyst(deps: WorkflowDeps):
    def analyst(state: WorkflowState) -> dict:
        verdicts = state.get("critic_verdicts", [])
        note = ""
        if verdicts and not verdicts[-1].approved:
            last = verdicts[-1]
            note = ("\n".join(f"  - [{p.location}] {p.issue}" for p in last.problems)
                    + "\nRequired changes:\n"
                    + "\n".join(f"  - {r}" for r in last.required_revisions))

        revision = state.get("revision_count", 0)
        outcome = reviewers.analyse(
            state["brief"], state.get("evidence", []), state.get("research_handoffs", []),
            usage=deps.usage, revision_note=note, revision=revision,
        )
        if outcome.failed or outcome.output is None:
            return _fail({"trace": outcome.trace}, AgentId.ANALYST, "analyst",
                         outcome.error, "The analyst could not produce a valid analysis.")

        analysis = outcome.output
        return {
            "analysis": analysis,
            "status": WorkflowStatus.FACT_CHECKING,
            "trace": [*outcome.trace, _trace(
                AgentId.ANALYST, "handoff", "analyst",
                f"{len(analysis.conclusions)} conclusions "
                f"({len(analysis.major_conclusions())} major), revision {revision}",
                seconds=outcome.duration_seconds)],
        }

    return analyst


# --------------------------------------------------------------------------- #
# 8. fact-checker
# --------------------------------------------------------------------------- #
def make_fact_checker(deps: WorkflowDeps):
    def fact_checker(state: WorkflowState) -> dict:
        outcome = reviewers.fact_check(state["analysis"], state.get("evidence", []),
                                       usage=deps.usage)
        report = outcome.output
        update: dict = {
            "fact_check": report,
            "status": WorkflowStatus.REVIEWING,
            "trace": [*outcome.trace, _trace(
                AgentId.FACT_CHECKER, "handoff", "fact_checker",
                f"{len(report.checks) if report else 0} checked, "
                f"{len(report.fabricated) if report else 0} fabricated, "
                f"{len(report.unsupported_ids) if report else 0} unsupported",
                seconds=outcome.duration_seconds)],
        }
        if outcome.error:
            # Degraded, not failed: the deterministic half of the check still ran.
            update["errors"] = [outcome.error]
        return update

    return fact_checker


# --------------------------------------------------------------------------- #
# 9. critic
# --------------------------------------------------------------------------- #
def make_critic(deps: WorkflowDeps):
    def critic(state: WorkflowState) -> dict:
        if not deps.critic_enabled:
            # Experiment 2's control arm. Recorded explicitly so a run without review can never
            # be mistaken for a run that passed review.
            from app.schemas.handoffs import CriticVerdict

            return {
                "critic_verdicts": [CriticVerdict(
                    approved=True, cycle=state.get("revision_count", 0),
                    missing_evidence=["Critic was disabled for this run (experiment control)."],
                )],
                "status": WorkflowStatus.WRITING,
                "trace": [_trace(AgentId.CRITIC, "node_end", "critic", "disabled (experiment)")],
            }

        cycle = state.get("revision_count", 0)
        outcome = reviewers.review(
            state["brief"], state["analysis"], state.get("evidence", []),
            state.get("fact_check"), state.get("research_handoffs", []),
            usage=deps.usage, cycle=cycle,
        )
        verdict = outcome.output
        if verdict is None:
            return _fail({"trace": outcome.trace}, AgentId.CRITIC, "critic", outcome.error,
                         "The critic produced no verdict.")

        update: dict = {
            "critic_verdicts": [verdict],
            "status": WorkflowStatus.WRITING if verdict.approved else WorkflowStatus.REVISING,
            "trace": [*outcome.trace, _trace(
                AgentId.CRITIC, "handoff", "critic",
                f"{'approved' if verdict.approved else 'rejected'} — "
                f"{len(verdict.problems)} problem(s), cycle {cycle}",
                seconds=outcome.duration_seconds)],
        }
        if outcome.error:
            update["errors"] = [outcome.error]
        return update

    return critic


# --------------------------------------------------------------------------- #
# 10. revision counter — deterministic, and the reason the loop terminates
# --------------------------------------------------------------------------- #
def make_revision(deps: WorkflowDeps):
    def revision(state: WorkflowState) -> dict:
        """Increment the revision counter before looping back to the analyst.

        Separated from the critic node deliberately. If the counter were incremented inside the
        critic, a critic that failed and retried would inflate it; if it were incremented in the
        analyst, a first-pass analysis would count as a revision. A dedicated node on the
        rejection edge increments exactly once per completed loop.
        """
        count = state.get("revision_count", 0) + 1
        return {
            "revision_count": count,
            "status": WorkflowStatus.REVISING,
            "trace": [_trace(AgentId.SYSTEM, "node_end", "revision",
                             f"revision {count} of {deps.revision_cap()}")],
        }

    return revision


# --------------------------------------------------------------------------- #
# 11. writer
# --------------------------------------------------------------------------- #
def make_writer(deps: WorkflowDeps):
    def writer(state: WorkflowState) -> dict:
        verdicts = state.get("critic_verdicts", [])
        outcome = reviewers.write_report(
            state["brief"], state["analysis"], state.get("evidence", []),
            state.get("research_handoffs", []),
            verdict=verdicts[-1] if verdicts else None, usage=deps.usage,
        )
        if outcome.failed or outcome.output is None:
            return _fail({"trace": outcome.trace}, AgentId.WRITER, "writer", outcome.error,
                         "The writer could not produce a report.")

        report = outcome.output
        return {
            "report": report,
            "status": WorkflowStatus.AWAITING_FINAL_REVIEW,
            "trace": [*outcome.trace, _trace(
                AgentId.WRITER, "handoff", "writer",
                f"report: {len(report.key_findings)} findings, "
                f"{len(report.evidence_used)} evidence cited",
                seconds=outcome.duration_seconds)],
        }

    return writer


# --------------------------------------------------------------------------- #
# 12. finalise — persist and close the run
# --------------------------------------------------------------------------- #
def make_finalise(deps: WorkflowDeps):
    def finalise(state: WorkflowState) -> dict:
        report = state.get("report")
        run_id = state["run_id"]

        if deps.store is not None and getattr(deps.store, "enabled", False):
            # Persistence must never turn a completed run into a failed one.
            for label, fn in (
                ("evidence", lambda: deps.store.save_evidence(run_id, state.get("evidence", []))),
                ("trace", lambda: deps.store.save_trace(run_id, state.get("trace", []))),
                ("errors", lambda: deps.store.save_errors(run_id, state.get("errors", []))),
                ("report", lambda: deps.store.save_report(
                    run_id, report.title, report.to_markdown()) if report else None),
                ("run", lambda: deps.store.finish_run(
                    run_id, status=WorkflowStatus.COMPLETED.value,
                    agent_calls=deps.usage.billable_calls,
                    input_tokens=deps.usage.total_input_tokens,
                    output_tokens=deps.usage.total_output_tokens,
                    cost_usd=deps.usage.total_cost_usd,
                    revision_count=state.get("revision_count", 0),
                    wall_seconds=deps.usage.elapsed_seconds)),
            ):
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

        return {
            "status": WorkflowStatus.COMPLETED,
            "awaiting": "",
            "agent_calls": deps.usage.billable_calls,
            "trace": [_trace(AgentId.SYSTEM, "node_end", "finalise",
                             f"{deps.usage.billable_calls} model calls "
                             f"({deps.usage.total_calls} attempted), "
                             f"{deps.usage.total_input_tokens + deps.usage.total_output_tokens} "
                             f"tokens, {deps.usage.elapsed_seconds:.1f}s")],
        }

    return finalise
