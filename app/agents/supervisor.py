"""
Supervisor: request analysis, dynamic planning, and completion decisions.

The Supervisor is two model calls, not one, and the split is deliberate. Analysis decides *what*
the question is; planning decides *how* to answer it. Fusing them lets a model that misread the
question produce a confident plan for the wrong work, and the clarification gate then has nothing
to fire on — the plan already exists.

Everything after planning is deterministic. :func:`evidence_gate` and :func:`next_action` decide
routing in plain Python, because "is coverage above threshold" and "have we hit the revision cap"
are not judgement calls, and paying a model to make them would add latency, cost and a failure
mode for no gain.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import AgentOutcome, structured_step
from app.agents.context import supervisor_context
from app.agents.prompts import SUPERVISOR_ANALYSE, SUPERVISOR_PLAN
from app.config import settings
from app.schemas.common import AgentId, TaskStatus
from app.schemas.evidence import Evidence, EvidenceIndex
from app.schemas.handoffs import CriticVerdict, ResearchHandoff
from app.schemas.request import RequestBrief
from app.schemas.tasks import Task, TaskPlan
from app.services.usage import UsageTracker


class PlanDraft(BaseModel):
    """What the model emits for planning.

    Separate from :class:`TaskPlan` on purpose. ``TaskPlan`` validates as a DAG and rejects
    cycles, and a validation error raised *inside* structured output is retried blindly by the
    provider. Accepting a loose draft and validating it ourselves turns a bad plan into a
    reportable failure with a specific message instead of an opaque retry loop.
    """

    tasks: list[Task] = Field(default_factory=list)
    rationale: str = ""


def analyse_request(
    user_request: str, usage: UsageTracker | None = None, clarification_answers: str = ""
) -> AgentOutcome[RequestBrief]:
    """First call: turn prose into a structured brief, or ask for clarification."""
    user = f"USER REQUEST:\n{user_request}"
    if clarification_answers:
        user += (
            f"\n\nTHE USER HAS SINCE ANSWERED YOUR QUESTIONS:\n{clarification_answers}\n\n"
            f"Produce the brief now. Do not ask again — you have what you need."
        )
    return structured_step(
        agent_id=AgentId.SUPERVISOR, node="supervisor_analyse",
        system=SUPERVISOR_ANALYSE, user=user, schema=RequestBrief, usage=usage,
        detail="request brief",
    )


def build_plan(
    brief: RequestBrief, usage: UsageTracker | None = None, revision_note: str = ""
) -> AgentOutcome[TaskPlan]:
    """Second call: a dependency-ordered plan derived from the brief."""
    user = supervisor_context(brief.objective, brief, None, [], [])
    if revision_note:
        user += f"\n\nRE-PLANNING REQUIRED:\n{revision_note}"
    user += (
        f"\n\nProduce a task plan. Use at most {settings.max_parallel_researchers} research "
        f"tasks — one per sub-question, the most important ones if there are more."
    )

    outcome = structured_step(
        agent_id=AgentId.SUPERVISOR, node="supervisor_plan",
        system=SUPERVISOR_PLAN, user=user, schema=PlanDraft, usage=usage,
        detail="task plan",
    )
    if outcome.failed or outcome.output is None:
        return AgentOutcome(agent_id=AgentId.SUPERVISOR, ok=False, error=outcome.error,
                            trace=outcome.trace, duration_seconds=outcome.duration_seconds)

    draft = outcome.output
    try:
        plan = TaskPlan(tasks=draft.tasks, rationale=draft.rationale)
    except Exception as e:  # noqa: BLE001  (pydantic ValidationError)
        from app.agents.base import _error

        return AgentOutcome(
            agent_id=AgentId.SUPERVISOR, ok=False, trace=outcome.trace,
            duration_seconds=outcome.duration_seconds,
            error=_error(AgentId.SUPERVISOR, "supervisor_plan", "", "invalid_output",
                         f"Plan failed validation: {e}"),
        )

    plan = collapse_duplicates(plan)
    return AgentOutcome(agent_id=AgentId.SUPERVISOR, ok=True, output=plan,
                        trace=outcome.trace, duration_seconds=outcome.duration_seconds)


def collapse_duplicates(plan: TaskPlan) -> TaskPlan:
    """Drop research tasks that duplicate another's question (§22, "duplicate tasks").

    Deterministic string similarity, not a model call. Dependents are rewired onto the surviving
    task so the DAG stays valid. Running the same search twice costs a real call and produces
    duplicate evidence that inflates every coverage metric downstream.
    """
    groups = plan.duplicate_groups()
    if not groups:
        return plan

    replace: dict[str, str] = {}
    drop: set[str] = set()
    for group in groups:
        keeper, *rest = sorted(group)
        for dup in rest:
            replace[dup] = keeper
            drop.add(dup)

    tasks: list[Task] = []
    for task in plan.tasks:
        if task.task_id in drop:
            continue
        deps = []
        for dep in task.depends_on:
            mapped = replace.get(dep, dep)
            if mapped not in deps:
                deps.append(mapped)
        tasks.append(task.model_copy(update={"depends_on": deps}))

    note = f" Collapsed {len(drop)} duplicate research task(s): {', '.join(sorted(drop))}."
    return TaskPlan(tasks=tasks, rationale=(plan.rationale + note).strip(),
                    revision=plan.revision)


# --------------------------------------------------------------------------- #
# Deterministic routing
# --------------------------------------------------------------------------- #
class GateDecision(BaseModel):
    """Why the workflow is going where it is going. Recorded in the trace."""

    proceed: bool
    reason: str
    unresolved_questions: list[str] = Field(default_factory=list)
    coverage: dict[str, float] = Field(default_factory=dict)


def evidence_gate(
    brief: RequestBrief,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
    research_round: int,
    questions: list[str] | None = None,
) -> GateDecision:
    """Is there enough evidence to analyse? Plain code, no model call.

    ``questions`` must be the questions **actually assigned to research tasks**, not every
    sub-question in the brief. The first end-to-end run failed because of this: the Supervisor
    decomposed a request into ten sub-questions, the plan was capped at four research tasks, and
    the gate then scored coverage against all ten. Six questions nobody had been assigned could
    never be covered, so the gate re-planned on every pass and the run burned its entire call
    budget without reaching the Analyst.

    Scoring against unassigned questions is not conservatism, it is a guaranteed false negative.
    Coverage of the brief as a whole is a separate concern, reported to the Critic as
    completeness rather than used to gate progress here.

    Coverage is confidence-weighted and ignores assumptions, so three low-confidence guesses do
    not clear the bar one solid finding would. When coverage is thin the workflow re-plans — but
    only while ``research_round`` is under the cap, after which it proceeds with the gaps
    recorded rather than looping. §22 requires termination even when evidence never arrives.
    """
    scored = questions if questions is not None else brief.sub_questions
    index = EvidenceIndex(items=evidence)
    coverage = index.coverage(scored)
    unresolved = index.unresolved(scored)

    if not scored:
        # No research task carried a question. Nothing to gate on; let the Analyst and Critic
        # report the emptiness rather than looping here.
        return GateDecision(proceed=True, reason="No research questions were assigned.",
                            unresolved_questions=[], coverage={})

    if not evidence:
        if research_round < settings.max_research_rounds:
            return GateDecision(proceed=False, reason="No evidence was gathered; re-planning.",
                                unresolved_questions=unresolved, coverage=coverage)
        return GateDecision(
            proceed=True,
            reason="No evidence after the maximum research rounds. Proceeding so the workflow "
                   "terminates; the report will record this as a limitation.",
            unresolved_questions=unresolved, coverage=coverage,
        )

    if unresolved and research_round < settings.max_research_rounds:
        return GateDecision(
            proceed=False,
            reason=f"{len(unresolved)} of {len(scored)} assigned question(s) have thin "
                   f"coverage; running one more research round.",
            unresolved_questions=unresolved, coverage=coverage,
        )

    reason = "Evidence coverage is sufficient."
    if unresolved:
        reason = (f"Proceeding with {len(unresolved)} thin sub-question(s) — research round cap "
                  f"reached. Gaps will be reported.")
    return GateDecision(proceed=True, reason=reason, unresolved_questions=unresolved,
                        coverage=coverage)


class RouteDecision(BaseModel):
    next_step: str
    reason: str
    terminal: bool = False


def next_action(
    verdict: CriticVerdict | None, revision_count: int, research_round: int
) -> RouteDecision:
    """Route after the Critic. Deterministic, and guaranteed to terminate.

    §18 requires the workflow to finish even if the Critic keeps rejecting. That guarantee cannot
    live in a prompt: a model asked to "stop after two revisions" will occasionally not. Here the
    cap is a comparison, so the third rejection routes to the Writer with the objections recorded
    in the report rather than acted on.
    """
    if verdict is None:
        return RouteDecision(next_step="critic", reason="No verdict yet.")

    if verdict.approved:
        return RouteDecision(next_step="writer", reason="Critic approved the analysis.")

    if revision_count >= settings.max_revision_cycles:
        return RouteDecision(
            next_step="writer",
            reason=(f"Critic still rejects after {revision_count} revision(s), the configured "
                    f"maximum. Proceeding to report; unresolved objections will be recorded in "
                    f"Risks and Limitations."),
        )

    if verdict.needs_more_research and research_round < settings.max_research_rounds:
        return RouteDecision(next_step="research",
                             reason="Critic requires evidence that has not been gathered.")

    return RouteDecision(next_step="analyst",
                         reason=f"Revision {revision_count + 1} of "
                                f"{settings.max_revision_cycles} requested by the Critic.")


def apply_task_status(plan: TaskPlan, statuses: dict[str, TaskStatus]) -> TaskPlan:
    """Fold reported statuses into the plan, marking unreachable tasks SKIPPED.

    A task whose dependency failed can never run. Leaving it PENDING makes ``is_complete()``
    false forever and the workflow hangs — so it is closed out explicitly.
    """
    tasks = [
        t.model_copy(update={"status": statuses[t.task_id]}) if t.task_id in statuses else t
        for t in plan.tasks
    ]
    updated = TaskPlan(tasks=tasks, rationale=plan.rationale, revision=plan.revision)
    blocked = {t.task_id for t in updated.blocked()}
    if not blocked:
        return updated
    return TaskPlan(
        tasks=[t.model_copy(update={"status": TaskStatus.SKIPPED})
               if t.task_id in blocked else t for t in updated.tasks],
        rationale=updated.rationale, revision=updated.revision,
    )
