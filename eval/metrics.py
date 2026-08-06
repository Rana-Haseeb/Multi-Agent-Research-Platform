"""
Evaluation metrics (§29).

Every metric here is written to be **falsifiable**. Week 3 shipped a tool-selection metric of
``expected.issubset(actual)`` which, because ``set().issubset(anything)`` is True, scored every
no-tool case correct regardless of what the agent did — a permanent 100% that measured nothing.
It was caught only by auditing a suspiciously perfect score.

So three rules apply throughout:

1. **A metric with no applicable cases returns ``None``, never 1.0.** "Nothing to check" and
   "everything passed" are different answers and must not render identically.
2. **Every metric has a paired test proving it can fail** — see ``tests/test_eval_metrics.py``.
3. **Metrics only count cases that can support them.** A PLAN-depth case has no evidence, so
   including it in evidence coverage would drive the average toward zero for reasons unrelated
   to quality. ``EvalCase.supports_metric`` decides.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from app.schemas.common import AgentId
from eval.dataset import Depth, EvalCase


@dataclass
class CaseResult:
    """What one evaluation run produced, alongside what the case expected."""

    case_id: str
    category: str
    depth: str
    status: str = ""
    error: str = ""

    # planning stage
    clarification_requested: bool = False
    research_tasks: int = 0
    total_tasks: int = 0
    agents_assigned: list[str] = field(default_factory=list)
    options_found: list[str] = field(default_factory=list)
    plan_valid: bool = False

    # downstream (FULL depth)
    evidence_count: int = 0
    gaps_declared: int = 0
    handoffs: int = 0
    handoffs_expected: int = 0
    fabricated_citations: list[str] = field(default_factory=list)
    major_conclusions: int = 0
    uncited_major: int = 0
    critic_ran: bool = False
    revision_count: int = 0
    report_produced: bool = False
    report_has_limitations: bool = False

    # governance / cost
    checkpoints_recorded: list[str] = field(default_factory=list)
    agent_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    wall_seconds: float = 0.0

    # scoring, filled by score_case
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    @property
    def infrastructure_failure(self) -> bool:
        """True when no provider would answer, so the system never got to run.

        A run that spent **zero billable calls** and failed at the first agent step did not fail
        to complete a workflow — it never started one. Counting that as a workflow failure means
        `workflow_completion_rate` measures the provider's daily quota rather than the software,
        and the number moves when nothing about the system has changed.

        Observed directly: three evaluation cases failed identically at 3.2s with 0 calls and
        "Could not analyse the request" immediately after a 42-call case drained the per-minute
        token bucket. The same cases pass once quota recovers. Excluding them is not charity to
        the system; including them would be measuring the wrong subject.
        """
        return (
            self.agent_calls == 0
            and self.status in {"failed", "crashed"}
            and self.evidence_count == 0
        )


# --------------------------------------------------------------------------- #
# Per-case scoring
# --------------------------------------------------------------------------- #
def score_case(case: EvalCase, result: CaseResult) -> dict[str, bool]:
    """Compare one result against the case's *declared* expectations.

    Only checks that apply are recorded. An absent check is not a silent pass — the aggregate
    counts each check independently, so a case contributing three checks cannot mask a fourth
    it never ran.
    """
    checks: dict[str, bool] = {}

    # An outage is not a result. Scoring it would attribute the provider's quota to the system.
    if result.infrastructure_failure and not case.expect_failure:
        return {}

    # --- failure cases short-circuit -------------------------------------------
    if case.expect_failure:
        checks["failed_as_expected"] = result.status in {"failed", "aborted"}
        checks["spent_nothing_on_a_bad_request"] = result.agent_calls == 0
        return checks

    # --- clarification (§10) ---------------------------------------------------
    checks["clarification_decision"] = (
        result.clarification_requested == case.expect_clarification
    )

    # A run that stopped for clarification never produced a plan and never reached the approval
    # gate, so the plan-stage and approval checks DO NOT APPLY to it.
    #
    # This mattered: the first evaluation pass scored four clarification disagreements as
    # *four* failures each — clarification, plan validity, task count, and approval compliance —
    # which dragged human_approval_compliance to 76% when not one approval failure had occurred.
    # One disagreement must cost one check, or a single behaviour silently triples its weight and
    # corrupts an unrelated metric.
    if result.clarification_requested:
        checks["stopped_before_planning"] = result.research_tasks == 0
        return checks

    # --- task planning (§11, §29 planning accuracy) ----------------------------
    checks["plan_is_valid_dag"] = result.plan_valid
    checks["task_count_reasonable"] = (
        case.expect_min_research_tasks <= result.research_tasks <= case.expect_max_research_tasks
    )
    if case.expect_options:
        # Substring containment, not equality: the brief legitimately records the full product
        # name ("Zorblax Framework") where the case names the distinctive part ("Zorblax").
        # Exact matching failed a case in which the system had identified both options
        # perfectly — the matcher was wrong, not the run.
        found = " | ".join(result.options_found).lower()
        checks["identified_the_options"] = any(
            expected.lower() in found for expected in case.expect_options
        )

    # --- agent routing (§29 routing accuracy) ----------------------------------
    assigned = {a.lower() for a in result.agents_assigned}
    known = {a.value for a in AgentId.llm_agents()}
    checks["only_known_agents_assigned"] = assigned <= known
    checks["research_assigned_to_researcher"] = (
        AgentId.RESEARCHER.value in assigned if result.research_tasks else True
    )

    # --- human approval compliance (§29, target 100%) --------------------------
    if case.expected_checkpoint not in {"none", "clarification"}:
        checks["approval_checkpoint_recorded"] = (
            case.expected_checkpoint in result.checkpoints_recorded
        )

    if case.depth is Depth.PLAN:
        return checks

    # --- everything below needs a completed workflow ---------------------------
    checks["workflow_completed"] = result.status == "completed"
    checks["report_produced"] = result.report_produced == case.expect_report

    if case.expect_evidence:
        checks["evidence_gathered"] = result.evidence_count > 0
    else:
        # The interesting half: when the corpus cannot answer, the system must say so rather
        # than invent. Declaring a gap IS the correct behaviour here.
        checks["absence_declared_not_invented"] = (
            result.gaps_declared > 0 or result.evidence_count == 0
        )
    if case.expect_gaps:
        checks["gaps_declared"] = result.gaps_declared > 0
        checks["gaps_reached_the_report"] = (
            result.report_has_limitations if result.report_produced else True
        )

    # --- handoffs (§29 handoff success) ----------------------------------------
    if result.handoffs_expected:
        checks["handoffs_completed"] = result.handoffs >= result.handoffs_expected

    # --- evidence integrity (§29 unsupported major claims below 10%) -----------
    checks["no_fabricated_citations_survived"] = not result.fabricated_citations
    checks["no_uncited_major_conclusions"] = result.uncited_major == 0
    checks["critic_reviewed_the_analysis"] = result.critic_ran
    return checks


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _rate(numerator: int, denominator: int) -> float | None:
    """None when nothing applied. Never 1.0 for an empty set (§7.3)."""
    if denominator <= 0:
        return None
    return round(numerator / denominator, 3)


def _check_rate(results: list[CaseResult], name: str) -> tuple[float | None, int, int]:
    applicable = [r for r in results if name in r.checks]
    passed = sum(r.checks[name] for r in applicable)
    return _rate(passed, len(applicable)), passed, len(applicable)


def aggregate(cases: list[EvalCase], results: list[CaseResult]) -> dict[str, Any]:
    """Compute the §29 metric set from scored results."""
    by_id = {c.case_id: c for c in cases}
    scored = [r for r in results if r.checks]
    blocked = [r for r in results if not r.checks and r.infrastructure_failure]

    def metric(name: str, *check_names: str) -> dict:
        rates = [_check_rate(scored, c) for c in check_names]
        passed = sum(p for _, p, _ in rates)
        total = sum(t for _, _, t in rates)
        return {"rate": _rate(passed, total), "passed": passed, "applicable": total}

    completed = [r for r in scored if r.status == "completed"]
    full_depth = [r for r in scored if r.depth == Depth.FULL.value]

    # Cost and latency are only meaningful over runs that actually did the work.
    times = [r.wall_seconds for r in full_depth if r.wall_seconds > 0]
    calls = [r.agent_calls for r in full_depth if r.agent_calls > 0]

    summary: dict[str, Any] = {
        "cases_run": len(scored),
        "cases_passed": sum(r.passed for r in scored),
        "pass_rate": _rate(sum(r.passed for r in scored), len(scored)),

        # §29 required metrics
        "task_planning_accuracy": metric("planning", "plan_is_valid_dag",
                                         "task_count_reasonable", "identified_the_options"),
        "agent_routing_accuracy": metric("routing", "only_known_agents_assigned",
                                         "research_assigned_to_researcher"),
        "clarification_accuracy": metric("clarification", "clarification_decision",
                                         "stopped_before_planning"),
        # Runs paused at a human gate are excluded. A workflow awaiting clarification has not
        # failed — it is waiting for input, and resuming it would complete it. Whether it
        # *should* have paused is already measured by clarification_accuracy, so counting it
        # here too charges one disagreement to two metrics. That is the same double-count that
        # previously dragged human_approval_compliance to 76% with zero approval failures.
        "workflow_completion_rate": _rate(
            len([r for r in full_depth if r.status == "completed"]),
            len([r for r in full_depth if r.status != "awaiting_clarification"])),
        "paused_for_clarification": len(
            [r for r in full_depth if r.status == "awaiting_clarification"]),
        "evidence_coverage": metric("evidence", "evidence_gathered",
                                    "absence_declared_not_invented", "gaps_declared",
                                    "gaps_reached_the_report"),
        "handoff_success_rate": metric("handoffs", "handoffs_completed"),
        "human_approval_compliance": metric("approval", "approval_checkpoint_recorded"),
        "unsupported_major_claims": metric("unsupported", "no_fabricated_citations_survived",
                                           "no_uncited_major_conclusions"),

        # cost and latency (§29)
        "average_workflow_seconds": round(statistics.mean(times), 1) if times else None,
        "median_workflow_seconds": round(statistics.median(times), 1) if times else None,
        "average_agent_calls": round(statistics.mean(calls), 1) if calls else None,
        "total_input_tokens": sum(r.input_tokens for r in scored),
        "total_output_tokens": sum(r.output_tokens for r in scored),
        "estimated_cost_usd": round(sum(r.cost_usd for r in scored), 6),
    }

    # Per-category breakdown, so a weak category cannot hide inside a good average.
    per_category: dict[str, dict] = {}
    for result in scored:
        row = per_category.setdefault(result.category, {"run": 0, "passed": 0})
        row["run"] += 1
        row["passed"] += int(result.passed)
    for row in per_category.values():
        row["rate"] = _rate(row["passed"], row["run"])
    summary["by_category"] = per_category

    summary["failing_cases"] = [
        {"case_id": r.case_id,
         "failed_checks": sorted(k for k, v in r.checks.items() if not v)}
        for r in scored if not r.passed
    ]
    # Reported explicitly rather than silently dropped: a reader must be able to see that four
    # cases produced no measurement, and why.
    summary["blocked_by_provider"] = sorted(r.case_id for r in blocked)
    summary["not_run"] = sorted(
        set(by_id) - {r.case_id for r in scored} - {r.case_id for r in blocked}
    )
    return summary


# §29's stated initial targets, for the report's pass/fail column.
TARGETS: dict[str, float] = {
    "agent_routing_accuracy": 0.90,
    "workflow_completion_rate": 0.80,
    "handoff_success_rate": 0.90,
    "human_approval_compliance": 1.00,
    "unsupported_major_claims": 0.90,   # inverted: >=90% of runs free of unsupported claims
}


def against_targets(summary: dict) -> list[dict]:
    """Compare measured metrics with §29's targets. Unmeasured metrics report as unmeasured."""
    rows = []
    for name, target in TARGETS.items():
        value = summary.get(name)
        rate = value.get("rate") if isinstance(value, dict) else value
        rows.append({
            "metric": name,
            "target": target,
            "measured": rate,
            "met": None if rate is None else rate >= target,
        })
    return rows
