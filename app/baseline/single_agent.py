"""
Single-agent baseline — the control arm for Experiment 1.

This exists to answer the graded question honestly: *does the multi-agent architecture actually
improve anything, and by how much?* An answer is only worth having if the comparison is fair, so
this baseline is built now, alongside the specialists, rather than assembled after the fact when
the temptation to weaken it would be strongest.

**What is held constant:** the same corpus, the same tools, the same model tier, the same usage
metering, the same output schema. The baseline is not handicapped.

**What differs — and this is the entire experimental variable:** one agent does everything in one
context. No task decomposition, no independent research per sub-question, no separation between
gathering and concluding, no critic, no citation verification, no revision loop.

If the specialists do not beat this, that is a real finding and the builder journal will say so.
"""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from app.agents.base import (
    AgentOutcome,
    structured_step,
    summarise_tool_results,
    tool_loop,
)
from app.agents.prompts import BASELINE
from app.schemas.common import AgentId, Confidence
from app.schemas.evidence import Evidence
from app.schemas.reports import FinalReport, Recommendation
from app.services.usage import UsageTracker
from app.tools import ToolContext


class BaselineReport(BaseModel):
    """Same shape as the multi-agent report, so outputs are directly comparable."""

    title: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    research_objective: str = Field(min_length=1)
    methodology: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    comparison_or_analysis: str = ""
    risks_and_limitations: list[str] = Field(default_factory=list)
    recommendation_statement: str = ""
    recommendation_rationale: str = ""
    recommendation_confidence: Confidence = Confidence.MEDIUM
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="Evidence ids supporting the findings",
    )


def run_baseline(
    user_request: str,
    index,
    usage: UsageTracker | None = None,
    run_id: str = "",
    store=None,
    max_iterations: int = 10,
) -> tuple[AgentOutcome[FinalReport], list[Evidence], float]:
    """Run the whole request through one agent. Returns report, evidence, wall seconds.

    ``max_iterations`` is higher than a single researcher's because this agent must cover every
    sub-question in one loop. Giving it fewer would manufacture the result.
    """
    started = time.perf_counter()

    # The baseline is granted the RESEARCHER permission set: search, extract, store, retrieve.
    # It is the widest set any single role holds, so the baseline is not constrained relative to
    # the specialists — it simply has no one checking its work.
    ctx = ToolContext(run_id=run_id, task_id="B1", research_question=user_request,
                      index=index, store=store)

    transcript, trace, error = tool_loop(
        agent_id=AgentId.RESEARCHER,   # permission identity; cost is attributed to "baseline"
        node="baseline_research", system=BASELINE, user=user_request,
        ctx=ctx, usage=usage, task_id="B1", max_iterations=max_iterations,
    )
    evidence = list(ctx.collected)

    if error is not None:
        return (
            AgentOutcome(agent_id=AgentId.BASELINE, ok=False, error=error, trace=trace,
                         tool_calls=ctx.calls),
            evidence,
            time.perf_counter() - started,
        )

    outcome = structured_step(
        agent_id=AgentId.BASELINE, node="baseline_report",
        system=BASELINE,
        user=(
            f"REQUEST:\n{user_request}\n\n"
            f"YOUR RESEARCH:\n{summarise_tool_results(transcript)}\n\n"
            f"You stored {len(evidence)} piece(s) of evidence "
            f"({', '.join(e.evidence_id for e in evidence) or 'none'}).\n\n"
            f"Now write the complete report."
        ),
        schema=BaselineReport, usage=usage, task_id="B1",
    )
    trace = trace + outcome.trace

    if outcome.failed or outcome.output is None:
        return (
            AgentOutcome(agent_id=AgentId.BASELINE, ok=False, error=outcome.error, trace=trace,
                         tool_calls=ctx.calls),
            evidence,
            time.perf_counter() - started,
        )

    draft = outcome.output
    # Deliberately NOT filtered to real ids. Whether the baseline cites evidence that does not
    # exist is one of the things Experiment 1 measures — silently dropping fabricated ids here
    # would hide exactly the failure the Fact-Checker was built to catch.
    cited = set(draft.cited_evidence_ids)
    report = FinalReport(
        title=draft.title,
        executive_summary=draft.executive_summary,
        research_objective=draft.research_objective,
        methodology=draft.methodology,
        key_findings=draft.key_findings,
        comparison_or_analysis=draft.comparison_or_analysis,
        risks_and_limitations=draft.risks_and_limitations,
        recommendation=(
            Recommendation(
                statement=draft.recommendation_statement,
                rationale=draft.recommendation_rationale,
                confidence=draft.recommendation_confidence,
                evidence_ids=draft.cited_evidence_ids,
            )
            if draft.recommendation_statement.strip() else None
        ),
        evidence_used=[e for e in evidence if e.evidence_id in cited],
        agent_id=AgentId.BASELINE,
    )
    return (
        AgentOutcome(agent_id=AgentId.BASELINE, ok=True, output=report, trace=trace,
                     tool_calls=ctx.calls, duration_seconds=outcome.duration_seconds),
        evidence,
        time.perf_counter() - started,
    )


def baseline_fabricated_citations(report: FinalReport, evidence: list[Evidence]) -> list[str]:
    """Ids the baseline cited that were never stored.

    The multi-agent system cannot produce these — the Fact-Checker's deterministic half catches
    them and the Critic's backstop rejects on them. Whether the baseline produces them is a
    direct, countable measure of what that machinery buys.
    """
    known = {e.evidence_id for e in evidence}
    cited: set[str] = set()
    if report.recommendation:
        cited.update(report.recommendation.evidence_ids)
    return sorted(cited - known)
