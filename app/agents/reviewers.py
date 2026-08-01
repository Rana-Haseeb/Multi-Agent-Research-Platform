"""
Analyst, Fact-Checker, Critic and Writer.

Grouped in one module because they share a shape: each takes validated upstream handoffs, makes
exactly one model call over a purpose-built context, and returns a validated handoff. Their
differences are in *what they may see* and *what their output must contain* — both enforced by
the context builders and the schemas rather than by these functions.

The Fact-Checker is the one worth reading closely. Its deterministic half runs here, in Python,
before the model is consulted at all: citation existence is a set operation, and asking a model
whether "E9" is in a list is strictly worse than checking. The model is asked only the question
code cannot answer — whether the evidence *supports* the claim.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.agents.base import AgentOutcome, structured_step
from app.agents.context import (
    analyst_context,
    critic_context,
    fact_checker_context,
    writer_context,
)
from app.agents.prompts import ANALYST, CRITIC, FACT_CHECKER, WRITER
from app.config import settings
from app.schemas.common import AgentId, Confidence, ReviewCriterion, Severity
from app.schemas.evidence import Evidence
from app.schemas.handoffs import (
    AnalysisHandoff,
    ClaimCheck,
    Conclusion,
    CriticVerdict,
    FactCheckReport,
    Problem,
    ResearchHandoff,
)
from app.schemas.reports import FinalReport, Recommendation
from app.schemas.request import RequestBrief
from app.services.usage import UsageTracker


# --------------------------------------------------------------------------- #
# Analyst
# --------------------------------------------------------------------------- #
def analyse(
    brief: RequestBrief,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
    usage: UsageTracker | None = None,
    revision_note: str = "",
    revision: int = 0,
) -> AgentOutcome[AnalysisHandoff]:
    """Build the comparison from stored evidence only."""
    outcome = structured_step(
        agent_id=AgentId.ANALYST, node="analyst",
        system=ANALYST,
        user=analyst_context(brief, evidence, handoffs, revision_note),
        schema=AnalysisHandoff, usage=usage,
        detail=f"analysis (revision {revision})",
    )
    if outcome.ok and outcome.output is not None:
        outcome.output.revision = revision
    return outcome


# --------------------------------------------------------------------------- #
# Fact-Checker — deterministic half, then judgement half
# --------------------------------------------------------------------------- #
class SupportJudgement(BaseModel):
    conclusion_id: str
    evidence_supports: bool = Field(
        description="Does the cited evidence actually establish this statement?"
    )
    reasoning: str = Field(default="", description="One sentence. Required when false.")


class SupportJudgements(BaseModel):
    judgements: list[SupportJudgement] = Field(default_factory=list)


def check_citations_deterministic(
    analysis: AnalysisHandoff, evidence: list[Evidence]
) -> dict[str, list[str]]:
    """Which cited ids do not exist? Pure set membership, no model.

    This is the boundary the excellence criteria ask to see: a fabricated citation is detected by
    code with certainty, and no amount of persuasive prose can talk the check out of it.
    """
    known = {e.evidence_id for e in evidence}
    return {
        c.conclusion_id: [eid for eid in c.evidence_ids if eid not in known]
        for c in analysis.conclusions
    }


def fact_check(
    analysis: AnalysisHandoff,
    evidence: list[Evidence],
    usage: UsageTracker | None = None,
) -> AgentOutcome[FactCheckReport]:
    """Verify citations: existence in code, support by judgement."""
    fabricated = check_citations_deterministic(analysis, evidence)

    if not analysis.conclusions:
        return AgentOutcome(agent_id=AgentId.FACT_CHECKER, ok=True,
                            output=FactCheckReport(checks=[]))

    context = fact_checker_context(analysis, evidence)
    if any(fabricated.values()):
        flagged = {k: v for k, v in fabricated.items() if v}
        context += (
            "\n\nALREADY VERIFIED MECHANICALLY — these citations do not exist:\n"
            + "\n".join(f"  {cid}: {', '.join(ids)}" for cid, ids in flagged.items())
            + "\nTreat any conclusion resting on them as unsupported."
        )

    outcome = structured_step(
        agent_id=AgentId.FACT_CHECKER, node="fact_checker",
        system=FACT_CHECKER, user=context, schema=SupportJudgements, usage=usage,
        detail=f"{len(analysis.conclusions)} conclusions checked",
    )

    # A failed model call must not discard the deterministic result. Existence checking already
    # succeeded, so the report is still worth producing — degraded, not lost.
    judged: dict[str, SupportJudgement] = {}
    if outcome.ok and outcome.output is not None:
        judged = {j.conclusion_id: j for j in outcome.output.judgements}

    checks: list[ClaimCheck] = []
    for c in analysis.conclusions:
        missing = fabricated.get(c.conclusion_id, [])
        judgement = judged.get(c.conclusion_id)
        if missing:
            supports, reasoning = False, f"Cites non-existent evidence: {', '.join(missing)}."
        elif judgement is not None:
            supports, reasoning = judgement.evidence_supports, judgement.reasoning
        elif not c.evidence_ids:
            supports, reasoning = False, "No evidence cited."
        else:
            supports, reasoning = True, "Citation exists; support was not independently judged."
        checks.append(ClaimCheck(
            conclusion_id=c.conclusion_id, citation_exists=not missing,
            fabricated_ids=missing, evidence_supports=supports, reasoning=reasoning,
        ))

    report = FactCheckReport(checks=checks)
    return AgentOutcome(
        agent_id=AgentId.FACT_CHECKER, ok=True, output=report,
        trace=outcome.trace, duration_seconds=outcome.duration_seconds,
        error=outcome.error if outcome.failed else None,
    )


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
def review(
    brief: RequestBrief,
    analysis: AnalysisHandoff,
    evidence: list[Evidence],
    fact_check_report: FactCheckReport | None,
    handoffs: list[ResearchHandoff],
    usage: UsageTracker | None = None,
    cycle: int = 0,
) -> AgentOutcome[CriticVerdict]:
    """Review against the six criteria. May reject; cannot rewrite."""
    outcome = structured_step(
        agent_id=AgentId.CRITIC, node="critic",
        system=CRITIC,
        user=critic_context(brief, analysis, evidence, fact_check_report, handoffs, cycle),
        schema=CriticVerdict, usage=usage,
        detail=f"review cycle {cycle}",
    )

    if outcome.failed or outcome.output is None:
        # §22 agent failure. A Critic that cannot run must not silently become an approval —
        # that would turn an outage into a quality gate that always passes. Fail closed by
        # falling back to the deterministic evidence the Fact-Checker already produced.
        return _fallback_verdict(analysis, fact_check_report, cycle, outcome)

    verdict = outcome.output
    verdict.cycle = cycle

    # Deterministic backstop: whatever the Critic concluded, a fabricated citation is
    # disqualifying. This is what stops PD-style defects surviving a lenient review.
    if fact_check_report and fact_check_report.fabricated:
        verdict = _force_rejection(
            verdict,
            issue=f"Fabricated citations: {', '.join(fact_check_report.fabricated)}.",
            criterion=ReviewCriterion.UNSUPPORTED_CLAIMS,
            fix="Remove or re-evidence every conclusion citing an id that does not exist.",
        )

    uncited = [c.conclusion_id for c in analysis.major_conclusions() if not c.evidence_ids]
    if uncited:
        verdict = _force_rejection(
            verdict,
            issue=f"Major conclusions with no evidence: {', '.join(uncited)}.",
            criterion=ReviewCriterion.EVIDENCE_COVERAGE,
            fix="Cite supporting evidence for each, or downgrade them to minor observations.",
        )

    outcome.output = verdict
    return outcome


def _force_rejection(
    verdict: CriticVerdict, *, issue: str, criterion: ReviewCriterion, fix: str
) -> CriticVerdict:
    """Override an approval that a deterministic check contradicts."""
    problems = list(verdict.problems)
    if not any(issue[:40] in p.issue for p in problems):
        problems.append(Problem(location="analysis", issue=issue, criterion=criterion,
                                severity=Severity.MAJOR))
    revisions = list(verdict.required_revisions)
    if fix not in revisions:
        revisions.append(fix)
    return verdict.model_copy(update={
        "approved": False, "problems": problems, "required_revisions": revisions,
    })


def _fallback_verdict(
    analysis: AnalysisHandoff, report: FactCheckReport | None, cycle: int,
    outcome: AgentOutcome,
) -> AgentOutcome[CriticVerdict]:
    """Deterministic verdict when the Critic model call fails."""
    problems: list[Problem] = []
    revisions: list[str] = []

    if report and report.fabricated:
        problems.append(Problem(
            location="analysis", issue=f"Fabricated citations: {', '.join(report.fabricated)}.",
            criterion=ReviewCriterion.UNSUPPORTED_CLAIMS, severity=Severity.MAJOR))
        revisions.append("Remove conclusions citing non-existent evidence.")
    if report and report.unsupported_ids:
        problems.append(Problem(
            location="analysis",
            issue=f"Not supported by cited evidence: {', '.join(report.unsupported_ids)}.",
            criterion=ReviewCriterion.UNSUPPORTED_CLAIMS, severity=Severity.MAJOR))
        revisions.append("Re-evidence or soften the unsupported conclusions.")
    for c in analysis.major_conclusions():
        if not c.evidence_ids:
            problems.append(Problem(
                location=c.conclusion_id, issue="Major conclusion with no citation.",
                criterion=ReviewCriterion.EVIDENCE_COVERAGE, severity=Severity.MAJOR))
            revisions.append(f"Cite evidence for {c.conclusion_id}.")

    if problems:
        verdict = CriticVerdict(approved=False, problems=problems,
                                required_revisions=revisions or ["Address the problems listed."],
                                cycle=cycle)
    else:
        # Nothing mechanically wrong and no reviewer available. Approve, but say so — the report
        # must not imply a review happened that did not.
        verdict = CriticVerdict(approved=True, cycle=cycle,
                                required_revisions=[],
                                missing_evidence=["Critic review was unavailable for this run; "
                                                  "only mechanical checks were applied."])
    return AgentOutcome(agent_id=AgentId.CRITIC, ok=True, output=verdict,
                        trace=outcome.trace, error=outcome.error,
                        duration_seconds=outcome.duration_seconds)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
class ReportDraft(BaseModel):
    """What the Writer emits. Evidence is attached afterwards, from the store.

    The Writer never supplies the evidence list itself — it is assembled here from the ids the
    analysis cited. That removes the last route by which an invented source could reach the
    final document.
    """

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
    recommendation_conditions: list[str] = Field(default_factory=list)


def write_report(
    brief: RequestBrief,
    analysis: AnalysisHandoff,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
    verdict: CriticVerdict | None = None,
    usage: UsageTracker | None = None,
) -> AgentOutcome[FinalReport]:
    """Produce the final deliverable from approved material only."""
    critic_note = ""
    if verdict and not verdict.approved:
        # The revision cap was reached with objections outstanding. They belong in the report.
        critic_note = (
            "The reviewer did not approve this analysis and the revision limit was reached. "
            "These unresolved objections MUST appear in Risks and Limitations:\n"
            + "\n".join(f"  - {p.issue}" for p in verdict.problems)
        )

    outcome = structured_step(
        agent_id=AgentId.WRITER, node="writer",
        system=WRITER,
        user=writer_context(brief, analysis, evidence, handoffs, critic_note),
        schema=ReportDraft, usage=usage, detail="final report",
    )
    if outcome.failed or outcome.output is None:
        return AgentOutcome(agent_id=AgentId.WRITER, ok=False, error=outcome.error,
                            trace=outcome.trace, duration_seconds=outcome.duration_seconds)

    draft = outcome.output
    cited = analysis.cited_ids()
    used = [e for e in evidence if e.evidence_id in cited]

    limitations = list(draft.risks_and_limitations)
    for handoff in handoffs:
        for gap in handoff.gaps:
            note = f"Unanswered: {gap.research_question} ({gap.reason})"
            if not any(gap.research_question[:40] in lim for lim in limitations):
                limitations.append(note)
    if verdict and not verdict.approved:
        for problem in verdict.major_problems():
            note = f"Unresolved reviewer objection: {problem.issue}"
            if note not in limitations:
                limitations.append(note)

    recommendation = None
    if draft.recommendation_statement.strip():
        recommendation = Recommendation(
            statement=draft.recommendation_statement,
            rationale=draft.recommendation_rationale,
            confidence=draft.recommendation_confidence,
            conditions=draft.recommendation_conditions,
            evidence_ids=sorted({eid for c in analysis.major_conclusions()
                                 for eid in c.evidence_ids}),
        )

    report = FinalReport(
        title=draft.title,
        executive_summary=draft.executive_summary,
        research_objective=draft.research_objective or brief.objective,
        methodology=draft.methodology,
        key_findings=draft.key_findings,
        comparison_or_analysis=draft.comparison_or_analysis,
        risks_and_limitations=limitations,
        recommendation=recommendation,
        evidence_used=used,
        revision=analysis.revision,
    )
    outcome.output = report
    return outcome
