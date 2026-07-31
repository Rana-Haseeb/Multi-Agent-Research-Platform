"""
Agent-to-agent handoff contracts (§17).

Every transfer between agents is a validated Pydantic object, never a chat transcript. That is
the difference between a multi-agent *system* and five prompts in a trench coat: if the Analyst
cannot produce a conclusion without attaching evidence ids, then "cite your sources" stops being
a request in a prompt and becomes a property of the type.

The contracts, in workflow order::

    Researcher  -> Analyst        ResearchHandoff
    Analyst     -> FactChecker    AnalysisHandoff
    FactChecker -> Critic         FactCheckReport
    Critic      -> Supervisor     CriticVerdict
    Supervisor  -> Writer         (AnalysisHandoff + CriticVerdict, already validated)
    Writer      -> User           FinalReport      (see app.schemas.reports)

Each carries exactly what §17 requires and, importantly, **not the sender's full context**.
The Analyst never sees the Researcher's raw search results, only its findings and evidence ids.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import AgentId, Confidence, ReviewCriterion, Severity
from app.schemas.evidence import EvidenceGap


# --------------------------------------------------------------------------- #
# Researcher -> Analyst
# --------------------------------------------------------------------------- #
class ResearchHandoff(BaseModel):
    """§17: research question, findings, evidence ids, confidence, known gaps.

    One of these per research task. Because researchers run in parallel, these accumulate into
    state through an additive reducer and are never overwritten.
    """

    task_id: str
    research_question: str = Field(min_length=1)
    findings: str = Field(min_length=1, description="Prose summary of what was found")
    evidence_ids: list[str] = Field(
        default_factory=list, description="Ids of Evidence records this researcher stored"
    )
    confidence: Confidence
    gaps: list[EvidenceGap] = Field(default_factory=list)
    sources_consulted: list[str] = Field(default_factory=list)
    agent_id: AgentId = AgentId.RESEARCHER

    @model_validator(mode="after")
    def _empty_research_is_declared_not_implied(self) -> ResearchHandoff:
        """A researcher that found nothing must say so explicitly.

        §22 lists "empty research results" as a failure to handle. The dangerous version is not
        an empty handoff — it is a confident-sounding ``findings`` paragraph with no evidence
        behind it, which reads downstream as substance. Requiring a declared gap makes the
        absence visible to the evidence gate and to the report's Limitations section.
        """
        if not self.evidence_ids and not self.gaps:
            raise ValueError(
                f"task {self.task_id}: a handoff with no evidence_ids must declare at least "
                f"one gap explaining why nothing was found"
            )
        if not self.evidence_ids and self.confidence is Confidence.HIGH:
            raise ValueError(
                f"task {self.task_id}: cannot report high confidence with no evidence"
            )
        return self


# --------------------------------------------------------------------------- #
# Analyst -> FactChecker / Critic
# --------------------------------------------------------------------------- #
class Conclusion(BaseModel):
    """A single analytical conclusion, bound to the evidence that supports it."""

    conclusion_id: str = Field(description="C1, C2, ...")
    statement: str = Field(min_length=1)
    evidence_ids: list[str] = Field(
        default_factory=list, description="Evidence ids supporting this specific statement"
    )
    confidence: Confidence
    is_major: bool = Field(
        default=False,
        description="True if the recommendation depends on it. Drives the §29 "
                    "'unsupported major claims' metric.",
    )

    @model_validator(mode="after")
    def _major_conclusions_need_evidence(self) -> Conclusion:
        """The §29 target is "unsupported major claims below 10%".

        Enforcing it in the type means the Analyst structurally cannot emit an uncited major
        conclusion; the metric then measures the Fact-Checker's residual catch rate rather than
        the Analyst's carelessness. Minor observations may stand uncited.
        """
        if self.is_major and not self.evidence_ids:
            raise ValueError(
                f"{self.conclusion_id}: a major conclusion must cite at least one evidence id"
            )
        return self


class ComparisonRow(BaseModel):
    """One option scored across the brief's evaluation criteria."""

    option: str
    scores: dict[str, str] = Field(
        default_factory=dict, description="criterion -> short assessment"
    )
    evidence_ids: list[str] = Field(default_factory=list)


class AnalysisHandoff(BaseModel):
    """§17: analysis, conclusions, evidence references, assumptions."""

    summary: str = Field(min_length=1)
    conclusions: list[Conclusion] = Field(min_length=1)
    comparison: list[ComparisonRow] = Field(default_factory=list)
    trade_offs: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(
        default_factory=list,
        description="Anything asserted that evidence does not establish — stated, not hidden",
    )
    evidence_ids_used: list[str] = Field(default_factory=list)
    revision: int = Field(default=0, description="Bumped each time the Critic sends it back")
    agent_id: AgentId = AgentId.ANALYST

    def major_conclusions(self) -> list[Conclusion]:
        return [c for c in self.conclusions if c.is_major]

    def cited_ids(self) -> set[str]:
        ids = set(self.evidence_ids_used)
        for c in self.conclusions:
            ids.update(c.evidence_ids)
        for row in self.comparison:
            ids.update(row.evidence_ids)
        return ids


# --------------------------------------------------------------------------- #
# FactChecker -> Critic
# --------------------------------------------------------------------------- #
class ClaimCheck(BaseModel):
    """Verdict on one conclusion.

    ``citation_exists`` is decided by code — set membership against the evidence store, no model
    involved. ``evidence_supports`` is the judgement call and is the only part a model decides.
    Keeping them as separate fields is what makes the deterministic/agentic boundary auditable
    rather than a claim in the README.
    """

    conclusion_id: str
    citation_exists: bool = Field(description="Deterministic: do all cited ids exist?")
    fabricated_ids: list[str] = Field(
        default_factory=list, description="Cited ids with no matching evidence record"
    )
    evidence_supports: bool = Field(description="Model judgement: does the evidence back it?")
    reasoning: str = ""


class FactCheckReport(BaseModel):
    """Aggregate citation integrity report."""

    checks: list[ClaimCheck] = Field(default_factory=list)
    agent_id: AgentId = AgentId.FACT_CHECKER

    @property
    def fabricated(self) -> list[str]:
        return sorted({fid for c in self.checks for fid in c.fabricated_ids})

    @property
    def unsupported_ids(self) -> list[str]:
        return [c.conclusion_id for c in self.checks if not c.evidence_supports]

    def unsupported_rate(self) -> float:
        """Fraction of checked conclusions that evidence does not support.

        Returns 0.0 for an empty report — with nothing checked there is nothing unsupported.
        Callers that need "was anything actually verified?" must check ``len(checks)``; the
        Week 3 post-mortem (§7.3) is precisely about a metric that scored perfectly on an
        empty set.
        """
        if not self.checks:
            return 0.0
        return len(self.unsupported_ids) / len(self.checks)

    def is_clean(self) -> bool:
        return bool(self.checks) and not self.fabricated and not self.unsupported_ids


# --------------------------------------------------------------------------- #
# Critic -> Supervisor
# --------------------------------------------------------------------------- #
class Problem(BaseModel):
    location: str = Field(description="Which conclusion or section")
    issue: str = Field(min_length=1)
    criterion: ReviewCriterion
    severity: Severity


class CriticVerdict(BaseModel):
    """§17: approved or rejected, problems found, missing evidence, required revisions."""

    approved: bool
    problems: list[Problem] = Field(default_factory=list)
    missing_evidence: list[str] = Field(
        default_factory=list, description="Questions that still need researching"
    )
    required_revisions: list[str] = Field(
        default_factory=list, description="Specific, actionable instructions for the Analyst"
    )
    scores: dict[ReviewCriterion, int] = Field(
        default_factory=dict, description="1-5 per criterion"
    )
    needs_more_research: bool = Field(
        default=False,
        description="True routes back to research rather than re-analysis",
    )
    cycle: int = Field(default=0, description="Which revision cycle produced this verdict")
    agent_id: AgentId = AgentId.CRITIC

    @model_validator(mode="after")
    def _rejection_must_be_actionable(self) -> CriticVerdict:
        """A rejection with no instructions cannot be acted on.

        §18 says the Critic evaluates rather than rewrites — but a verdict that only says "not
        good enough" guarantees the next cycle repeats the same mistake and burns the revision
        budget for nothing. Rejection therefore requires a stated problem and a stated fix.
        """
        if not self.approved and not self.problems:
            raise ValueError("a rejection must list at least one problem")
        if not self.approved and not (self.required_revisions or self.missing_evidence):
            raise ValueError(
                "a rejection must state required_revisions or missing_evidence so the next "
                "cycle has something concrete to act on"
            )
        return self

    def major_problems(self) -> list[Problem]:
        return [p for p in self.problems if p.severity is Severity.MAJOR]
