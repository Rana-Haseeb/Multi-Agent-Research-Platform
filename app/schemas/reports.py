"""
Final report (§25) and execution trace records (§23).

§25 requires eight named sections and, crucially, that the report "clearly distinguish evidence,
analysis, and recommendation". That separation is modelled as separate fields rather than left
to prose discipline, so a reader — or a marker — can see which is which without parsing English.

§23 also says: store operational traces and structured outputs only, **not** private
chain-of-thought. :class:`TraceEvent` therefore records what an agent *did* (node, tool, timing,
tokens, handoff) and never why it thought it should.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.schemas.common import AgentId, Confidence
from app.schemas.evidence import Evidence


class Recommendation(BaseModel):
    """Kept separate from findings on purpose — this is the part that is *not* evidence."""

    statement: str = Field(min_length=1)
    rationale: str = ""
    confidence: Confidence = Confidence.MEDIUM
    conditions: list[str] = Field(
        default_factory=list, description="When this recommendation would change"
    )
    evidence_ids: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    """The §25 deliverable. Section names map 1:1 to the spec's required headings."""

    title: str
    executive_summary: str = Field(min_length=1)
    research_objective: str = Field(min_length=1)
    methodology: str = Field(min_length=1)
    key_findings: list[str] = Field(default_factory=list)
    comparison_or_analysis: str = ""
    risks_and_limitations: list[str] = Field(default_factory=list)
    recommendation: Recommendation | None = None
    evidence_used: list[Evidence] = Field(default_factory=list)

    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: AgentId = AgentId.WRITER
    revision: int = 0

    def to_markdown(self) -> str:
        """Export as Markdown (§25). PDF is optional and not implemented."""
        out: list[str] = [f"# {self.title}", ""]
        out += [f"*Generated {self.generated_at:%Y-%m-%d %H:%M UTC}*", ""]

        out += ["## Executive Summary", "", self.executive_summary, ""]
        out += ["## Research Objective", "", self.research_objective, ""]
        out += ["## Methodology", "", self.methodology, ""]

        out += ["## Key Findings", ""]
        out += [f"{i}. {f}" for i, f in enumerate(self.key_findings, 1)] or ["_None recorded._"]
        out += [""]

        if self.comparison_or_analysis:
            out += ["## Comparison and Analysis", "", self.comparison_or_analysis, ""]

        out += ["## Risks and Limitations", ""]
        out += [f"- {r}" for r in self.risks_and_limitations] or ["_None recorded._"]
        out += [""]

        out += ["## Recommendation", ""]
        if self.recommendation:
            r = self.recommendation
            out += [f"**{r.statement}**", ""]
            if r.rationale:
                out += [r.rationale, ""]
            out += [f"Confidence: **{r.confidence.value}**", ""]
            if r.conditions:
                out += ["This recommendation would change if:", ""]
                out += [f"- {c}" for c in r.conditions] + [""]
            if r.evidence_ids:
                out += [f"Supported by: {', '.join(r.evidence_ids)}", ""]
        else:
            out += ["_No recommendation was issued._", ""]

        out += ["## Evidence and References", ""]
        if self.evidence_used:
            out += ["| ID | Type | Confidence | Claim | Source |",
                    "|---|---|---|---|---|"]
            for e in self.evidence_used:
                claim = e.claim.replace("|", "\\|")
                title = e.source_title.replace("|", "\\|")
                out.append(
                    f"| {e.evidence_id} | {e.claim_type.value} | {e.confidence.value} "
                    f"| {claim} | {title} |"
                )
        else:
            out += ["_No evidence was recorded._"]
        out += [""]
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Execution tracing (§23)
# --------------------------------------------------------------------------- #
class TraceEvent(BaseModel):
    """One operational event. No chain-of-thought, ever.

    Written by every node, including the deterministic ones, so the trace shows the whole run
    and not just the model calls.
    """

    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: AgentId
    event: str = Field(description="node_start, node_end, tool_call, handoff, interrupt, error")
    node: str = ""
    task_id: str = ""
    detail: str = Field(default="", description="Short operational summary — never reasoning")
    duration_seconds: float = 0.0
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    def line(self) -> str:
        bits = [f"{self.at:%H:%M:%S}", f"{self.agent_id.value:<12}", f"{self.event:<11}"]
        if self.node:
            bits.append(self.node)
        if self.task_id:
            bits.append(f"[{self.task_id}]")
        if self.duration_seconds:
            bits.append(f"{self.duration_seconds:.2f}s")
        if self.detail:
            bits.append(f"— {self.detail}")
        return " ".join(bits)


class ErrorRecord(BaseModel):
    """A handled failure (§22). Recorded rather than raised, so the run can continue degraded."""

    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_id: AgentId
    node: str = ""
    task_id: str = ""
    kind: str = Field(description="tool_failure, invalid_output, timeout, budget, api_failure, ...")
    message: str
    recovered: bool = False
    action_taken: str = Field(default="", description="retry, fallback_provider, skip, abort")
