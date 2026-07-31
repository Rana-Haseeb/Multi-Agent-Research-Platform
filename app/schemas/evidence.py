"""
The evidence model (§15) — the unit of truth for the whole system.

Everything downstream of research is *about* evidence: the Analyst may only reason from it, the
Fact-Checker verifies claims against it, the Critic judges coverage of it, and the Writer may
only cite it. So this schema is the one place where getting the fields wrong forces a rewrite.

Two design choices worth defending in review:

1. **``claim`` and ``supporting_text`` are separate fields.** ``claim`` is the agent's one-line
   assertion; ``supporting_text`` is the verbatim source span that backs it. Collapsing them
   would make fact-checking impossible — you cannot ask "does the source actually say this?"
   if the agent's paraphrase has already replaced the source.

2. **``claim_type`` is mandatory.** A FACT and an ASSUMPTION are indistinguishable by the time
   they reach a report unless the distinction is carried in the data. §2 asks for it explicitly.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ClaimType, Confidence

EVIDENCE_ID_RE = re.compile(r"^E\d+$")


class Evidence(BaseModel):
    """One recorded finding. Immutable once stored.

    Field set is the §15 minimum plus ``claim_type`` and ``task_id``, which are needed to
    distinguish fact from assumption and to attribute a finding to the task that produced it.
    """

    evidence_id: str = Field(description="Stable id of the form E1, E2, ...")
    claim: str = Field(min_length=1, description="One-line assertion this evidence supports")
    supporting_text: str = Field(
        min_length=1,
        description="Verbatim span or close summary from the source that backs the claim",
    )
    source_id: str = Field(min_length=1, description="Corpus document id or URL")
    source_title: str = Field(min_length=1)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    research_question: str = Field(min_length=1, description="The sub-question this answers")
    confidence: Confidence
    claim_type: ClaimType
    agent_id: str = Field(description="Which agent recorded it")
    task_id: str = Field(description="Which planned task produced it")

    @field_validator("evidence_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not EVIDENCE_ID_RE.match(v):
            raise ValueError(f"evidence_id must look like E1, E2, ... (got {v!r})")
        return v

    def index_line(self) -> str:
        """Compact one-line form for the Critic's evidence index (§21 context budget).

        The Critic needs to know what evidence *exists* to judge coverage; it does not need
        every supporting span. Sending the index instead of the bodies is the single largest
        token saving in the system, and Experiment 4 measures it.
        """
        return (
            f"{self.evidence_id} | {self.claim_type.value} | conf={self.confidence.value} "
            f"| {self.claim} | src: {self.source_title}"
        )

    def cite(self, snippet_chars: int) -> str:
        """Fuller form for the Analyst, with supporting text truncated to a budget."""
        text = self.supporting_text
        if len(text) > snippet_chars:
            text = text[: snippet_chars - 1].rstrip() + "…"
        return (
            f"[{self.evidence_id}] ({self.claim_type.value}, {self.confidence.value} confidence) "
            f"{self.claim}\n    \"{text}\"\n    — {self.source_title} ({self.source_id})"
        )


class EvidenceGap(BaseModel):
    """A question that was asked and not answered.

    Recorded as first-class data rather than silence. A silent gap becomes an invented answer
    somewhere downstream; a recorded one becomes a line in the report's Limitations section and
    a signal the evidence gate can route on.
    """

    research_question: str
    reason: str = Field(description="Why nothing was found — no results, off-corpus, ambiguous")
    searched_queries: list[str] = Field(default_factory=list)


class EvidenceIndex(BaseModel):
    """A read-only view over collected evidence, grouped for context assembly."""

    items: list[Evidence] = Field(default_factory=list)

    def ids(self) -> set[str]:
        return {e.evidence_id for e in self.items}

    def by_question(self) -> dict[str, list[Evidence]]:
        out: dict[str, list[Evidence]] = {}
        for e in self.items:
            out.setdefault(e.research_question, []).append(e)
        return out

    def coverage(self, questions: list[str]) -> dict[str, float]:
        """Confidence-weighted coverage score per research question.

        Deliberately *not* a simple "has any evidence" boolean: three low-confidence assumptions
        are not coverage. Weights come from :meth:`Confidence.weight`, and ASSUMPTION-typed
        items contribute nothing, because an assumption is the absence of evidence.
        """
        grouped = self.by_question()
        scores: dict[str, float] = {}
        for q in questions:
            items = grouped.get(q, [])
            scores[q] = sum(
                e.confidence.weight()
                for e in items
                if e.claim_type in (ClaimType.FACT, ClaimType.CLAIM)
            )
        return scores

    def unresolved(self, questions: list[str], threshold: float = 1.0) -> list[str]:
        """Questions whose weighted coverage falls below ``threshold``."""
        return [q for q, s in self.coverage(questions).items() if s < threshold]
