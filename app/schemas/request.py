"""
Request analysis (§9) and clarification (§10).

The Supervisor's first job is to turn a sentence of prose into a structured objective. This is
where the system either commits to the right question or quietly commits to the wrong one, so
the schema forces the decomposition to be explicit rather than implied.

``needs_clarification`` deserves its own note. §10 says the system must not "make major
assumptions silently", and the cheapest place to enforce that is here — *before* the fan-out
spends N model calls researching a misread objective.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import HumanDecision


class ClarificationQuestion(BaseModel):
    question: str = Field(min_length=1)
    why_it_matters: str = Field(
        min_length=1,
        description="What would change in the research plan depending on the answer",
    )


class Clarification(BaseModel):
    """A question asked of the user and the answer they gave."""

    question: str
    answer: str
    answered_at: str = ""


class RequestBrief(BaseModel):
    """Structured form of the user's request — the §9 output schema.

    Produced by the Supervisor, readable by every agent, writable by nobody else.
    """

    objective: str = Field(min_length=1, description="The single main objective, one sentence")
    sub_questions: list[str] = Field(
        default_factory=list,
        description="Specific answerable research questions the objective decomposes into",
    )
    evaluation_criteria: list[str] = Field(
        default_factory=list,
        description="Dimensions on which options should be compared",
    )
    deliverable: str = Field(default="", description="What the user should receive")
    constraints: list[str] = Field(default_factory=list)
    options_under_comparison: list[str] = Field(
        default_factory=list,
        description="Named alternatives, when the request is a comparison. Drives the fan-out.",
    )
    time_horizon: str | None = None
    missing_information: list[str] = Field(default_factory=list)

    needs_clarification: bool = Field(
        default=False,
        description="True only when the request cannot be planned without more input",
    )
    clarifying_questions: list[ClarificationQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _clarification_is_actionable(self) -> RequestBrief:
        """Flagging ambiguity without saying what's missing is useless to the user.

        A brief that stops the workflow must also tell the user what to supply, otherwise the
        human checkpoint is a dead end. Enforced structurally so no prompt wording can bypass it.
        """
        if self.needs_clarification and not self.clarifying_questions:
            raise ValueError(
                "needs_clarification=True requires at least one clarifying question"
            )
        if not self.needs_clarification and not self.sub_questions:
            raise ValueError(
                "a brief that does not need clarification must have at least one sub-question"
            )
        return self

    def is_comparison(self) -> bool:
        return len(self.options_under_comparison) >= 2


class HumanCheckpointResponse(BaseModel):
    """What comes back from an ``interrupt()`` (§20).

    ``edited_payload`` carries the user's modifications when they choose EDIT — for the plan
    gate that is a revised task list, for the final gate a revised recommendation note.
    """

    decision: HumanDecision
    note: str = ""
    edited_payload: dict | None = None

    @model_validator(mode="after")
    def _edit_carries_changes(self) -> HumanCheckpointResponse:
        if self.decision is HumanDecision.EDIT and not (self.edited_payload or self.note):
            raise ValueError("decision=edit requires edited_payload or a note describing the edit")
        return self
