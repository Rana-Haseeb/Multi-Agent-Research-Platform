"""
Shared vocabulary for the whole workflow.

Every enum here is deliberately a ``str`` subclass so that values survive JSON serialisation
into the trace, the database, and LangGraph's checkpointer without custom encoders, while still
giving static typing and a closed set of valid values.

Closed sets matter more in a multi-agent system than in a single-agent one: an agent that
invents an agent name, a task status, or a confidence level produces a handoff that silently
routes nowhere. Pydantic rejects those at the schema boundary instead.
"""
from __future__ import annotations

from enum import Enum


class AgentId(str, Enum):
    """Every actor that can appear in the trace.

    ``SYSTEM`` covers the deterministic nodes (intake, evidence gate, budget guard). They are
    not agents — they make no model calls — but they do act on state, so the trace needs a name
    for them. Keeping them distinct is what lets the §29 "average agent calls" metric count
    only real model calls.
    """

    SUPERVISOR = "supervisor"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    FACT_CHECKER = "fact_checker"
    CRITIC = "critic"
    WRITER = "writer"
    SYSTEM = "system"
    BASELINE = "baseline"      # single-agent control path for Experiment 1

    @classmethod
    def llm_agents(cls) -> set[AgentId]:
        """The six specialists — everything that actually calls a model."""
        return {cls.SUPERVISOR, cls.RESEARCHER, cls.ANALYST,
                cls.FACT_CHECKER, cls.CRITIC, cls.WRITER}


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"          # dependency failed, so this task can never run

    @classmethod
    def terminal(cls) -> set[TaskStatus]:
        return {cls.COMPLETED, cls.FAILED, cls.SKIPPED}


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def weight(self) -> float:
        """Numeric weight for evidence-coverage scoring (§29)."""
        return {"high": 1.0, "medium": 0.6, "low": 0.3}[self.value]


class ClaimType(str, Enum):
    """§2 requires the Researcher to distinguish these four, not blur them together.

    This is the single most important distinction in the evidence model: a FACT and an
    ASSUMPTION look identical once they reach the Writer unless the type travels with them.
    """

    FACT = "fact"                # directly stated in a source
    CLAIM = "claim"              # asserted by a source, but the source is interested/unverified
    ASSUMPTION = "assumption"    # inferred by the agent, not stated anywhere
    MISSING = "missing"          # a known gap: the question was asked, nothing was found


class Severity(str, Enum):
    MAJOR = "major"              # blocks approval
    MINOR = "minor"              # noted, does not block


class WorkflowStatus(str, Enum):
    """Where the run is. Drives the dashboard and the §23 execution trace.

    The three ``AWAITING_*`` values are the human checkpoints. They are distinct states rather
    than a single ``paused`` flag so that the trace records *which* gate a run stopped at, which
    is what the §29 "human approval compliance" metric is computed from.
    """

    PENDING = "pending"
    ANALYSING_REQUEST = "analysing_request"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    PLANNING = "planning"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    RESEARCHING = "researching"
    ANALYSING_EVIDENCE = "analysing_evidence"
    FACT_CHECKING = "fact_checking"
    REVIEWING = "reviewing"
    REVISING = "revising"
    WRITING = "writing"
    AWAITING_FINAL_REVIEW = "awaiting_final_review"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"          # budget or revision cap forced termination

    def is_terminal(self) -> bool:
        return self in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.ABORTED}

    def is_waiting_for_human(self) -> bool:
        return self in {
            WorkflowStatus.AWAITING_CLARIFICATION,
            WorkflowStatus.AWAITING_PLAN_APPROVAL,
            WorkflowStatus.AWAITING_FINAL_REVIEW,
        }


class ReviewCriterion(str, Enum):
    """The six criteria the Critic must score against (§18).

    Fixed rather than free-form so that Critic output is comparable across runs — the §29
    "critic detection rate" needs the same axes every time.
    """

    EVIDENCE_COVERAGE = "evidence_coverage"
    LOGICAL_CONSISTENCY = "logical_consistency"
    COMPLETENESS = "completeness"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    CONTRADICTIONS = "contradictions"
    RELEVANCE = "relevance"


class HumanDecision(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
