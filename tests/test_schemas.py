"""
Phase 1 tests: the schemas and the shared state.

These are written to prove the contracts **reject** bad input, not merely accept good input.
A validator that has never been seen to fail is indistinguishable from no validator, which is
the Week 3 post-mortem's central lesson (§7.3).
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from pydantic import ValidationError

from app.graph.state import (
    FIELD_PERMISSIONS,
    PARALLEL_WRITE_CHANNELS,
    WorkflowState,
    evidence_id_for,
    initial_state,
    merge_status,
    permissions_table,
)
from app.schemas.common import (
    AgentId,
    ClaimType,
    Confidence,
    ReviewCriterion,
    Severity,
    TaskStatus,
    WorkflowStatus,
)
from app.schemas.evidence import Evidence, EvidenceIndex
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
from app.schemas.request import ClarificationQuestion, RequestBrief
from app.schemas.tasks import Task, TaskPlan


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_evidence(eid="E101", q="What is X?", conf=Confidence.HIGH, ctype=ClaimType.FACT):
    return Evidence(
        evidence_id=eid, claim="X is a thing", supporting_text="The docs say X is a thing.",
        source_id="doc-1", source_title="X Documentation", research_question=q,
        confidence=conf, claim_type=ctype, agent_id="researcher", task_id="R1",
    )


def research_task(tid="R1", q="What is X?", deps=None):
    return Task(task_id=tid, description=f"Research {q}", assigned_agent=AgentId.RESEARCHER,
                research_question=q, depends_on=deps or [])


# --------------------------------------------------------------------------- #
# Evidence (§15)
# --------------------------------------------------------------------------- #
def test_evidence_id_format_is_enforced():
    with pytest.raises(ValidationError):
        make_evidence(eid="not-an-id")


def test_evidence_requires_supporting_text():
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id="E1", claim="c", supporting_text="", source_id="d",
            source_title="t", research_question="q", confidence=Confidence.HIGH,
            claim_type=ClaimType.FACT, agent_id="researcher", task_id="R1",
        )


def test_coverage_ignores_assumptions():
    """An assumption is the absence of evidence, so it must not count toward coverage."""
    idx = EvidenceIndex(items=[make_evidence("E101", ctype=ClaimType.ASSUMPTION)])
    assert idx.coverage(["What is X?"])["What is X?"] == 0.0
    assert idx.unresolved(["What is X?"]) == ["What is X?"]


def test_coverage_is_confidence_weighted():
    idx = EvidenceIndex(items=[
        make_evidence("E101", conf=Confidence.LOW),
        make_evidence("E102", conf=Confidence.LOW),
    ])
    # two low-confidence findings (0.3 each) must NOT clear the 1.0 threshold
    assert idx.unresolved(["What is X?"]) == ["What is X?"]
    idx.items.append(make_evidence("E103", conf=Confidence.HIGH))
    assert idx.unresolved(["What is X?"]) == []


def test_index_line_omits_supporting_text():
    """The Critic's context budget depends on the index being small."""
    e = make_evidence()
    assert e.supporting_text not in e.index_line()
    assert e.evidence_id in e.index_line()


def test_cite_truncates_to_budget():
    e = make_evidence()
    e.supporting_text = "x" * 5000
    assert len(e.cite(snippet_chars=100)) < 400


# --------------------------------------------------------------------------- #
# RequestBrief (§9, §10)
# --------------------------------------------------------------------------- #
def test_clarification_without_questions_is_rejected():
    with pytest.raises(ValidationError, match="clarifying question"):
        RequestBrief(objective="o", needs_clarification=True)


def test_clarification_with_questions_is_accepted():
    b = RequestBrief(
        objective="o", needs_clarification=True,
        clarifying_questions=[ClarificationQuestion(question="Which language?",
                                                    why_it_matters="changes the candidates")],
    )
    assert b.needs_clarification


def test_plannable_brief_needs_sub_questions():
    with pytest.raises(ValidationError, match="sub-question"):
        RequestBrief(objective="o", needs_clarification=False)


# --------------------------------------------------------------------------- #
# TaskPlan (§11, §22)
# --------------------------------------------------------------------------- #
def test_researcher_task_without_question_is_rejected():
    with pytest.raises(ValidationError, match="research_question"):
        Task(task_id="R1", description="d", assigned_agent=AgentId.RESEARCHER)


def test_duplicate_task_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate task_id"):
        TaskPlan(tasks=[research_task("R1"), research_task("R1", q="other")])


def test_dependency_on_unknown_task_rejected():
    with pytest.raises(ValidationError, match="unknown task"):
        TaskPlan(tasks=[research_task("R1", deps=["R9"])])


def test_dependency_cycle_is_detected():
    """A cycle would deadlock the graph with no error, so it must fail at construction."""
    with pytest.raises(ValidationError, match="cycle"):
        TaskPlan(tasks=[
            research_task("R1", q="a", deps=["R2"]),
            research_task("R2", q="b", deps=["R1"]),
        ])


def test_self_dependency_is_detected():
    with pytest.raises(ValidationError, match="depends on itself"):
        TaskPlan(tasks=[research_task("R1", deps=["R1"])])


def test_ready_respects_dependencies():
    plan = TaskPlan(tasks=[
        research_task("R1", q="a"),
        Task(task_id="A1", description="analyse", assigned_agent=AgentId.ANALYST,
             depends_on=["R1"]),
    ])
    assert [t.task_id for t in plan.ready()] == ["R1"]
    plan.by_id("R1").status = TaskStatus.COMPLETED
    assert [t.task_id for t in plan.ready()] == ["A1"]


def test_blocked_detects_dead_dependencies():
    plan = TaskPlan(tasks=[
        research_task("R1", q="a"),
        Task(task_id="A1", description="analyse", assigned_agent=AgentId.ANALYST,
             depends_on=["R1"]),
    ])
    plan.by_id("R1").status = TaskStatus.FAILED
    assert [t.task_id for t in plan.blocked()] == ["A1"]
    assert plan.ready() == []


def test_duplicate_research_questions_detected():
    plan = TaskPlan(tasks=[
        research_task("R1", q="What are LangGraph's state features?"),
        research_task("R2", q="What are LangGraph's state features?"),
        research_task("R3", q="How much does CrewAI cost?"),
    ])
    groups = plan.duplicate_groups()
    assert len(groups) == 1 and set(groups[0]) == {"R1", "R2"}


def test_distinct_questions_are_not_flagged_duplicate():
    plan = TaskPlan(tasks=[
        research_task("R1", q="What are LangGraph's state features?"),
        research_task("R2", q="How much does CrewAI cost?"),
    ])
    assert plan.duplicate_groups() == []


# --------------------------------------------------------------------------- #
# Handoffs (§17)
# --------------------------------------------------------------------------- #
def test_empty_research_must_declare_a_gap():
    with pytest.raises(ValidationError, match="declare at least one gap"):
        ResearchHandoff(task_id="R1", research_question="q", findings="I found nothing useful.",
                        evidence_ids=[], confidence=Confidence.LOW)


def test_empty_research_cannot_claim_high_confidence():
    from app.schemas.evidence import EvidenceGap

    with pytest.raises(ValidationError, match="high confidence with no evidence"):
        ResearchHandoff(task_id="R1", research_question="q", findings="f", evidence_ids=[],
                        confidence=Confidence.HIGH,
                        gaps=[EvidenceGap(research_question="q", reason="nothing in corpus")])


def test_major_conclusion_without_evidence_is_rejected():
    with pytest.raises(ValidationError, match="must cite at least one evidence id"):
        Conclusion(conclusion_id="C1", statement="X is best", evidence_ids=[],
                   confidence=Confidence.HIGH, is_major=True)


def test_minor_conclusion_may_be_uncited():
    c = Conclusion(conclusion_id="C2", statement="X has a nice logo", evidence_ids=[],
                   confidence=Confidence.LOW, is_major=False)
    assert not c.is_major


def test_rejection_requires_a_problem():
    with pytest.raises(ValidationError, match="at least one problem"):
        CriticVerdict(approved=False, problems=[], required_revisions=["fix it"])


def test_rejection_requires_actionable_instructions():
    with pytest.raises(ValidationError, match="required_revisions or missing_evidence"):
        CriticVerdict(
            approved=False,
            problems=[Problem(location="C1", issue="weak", criterion=ReviewCriterion.RELEVANCE,
                              severity=Severity.MAJOR)],
        )


def test_approval_needs_no_problems():
    assert CriticVerdict(approved=True).approved


def test_unsupported_rate_is_zero_on_empty_report_but_not_clean():
    """The §7.3 trap: an empty check set must not look like a perfect score."""
    report = FactCheckReport(checks=[])
    assert report.unsupported_rate() == 0.0
    assert not report.is_clean()      # nothing verified is not the same as nothing wrong


def test_unsupported_rate_can_actually_fail():
    report = FactCheckReport(checks=[
        ClaimCheck(conclusion_id="C1", citation_exists=True, evidence_supports=True),
        ClaimCheck(conclusion_id="C2", citation_exists=False, fabricated_ids=["E9"],
                   evidence_supports=False),
    ])
    assert report.unsupported_rate() == 0.5
    assert report.fabricated == ["E9"]
    assert not report.is_clean()


def test_analysis_cited_ids_gathers_from_everywhere():
    a = AnalysisHandoff(
        summary="s",
        conclusions=[Conclusion(conclusion_id="C1", statement="s", evidence_ids=["E101"],
                                confidence=Confidence.HIGH, is_major=True)],
        evidence_ids_used=["E102"],
    )
    assert a.cited_ids() == {"E101", "E102"}


# --------------------------------------------------------------------------- #
# Report (§25)
# --------------------------------------------------------------------------- #
def test_markdown_has_all_eight_required_sections():
    report = FinalReport(
        title="T", executive_summary="es", research_objective="ro", methodology="m",
        key_findings=["f1"], comparison_or_analysis="c", risks_and_limitations=["r"],
        recommendation=Recommendation(statement="do X", evidence_ids=["E101"]),
        evidence_used=[make_evidence()],
    )
    md = report.to_markdown()
    for heading in ("Executive Summary", "Research Objective", "Methodology", "Key Findings",
                    "Comparison and Analysis", "Risks and Limitations", "Recommendation",
                    "Evidence and References"):
        assert f"## {heading}" in md, f"missing section: {heading}"


def test_markdown_escapes_pipes_in_evidence_table():
    e = make_evidence()
    e.claim = "A | B"
    md = FinalReport(title="T", executive_summary="e", research_objective="r",
                     methodology="m", evidence_used=[e]).to_markdown()
    assert "A \\| B" in md


# --------------------------------------------------------------------------- #
# State (§16, §27) and parallel safety (§19)
# --------------------------------------------------------------------------- #
def test_initial_state_has_no_none_collections():
    s = initial_state("compare things")
    for key in ("clarifications", "evidence", "research_handoffs", "trace", "errors",
                "critic_verdicts", "human_decisions"):
        assert s[key] == [], f"{key} must start as a list, not None"
    assert s["task_status"] == {}
    assert s["status"] is WorkflowStatus.PENDING


def test_evidence_ids_from_parallel_tasks_never_collide():
    """Two researchers minting ids concurrently must not produce the same id."""
    r1 = {evidence_id_for("R1", i) for i in range(1, 20)}
    r2 = {evidence_id_for("R2", i) for i in range(1, 20)}
    assert not (r1 & r2)
    assert evidence_id_for("R2", 3) == "E203"


def test_merge_status_combines_concurrent_updates():
    left = {"R1": TaskStatus.COMPLETED}
    right = {"R2": TaskStatus.FAILED}
    assert merge_status(left, right) == {"R1": TaskStatus.COMPLETED, "R2": TaskStatus.FAILED}


def test_every_parallel_channel_has_a_reducer():
    """The InvalidUpdateError guard: a fan-out channel without a reducer crashes at runtime.

    ``state.py`` uses ``from __future__ import annotations``, so raw ``__annotations__`` are
    strings and carry no ``__metadata__``. Hints must be resolved with ``include_extras=True``
    to see the ``Annotated[...]`` reducer at all.
    """
    from typing import get_type_hints

    hints = get_type_hints(WorkflowState, include_extras=True)
    for channel in PARALLEL_WRITE_CHANNELS:
        assert channel in hints, f"{channel} missing from WorkflowState"
        assert hasattr(hints[channel], "__metadata__"), (
            f"{channel} is written by parallel researchers but has no reducer — "
            f"LangGraph will raise InvalidUpdateError"
        )
        reducer = hints[channel].__metadata__[0]
        assert callable(reducer), f"{channel}'s reducer is not callable"


def test_reducer_check_would_catch_a_missing_reducer():
    """Proof the guard above can fail — a reducer check that cannot fail is not a check."""
    from typing import get_type_hints

    class Bad(TypedDict, total=False):
        evidence: list[Evidence]          # deliberately un-annotated

    hints = get_type_hints(Bad, include_extras=True)
    assert not hasattr(hints["evidence"], "__metadata__")


def test_permissions_cover_every_state_field():
    declared = set(FIELD_PERMISSIONS)
    actual = set(WorkflowState.__annotations__)
    assert declared == actual, (
        f"§27 table out of sync — missing: {actual - declared}, extra: {declared - actual}"
    )


def test_writer_cannot_write_evidence():
    """Permission intent check: the Writer must never be able to manufacture evidence."""
    assert AgentId.WRITER not in FIELD_PERMISSIONS["evidence"]["write"]
    assert AgentId.RESEARCHER in FIELD_PERMISSIONS["evidence"]["write"]


def test_researcher_cannot_write_the_report():
    assert AgentId.RESEARCHER not in FIELD_PERMISSIONS["report"]["write"]
    assert FIELD_PERMISSIONS["report"]["write"] == {AgentId.WRITER}


def test_permissions_table_renders_every_field():
    table = permissions_table()
    for field in FIELD_PERMISSIONS:
        assert f"`{field}`" in table


def test_langgraph_actually_merges_parallel_writes():
    """End-to-end proof the reducers work under a real LangGraph fan-out.

    This is the test that would have caught the InvalidUpdateError class of bug in Phase 7,
    six phases before the fan-out is built.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    class S(TypedDict, total=False):
        evidence: Annotated[list[Evidence], operator.add]
        task_status: Annotated[dict[str, TaskStatus], merge_status]

    def fan_out(_state):
        return [Send("research", {"task_id": t}) for t in ("R1", "R2", "R3")]

    def research(payload):
        tid = payload["task_id"]
        return {
            "evidence": [make_evidence(evidence_id_for(tid, 1), q=f"q for {tid}")],
            "task_status": {tid: TaskStatus.COMPLETED},
        }

    g = StateGraph(S)
    g.add_node("start", lambda s: {})
    g.add_node("research", research)
    g.add_edge(START, "start")
    g.add_conditional_edges("start", fan_out, ["research"])
    g.add_edge("research", END)

    out = g.compile().invoke({"evidence": [], "task_status": {}})
    assert len(out["evidence"]) == 3, "parallel evidence writes were lost"
    assert {e.evidence_id for e in out["evidence"]} == {"E101", "E201", "E301"}
    assert out["task_status"] == {t: TaskStatus.COMPLETED for t in ("R1", "R2", "R3")}
