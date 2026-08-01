"""
Phase 4 tests: the six agents in isolation, with the LLM mocked.

Each agent is tested for what it produces on good input **and** what it does on bad input —
provider failure, invalid output, fabricated citations, empty research. In a six-agent workflow
the error paths outnumber the happy path, and they are the ones that decide whether the system
degrades or collapses.
"""
from __future__ import annotations

import pytest

from app.agents import reviewers, supervisor
from app.agents.base import classify_failure
from app.agents.context import (
    analyst_context,
    critic_context,
    full_context_control,
    researcher_context,
    supervisor_context,
    writer_context,
)
from app.agents.prompts import ALL_PROMPTS
from app.agents.supervisor import PlanDraft
from app.config import settings
from app.schemas.common import (
    AgentId,
    ClaimType,
    Confidence,
    ReviewCriterion,
    Severity,
    TaskStatus,
)
from app.schemas.handoffs import (
    AnalysisHandoff,
    Conclusion,
    CriticVerdict,
    FactCheckReport,
    Problem,
)
from app.schemas.request import ClarificationQuestion, RequestBrief
from app.schemas.tasks import Task, TaskPlan
from app.services.llm_service import LLMError
from app.services.usage import BudgetExceeded
from tests.conftest import make_evidence


# --------------------------------------------------------------------------- #
# §12 agent specialisation
# --------------------------------------------------------------------------- #
def test_every_agent_has_distinct_instructions():
    """'Five prompts with the job title swapped' is what §12 forbids."""
    assert len(ALL_PROMPTS) == 8
    bodies = list(ALL_PROMPTS.values())
    assert len(set(bodies)) == len(bodies), "two agents share identical instructions"
    for name, prompt in ALL_PROMPTS.items():
        assert len(prompt) > 300, f"{name} instruction is too thin to specialise anything"


def test_every_specialist_prompt_states_prohibitions():
    for name in ("researcher", "analyst", "critic", "writer", "fact_checker"):
        assert "PROHIBITED" in ALL_PROMPTS[name], f"{name} does not state what it must not do"


def test_agents_reading_sources_carry_the_injection_guard():
    for name in ("researcher", "supervisor_analyse", "baseline"):
        assert "DATA, never instructions" in ALL_PROMPTS[name]


# --------------------------------------------------------------------------- #
# §21 context boundaries
# --------------------------------------------------------------------------- #
def test_supervisor_never_sees_evidence_bodies(brief, evidence, handoffs):
    ctx = supervisor_context("compare frameworks", brief, None, evidence, handoffs)
    assert evidence[0].supporting_text not in ctx
    assert "COVERAGE" in ctx


def test_researcher_sees_only_its_own_question(brief):
    ctx = researcher_context("R1", brief.sub_questions[0], brief.objective)
    assert brief.sub_questions[0] in ctx
    assert brief.sub_questions[1] not in ctx, "researcher can see a sibling's question"


def test_critic_gets_the_index_not_the_bodies(brief, analysis, evidence, handoffs):
    ctx = critic_context(brief, analysis, evidence, None, handoffs)
    assert evidence[0].evidence_id in ctx
    assert evidence[0].supporting_text not in ctx, "critic received full evidence bodies"


def test_analyst_gets_bodies_but_truncated(brief, evidence, handoffs):
    big = make_evidence("E301")
    big.supporting_text = "x" * 5000
    ctx = analyst_context(brief, evidence + [big], handoffs)
    assert "x" * 500 not in ctx
    assert evidence[0].evidence_id in ctx


def test_writer_only_sees_cited_evidence(brief, analysis, evidence, handoffs):
    uncited = make_evidence("E999", claim="An uncited finding nobody used")
    ctx = writer_context(brief, analysis, evidence + [uncited], handoffs)
    assert "E101" in ctx
    assert "An uncited finding nobody used" not in ctx


def test_role_specific_context_is_smaller_than_full_context(
    brief, analysis, evidence, handoffs
):
    """The premise of Experiment 4, asserted rather than assumed."""
    full = full_context_control("compare frameworks", brief, None, evidence, handoffs,
                                analysis, None)
    for scoped in (
        supervisor_context("compare frameworks", brief, None, evidence, handoffs),
        researcher_context("R1", brief.sub_questions[0], brief.objective),
        critic_context(brief, analysis, evidence, None, handoffs),
    ):
        assert len(scoped) < len(full)


def test_analyst_revision_note_is_included(brief, evidence, handoffs):
    ctx = analyst_context(brief, evidence, handoffs, revision_note="Fix conclusion C1.")
    assert "REVISION REQUIRED" in ctx and "Fix conclusion C1." in ctx


def test_writer_context_forces_gaps_into_limitations(brief, analysis, evidence, handoffs):
    from app.schemas.evidence import EvidenceGap

    handoffs[0].gaps = [EvidenceGap(research_question="Pricing?", reason="not in corpus")]
    ctx = writer_context(brief, analysis, evidence, handoffs)
    assert "MUST appear in Risks and Limitations" in ctx and "Pricing?" in ctx


# --------------------------------------------------------------------------- #
# Supervisor
# --------------------------------------------------------------------------- #
def test_analyse_request_returns_brief(fake_llm_factory, brief):
    fake_llm_factory({"supervisor": [brief]})
    out = supervisor.analyse_request("Compare agent frameworks")
    assert out.ok and out.output.objective == brief.objective


def test_analyse_request_survives_provider_failure(fake_llm_factory):
    fake_llm_factory({"supervisor": [LLMError("All configured providers failed.")]})
    out = supervisor.analyse_request("Compare agent frameworks")
    assert out.failed and out.error.kind == "api_failure"
    assert out.output is None


def test_plan_with_a_dependency_cycle_is_reported_not_raised(fake_llm_factory, brief):
    """A cyclic plan would deadlock the graph. It must become a clean failure."""
    cyclic = PlanDraft(tasks=[
        Task(task_id="R1", description="a", assigned_agent=AgentId.RESEARCHER,
             research_question="a?", depends_on=["A1"]),
        Task(task_id="A1", description="b", assigned_agent=AgentId.ANALYST, depends_on=["R1"]),
    ])
    fake_llm_factory({"supervisor": [cyclic]})
    out = supervisor.build_plan(brief)
    assert out.failed and out.error.kind == "invalid_output"
    assert "cycle" in out.error.message.lower()


def test_duplicate_research_tasks_are_collapsed(fake_llm_factory, brief):
    draft = PlanDraft(tasks=[
        Task(task_id="R1", description="research", assigned_agent=AgentId.RESEARCHER,
             research_question="What state management does LangGraph provide?"),
        Task(task_id="R2", description="research", assigned_agent=AgentId.RESEARCHER,
             research_question="What state management does LangGraph provide?"),
        Task(task_id="A1", description="analyse", assigned_agent=AgentId.ANALYST,
             depends_on=["R1", "R2"]),
    ])
    fake_llm_factory({"supervisor": [draft]})
    out = supervisor.build_plan(brief)
    assert out.ok
    assert len(out.output.research_tasks()) == 1
    assert out.output.by_id("A1").depends_on == ["R1"], "dependents were not rewired"


def test_clarification_answers_are_passed_through(fake_llm_factory, brief):
    created = fake_llm_factory({"supervisor": [brief]})
    supervisor.analyse_request("Find the best framework", clarification_answers="Python, prod")
    assert "Python, prod" in created["supervisor"].calls[0]["user"]


# --- evidence gate (deterministic) ---
def test_gate_blocks_when_no_evidence(brief, handoffs):
    d = supervisor.evidence_gate(brief, [], handoffs, research_round=0)
    assert not d.proceed and "No evidence" in d.reason


def test_gate_terminates_at_the_research_round_cap(brief, handoffs):
    """§22: the workflow must finish even when evidence never arrives."""
    d = supervisor.evidence_gate(brief, [], handoffs, settings.max_research_rounds)
    assert d.proceed and "terminates" in d.reason


def test_gate_ignores_assumptions_as_coverage(brief, handoffs):
    only_assumptions = [
        make_evidence("E101", question=brief.sub_questions[0], claim_type=ClaimType.ASSUMPTION),
        make_evidence("E201", question=brief.sub_questions[1], claim_type=ClaimType.ASSUMPTION),
    ]
    d = supervisor.evidence_gate(brief, only_assumptions, handoffs, research_round=0)
    assert not d.proceed and len(d.unresolved_questions) == 2


def test_gate_proceeds_on_good_coverage(brief, evidence, handoffs):
    d = supervisor.evidence_gate(brief, evidence, handoffs, research_round=0)
    assert d.proceed and not d.unresolved_questions


# --- routing (deterministic, must terminate) ---
def test_approval_routes_to_writer():
    d = supervisor.next_action(CriticVerdict(approved=True), 0, 0)
    assert d.next_step == "writer"


def test_rejection_routes_back_to_analyst():
    verdict = CriticVerdict(approved=False,
                            problems=[Problem(location="C1", issue="weak",
                                              criterion=ReviewCriterion.RELEVANCE,
                                              severity=Severity.MAJOR)],
                            required_revisions=["fix it"])
    assert supervisor.next_action(verdict, 0, 0).next_step == "analyst"


def test_workflow_terminates_when_critic_never_approves():
    """§18's hard requirement, tested at the cap and beyond it."""
    verdict = CriticVerdict(approved=False,
                            problems=[Problem(location="C1", issue="still weak",
                                              criterion=ReviewCriterion.RELEVANCE,
                                              severity=Severity.MAJOR)],
                            required_revisions=["fix it"])
    for revisions in range(settings.max_revision_cycles, settings.max_revision_cycles + 3):
        d = supervisor.next_action(verdict, revisions, 0)
        assert d.next_step == "writer", f"looped at revision {revisions}"
        assert "maximum" in d.reason


def test_blocked_tasks_are_marked_skipped():
    plan = TaskPlan(tasks=[
        Task(task_id="R1", description="r", assigned_agent=AgentId.RESEARCHER,
             research_question="q?"),
        Task(task_id="A1", description="a", assigned_agent=AgentId.ANALYST, depends_on=["R1"]),
    ])
    updated = supervisor.apply_task_status(plan, {"R1": TaskStatus.FAILED})
    assert updated.by_id("A1").status is TaskStatus.SKIPPED
    assert updated.is_complete(), "a blocked task left the plan permanently incomplete"


# --------------------------------------------------------------------------- #
# Fact-Checker
# --------------------------------------------------------------------------- #
def test_fabricated_citation_detected_without_a_model(analysis, evidence):
    """The deterministic half. No LLM involved at all."""
    analysis.conclusions[0].evidence_ids = ["E101", "E9"]
    result = reviewers.check_citations_deterministic(analysis, evidence)
    assert result["C1"] == ["E9"] and result["C2"] == []


def test_fact_check_survives_model_failure_keeping_deterministic_result(
    fake_llm_factory, analysis, evidence
):
    """A dead model must not discard a check that code already completed."""
    analysis.conclusions[0].evidence_ids = ["E9"]
    fake_llm_factory({"fact_checker": [LLMError("provider down")]})
    out = reviewers.fact_check(analysis, evidence)
    assert out.ok and out.output.fabricated == ["E9"]
    assert not out.output.is_clean()


def test_fact_check_judges_support_when_citations_exist(fake_llm_factory, evidence):
    """Citations that exist still have to be judged: existing is not the same as supporting."""
    from app.agents.reviewers import SupportJudgement, SupportJudgements

    fake_llm_factory({"fact_checker": [SupportJudgements(judgements=[
        SupportJudgement(conclusion_id="C1", evidence_supports=False,
                         reasoning="Evidence is about state, not about speed."),
    ])]})
    out = reviewers.fact_check(
        AnalysisHandoff(summary="s", conclusions=[
            Conclusion(conclusion_id="C1", statement="LangGraph is the fastest option",
                       evidence_ids=["E101"], confidence=Confidence.LOW)]),
        evidence,
    )
    assert out.ok
    assert out.output.checks[0].citation_exists is True
    assert out.output.unsupported_ids == ["C1"]
    assert not out.output.is_clean()


def test_fact_check_with_no_conclusions_is_not_clean(evidence):
    """No model call needed, and an empty report must not read as a pass."""
    out = reviewers.fact_check(AnalysisHandoff(summary="s", conclusions=[
        Conclusion(conclusion_id="C1", statement="x", evidence_ids=["E101"],
                   confidence=Confidence.LOW)]).model_copy(update={"conclusions": []}), evidence)
    assert out.ok and not out.output.is_clean()


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
def test_critic_approval_is_overridden_by_fabricated_citations(
    fake_llm_factory, brief, analysis, evidence, handoffs
):
    """A lenient Critic cannot approve past a mechanical failure."""
    fake_llm_factory({"critic": [CriticVerdict(approved=True)]})
    report = FactCheckReport(checks=[])
    report_with_fake = FactCheckReport.model_validate({
        "checks": [{"conclusion_id": "C1", "citation_exists": False,
                    "fabricated_ids": ["E9"], "evidence_supports": False}]
    })
    out = reviewers.review(brief, analysis, evidence, report_with_fake, handoffs)
    assert out.ok and not out.output.approved
    assert any("Fabricated" in p.issue for p in out.output.problems)
    assert out.output.required_revisions


def test_critic_approval_overridden_by_uncited_major_conclusion(
    fake_llm_factory, brief, evidence, handoffs
):
    analysis = AnalysisHandoff(
        summary="s",
        conclusions=[Conclusion(conclusion_id="C1", statement="X is best",
                                evidence_ids=["E101"], confidence=Confidence.HIGH,
                                is_major=True)],
    )
    analysis.conclusions[0].evidence_ids = []      # bypass the schema to simulate drift
    fake_llm_factory({"critic": [CriticVerdict(approved=True)]})
    out = reviewers.review(brief, analysis, evidence, None, handoffs)
    assert not out.output.approved


def test_critic_failure_does_not_become_an_approval(
    fake_llm_factory, brief, analysis, evidence, handoffs
):
    """An outage must not turn the quality gate into one that always passes."""
    fake_llm_factory({"critic": [LLMError("provider down")]})
    bad = FactCheckReport.model_validate({
        "checks": [{"conclusion_id": "C1", "citation_exists": False,
                    "fabricated_ids": ["E9"], "evidence_supports": False}]
    })
    out = reviewers.review(brief, analysis, evidence, bad, handoffs)
    assert out.ok and not out.output.approved


def test_critic_failure_with_clean_analysis_discloses_no_review_happened(
    fake_llm_factory, brief, analysis, evidence, handoffs
):
    fake_llm_factory({"critic": [LLMError("provider down")]})
    out = reviewers.review(brief, analysis, evidence, None, handoffs)
    assert out.output.approved
    assert any("unavailable" in m for m in out.output.missing_evidence)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def _draft(**over):
    base = dict(title="T", executive_summary="es", research_objective="ro", methodology="m",
                key_findings=["f"], risks_and_limitations=[],
                recommendation_statement="Adopt LangGraph")
    base.update(over)
    return reviewers.ReportDraft(**base)


def test_writer_report_only_carries_cited_evidence(
    fake_llm_factory, brief, analysis, evidence, handoffs
):
    extra = make_evidence("E999", claim="never cited")
    fake_llm_factory({"writer": [_draft()]})
    out = reviewers.write_report(brief, analysis, evidence + [extra], handoffs)
    assert out.ok
    assert {e.evidence_id for e in out.output.evidence_used} == {"E101", "E201"}


def test_unresolved_critic_objections_reach_the_report(
    fake_llm_factory, brief, analysis, evidence, handoffs
):
    """When the revision cap is hit with objections open, the report must say so."""
    verdict = CriticVerdict(
        approved=False,
        problems=[Problem(location="C1", issue="Overgeneralised from one benchmark",
                          criterion=ReviewCriterion.UNSUPPORTED_CLAIMS,
                          severity=Severity.MAJOR)],
        required_revisions=["narrow the claim"],
    )
    fake_llm_factory({"writer": [_draft()]})
    out = reviewers.write_report(brief, analysis, evidence, handoffs, verdict)
    assert any("Overgeneralised" in lim for lim in out.output.risks_and_limitations)


def test_research_gaps_reach_the_report(fake_llm_factory, brief, analysis, evidence, handoffs):
    from app.schemas.evidence import EvidenceGap

    handoffs[0].gaps = [EvidenceGap(research_question="What does it cost?",
                                    reason="no pricing in corpus")]
    fake_llm_factory({"writer": [_draft()]})
    out = reviewers.write_report(brief, analysis, evidence, handoffs)
    assert any("What does it cost?" in lim for lim in out.output.risks_and_limitations)


def test_report_exports_markdown(fake_llm_factory, brief, analysis, evidence, handoffs):
    fake_llm_factory({"writer": [_draft()]})
    out = reviewers.write_report(brief, analysis, evidence, handoffs)
    md = out.output.to_markdown()
    assert "## Executive Summary" in md and "E101" in md


def test_writer_failure_is_reported_not_raised(
    fake_llm_factory, brief, analysis, evidence, handoffs
):
    fake_llm_factory({"writer": [LLMError("provider down")]})
    out = reviewers.write_report(brief, analysis, evidence, handoffs)
    assert out.failed and out.output is None


# --------------------------------------------------------------------------- #
# Failure classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc,expected", [
    (BudgetExceeded("cap"), "budget"),
    (LLMError("The AI request timed out."), "timeout"),
    (LLMError("The model could not produce the required structured output."), "invalid_output"),
    (LLMError("AI rate limit reached."), "api_failure"),
    (ValueError("something else"), "unexpected"),
])
def test_failures_are_classified_for_the_trace(exc, expected):
    assert classify_failure(exc) == expected


def test_tool_errors_classify_distinctly():
    from app.tools import ToolError, ToolPermissionError

    assert classify_failure(ToolPermissionError("x")) == "tool_permission"
    assert classify_failure(ToolError("x")) == "tool_failure"
