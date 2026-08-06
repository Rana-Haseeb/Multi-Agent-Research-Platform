"""
Tests for the evaluation dataset and metrics (§28, §29).

**These test the ruler, not the thing being measured.** Week 3 shipped a metric of
``expected.issubset(actual)`` which was permanently 100% because ``set().issubset(anything)``
is True — it was caught only by auditing a suspiciously perfect score, after it had already been
reported.

So for every metric here there is a test that constructs a deliberately bad result and asserts
the metric *notices*. A metric with no failing test is indistinguishable from ``return 1.0``.
"""
from __future__ import annotations

import pytest

from eval.dataset import (
    DATASET,
    MINIMUMS,
    Category,
    Depth,
    by_category,
    by_id,
    counts,
    estimated_calls,
)
from eval.metrics import CaseResult, aggregate, against_targets, score_case


def _result(case, **kw) -> CaseResult:
    """A result that passes everything, so each test can break exactly one thing."""
    base = dict(
        case_id=case.case_id, category=case.category.value, depth=case.depth.value,
        status="completed",
        clarification_requested=case.expect_clarification,
        research_tasks=0 if case.expect_clarification else case.expect_min_research_tasks,
        total_tasks=6, agents_assigned=["researcher", "analyst", "critic", "writer"],
        options_found=list(case.expect_options), plan_valid=True,
        evidence_count=0 if not case.expect_evidence else 5,
        gaps_declared=2 if case.expect_gaps else 0,
        handoffs=case.expect_min_research_tasks,
        handoffs_expected=case.expect_min_research_tasks,
        fabricated_citations=[], major_conclusions=2, uncited_major=0,
        critic_ran=True, report_produced=case.expect_report,
        report_has_limitations=True,
        checkpoints_recorded=["plan_approval", "final_review"],
        agent_calls=0 if case.expect_failure else 12,
        input_tokens=1000, output_tokens=200, wall_seconds=30.0,
    )
    base.update(kw)
    return CaseResult(**base)


def _scored(cases, results):
    for case, result in zip(cases, results):
        result.checks = score_case(case, result)
    return results


# --------------------------------------------------------------------------- #
# §28 dataset shape
# --------------------------------------------------------------------------- #
def test_dataset_meets_every_category_minimum():
    actual = counts()
    for category, minimum in MINIMUMS.items():
        assert actual[category.value] >= minimum, (
            f"{category.value}: {actual[category.value]} cases, §28 requires {minimum}"
        )


def test_dataset_has_at_least_25_cases():
    assert len(DATASET) >= 25, f"§28 requires 25; dataset has {len(DATASET)}"


def test_case_ids_are_unique():
    ids = [c.case_id for c in DATASET]
    assert len(ids) == len(set(ids))


def test_every_case_declares_its_expectations_and_notes():
    """A case whose expected values were filled in after observing behaviour measures nothing."""
    for case in DATASET:
        assert case.request is not None
        assert case.notes, f"{case.case_id} has no notes (§28 requires documentation)"
        assert case.expected_checkpoint, f"{case.case_id} declares no checkpoint expectation"


def test_the_ambiguous_category_contains_a_must_not_clarify_control():
    """Without it, 'always ask for clarification' scores 100% on clarification accuracy."""
    controls = [c for c in by_category(Category.AMBIGUOUS) if not c.expect_clarification]
    assert controls, "no near-miss control in the ambiguous category"


def test_full_depth_cases_span_the_interesting_categories():
    full = {c.category for c in DATASET if c.depth is Depth.FULL}
    assert Category.COMPARISON in full
    assert Category.INSUFFICIENT_EVIDENCE in full, (
        "no full-depth case proves the system declares absence rather than inventing"
    )


def test_plan_depth_cases_do_not_claim_downstream_metrics():
    for case in DATASET:
        if case.depth is Depth.PLAN:
            assert not case.supports_metric("evidence_coverage")
            assert case.supports_metric("routing_accuracy")


def test_estimated_cost_is_reported_for_quota_planning():
    assert estimated_calls() > 0


# --------------------------------------------------------------------------- #
# Scoring: each metric must be able to fail
# --------------------------------------------------------------------------- #
def test_a_perfect_run_passes_every_check():
    case = by_id("C1")
    result = _result(case)
    result.checks = score_case(case, result)
    assert result.passed, f"failing: {[k for k, v in result.checks.items() if not v]}"


def test_planning_accuracy_fails_on_too_many_tasks():
    case = by_id("C1")
    result = _result(case, research_tasks=99)
    result.checks = score_case(case, result)
    assert result.checks["task_count_reasonable"] is False


def test_planning_accuracy_fails_when_options_are_missed():
    case = by_id("C1")
    result = _result(case, options_found=["something else entirely"])
    result.checks = score_case(case, result)
    assert result.checks["identified_the_options"] is False


def test_routing_accuracy_fails_on_an_invented_agent():
    case = by_id("C1")
    result = _result(case, agents_assigned=["researcher", "wizard"])
    result.checks = score_case(case, result)
    assert result.checks["only_known_agents_assigned"] is False


def test_routing_accuracy_fails_when_research_goes_to_the_wrong_agent():
    case = by_id("C1")
    result = _result(case, agents_assigned=["analyst", "writer"], research_tasks=2)
    result.checks = score_case(case, result)
    assert result.checks["research_assigned_to_researcher"] is False


def test_clarification_fails_when_an_ambiguous_request_is_planned_anyway():
    case = by_id("A1")
    result = _result(case, clarification_requested=False, research_tasks=3)
    result.checks = score_case(case, result)
    assert result.checks["clarification_decision"] is False


def test_clarification_fails_when_a_clear_request_stalls():
    """The near-miss control: over-asking must cost something."""
    case = by_id("A4")
    result = _result(case, clarification_requested=True)
    result.checks = score_case(case, result)
    assert result.checks["clarification_decision"] is False


def test_approval_compliance_fails_when_no_checkpoint_is_recorded():
    case = by_id("C1")
    result = _result(case, checkpoints_recorded=[])
    result.checks = score_case(case, result)
    assert result.checks["approval_checkpoint_recorded"] is False


def test_evidence_coverage_fails_when_a_covered_question_finds_nothing():
    case = by_id("S1")
    result = _result(case, evidence_count=0)
    result.checks = score_case(case, result)
    assert result.checks["evidence_gathered"] is False


def test_insufficient_evidence_case_fails_if_the_gap_is_not_declared():
    """The hallucination check: inventing an answer must fail, not pass quietly."""
    case = by_id("I1")
    result = _result(case, evidence_count=9, gaps_declared=0)
    result.checks = score_case(case, result)
    assert result.checks["gaps_declared"] is False


def test_handoff_success_fails_when_a_researcher_never_reports():
    case = by_id("C1")
    result = _result(case, handoffs=0, handoffs_expected=3)
    result.checks = score_case(case, result)
    assert result.checks["handoffs_completed"] is False


def test_unsupported_claims_fails_on_a_fabricated_citation():
    case = by_id("C1")
    result = _result(case, fabricated_citations=["E999"])
    result.checks = score_case(case, result)
    assert result.checks["no_fabricated_citations_survived"] is False


def test_unsupported_claims_fails_on_an_uncited_major_conclusion():
    case = by_id("C1")
    result = _result(case, uncited_major=1)
    result.checks = score_case(case, result)
    assert result.checks["no_uncited_major_conclusions"] is False


def test_completion_fails_when_the_workflow_does_not_finish():
    case = by_id("C1")
    result = _result(case, status="failed")
    result.checks = score_case(case, result)
    assert result.checks["workflow_completed"] is False


def test_a_failure_case_fails_if_the_system_proceeds_anyway():
    case = by_id("F1")
    result = _result(case, status="completed", agent_calls=14)
    result.checks = score_case(case, result)
    assert result.checks["failed_as_expected"] is False
    assert result.checks["spent_nothing_on_a_bad_request"] is False


def test_a_failure_case_passes_when_it_fails_fast():
    case = by_id("F1")
    result = _result(case, status="failed", agent_calls=0)
    result.checks = score_case(case, result)
    assert result.passed


# --------------------------------------------------------------------------- #
# Aggregation: empty must not read as perfect (§7.3)
# --------------------------------------------------------------------------- #
def test_an_empty_run_reports_no_score_rather_than_a_perfect_one():
    summary = aggregate(DATASET, [])
    assert summary["cases_run"] == 0
    assert summary["pass_rate"] is None
    assert summary["task_planning_accuracy"]["rate"] is None
    assert summary["workflow_completion_rate"] is None
    assert summary["average_agent_calls"] is None


def test_targets_report_unmeasured_metrics_as_unmeasured():
    rows = against_targets(aggregate(DATASET, []))
    assert rows and all(r["met"] is None for r in rows)


def test_aggregate_detects_a_single_bad_case():
    cases = [by_id("C1"), by_id("S1")]
    results = _scored(cases, [_result(cases[0], fabricated_citations=["E9"]),
                              _result(cases[1])])
    summary = aggregate(DATASET, results)
    assert summary["cases_passed"] == 1
    assert summary["unsupported_major_claims"]["rate"] < 1.0
    assert summary["failing_cases"][0]["case_id"] == "C1"


def test_aggregate_reports_which_cases_were_not_run():
    cases = [by_id("C1")]
    summary = aggregate(DATASET, _scored(cases, [_result(cases[0])]))
    assert "S1" in summary["not_run"]
    assert "C1" not in summary["not_run"]


def test_per_category_breakdown_prevents_a_weak_category_hiding_in_the_average():
    cases = [by_id("C1"), by_id("I1")]
    results = _scored(cases, [_result(cases[0]),
                              _result(cases[1], gaps_declared=0, evidence_count=9)])
    summary = aggregate(DATASET, results)
    assert summary["by_category"]["comparison"]["rate"] == 1.0
    assert summary["by_category"]["insufficient"]["rate"] == 0.0


def test_plan_depth_results_do_not_pollute_completion_rate():
    """Completion is a full-workflow property; plan-depth cases must not be counted."""
    plan_case = by_id("S2")
    results = _scored([plan_case], [_result(plan_case, status="completed")])
    summary = aggregate(DATASET, results)
    assert summary["workflow_completion_rate"] is None
