"""
Tests for the Critic Detection Rate benchmark itself (§29).

A benchmark is a measuring instrument, and an uncalibrated instrument is worse than none — it
produces a number that looks authoritative and isn't. So these tests check the *scorer*, not the
Critic: they feed it Critics with known pathologies and assert it reports them.

The two that matter:

- An **always-approving** Critic must score 0 detection. If it doesn't, the bench cannot
  distinguish a working reviewer from a rubber stamp.
- An **always-rejecting** Critic must score 100% false positives. If it doesn't, "reject
  everything" is a winning strategy and the detection rate measures nothing.

This is the §7.3 discipline applied to the metric most vulnerable to it.
"""
from __future__ import annotations

import pytest

from app.schemas.common import ReviewCriterion, Severity
from app.schemas.handoffs import CriticVerdict, Problem
from eval.critic_bench import BRIEF, EVIDENCE, scenarios, summarise


def _approve() -> CriticVerdict:
    return CriticVerdict(approved=True, scores=dict.fromkeys(ReviewCriterion, 5))


def _reject(issue: str = "something is wrong",
            criterion: ReviewCriterion = ReviewCriterion.RELEVANCE) -> CriticVerdict:
    return CriticVerdict(
        approved=False,
        problems=[Problem(location="C1", issue=issue, criterion=criterion,
                          severity=Severity.MAJOR)],
        required_revisions=["fix it"],
        scores=dict.fromkeys(ReviewCriterion, 2),
    )


def _rows(verdict_for) -> list[dict]:
    """Score every scenario against a synthetic Critic, without any model call."""
    out = []
    for s in scenarios():
        verdict = verdict_for(s)
        out.append({
            "scenario": s.name, "ok": True, "should_reject": s.should_reject,
            "rejected": not verdict.approved, "correct": s.detected(verdict),
            "actionable": True, "all_six_criteria_scored": True,
        })
    return out


# --------------------------------------------------------------------------- #
# Calibration — the scorer must be able to fail
# --------------------------------------------------------------------------- #
def test_an_always_approving_critic_scores_zero_detection():
    stats = summarise(_rows(lambda s: _approve()))
    assert stats["detection_rate"] == 0.0
    assert stats["false_positive_rate"] == 0.0      # it approves the clean one too


def test_an_always_rejecting_critic_scores_maximum_false_positives():
    """'Reject everything' must not be a winning strategy."""
    stats = summarise(_rows(lambda s: _reject()))
    assert stats["false_positive_rate"] == 1.0


def test_rejecting_for_the_wrong_reason_is_not_a_detection():
    """A rejection naming the wrong defect sends the Analyst to fix the wrong thing.

    The real defect then survives the revision, so counting it as a catch would overstate the
    Critic's value precisely where it matters most.
    """
    unrelated = _reject(issue="the prose could be more concise",
                        criterion=ReviewCriterion.COMPLETENESS)
    fabricated = next(s for s in scenarios() if s.name == "fabricated_citation")
    assert not fabricated.detected(unrelated)


def test_rejecting_for_the_right_reason_is_a_detection():
    correct = _reject(issue="Conclusion C3 cites E999, which does not exist",
                      criterion=ReviewCriterion.UNSUPPORTED_CLAIMS)
    fabricated = next(s for s in scenarios() if s.name == "fabricated_citation")
    assert fabricated.detected(correct)


def test_detection_can_be_matched_by_criterion_alone():
    """Marker terms are one route; the structured criterion field is the other."""
    contradiction = next(s for s in scenarios() if s.name == "contradiction")
    by_criterion = _reject(issue="the two statements cannot both hold",
                           criterion=ReviewCriterion.CONTRADICTIONS)
    assert contradiction.detected(by_criterion)


def test_summary_reports_both_rates_or_neither():
    stats = summarise(_rows(lambda s: _approve()))
    assert "detection_rate" in stats and "false_positive_rate" in stats


def test_summary_handles_an_empty_run_without_inventing_a_score():
    stats = summarise([])
    assert stats["detection_rate"] is None and stats["false_positive_rate"] is None
    assert stats["scenarios_run"] == 0


# --------------------------------------------------------------------------- #
# Scenario integrity
# --------------------------------------------------------------------------- #
def test_there_is_a_clean_control():
    """Without it, detection rate is unfalsifiable."""
    controls = [s for s in scenarios() if not s.should_reject]
    assert len(controls) >= 1


def test_every_defective_scenario_declares_what_it_plants():
    for s in scenarios():
        if s.should_reject:
            assert s.criterion is not None, f"{s.name} has no expected criterion"
            assert s.markers, f"{s.name} has no marker terms"
            assert s.description, f"{s.name} has no description"


def test_scenarios_cover_the_distinct_defect_families():
    names = {s.name for s in scenarios()}
    assert {"fabricated_citation", "contradiction", "overgeneralisation",
            "vendor_claim_as_fact", "unresearched_criterion",
            "irrelevant_citation"} <= names


def test_the_clean_control_is_structurally_sound():
    """It must address every evaluation criterion and cite only evidence that exists.

    Five separate defects were found in this fixture by running the bench — an omitted
    criterion, an overstated conclusion, an irrelevant citation, an unsupported trade-off, and a
    claim outside the Analyst's remit. This asserts the first and third never come back.
    """
    clean = next(s for s in scenarios() if not s.should_reject).analysis
    known = {e.evidence_id for e in EVIDENCE}

    assert clean.cited_ids() <= known, (
        f"control cites evidence that does not exist: {clean.cited_ids() - known}"
    )
    for row in clean.comparison:
        missing = set(BRIEF.evaluation_criteria) - set(row.scores)
        assert not missing, f"{row.option} does not address: {missing}"
    for c in clean.conclusions:
        if c.is_major:
            assert c.evidence_ids, f"{c.conclusion_id} is major but uncited"


def test_critic_prompt_scopes_the_review_to_the_analysis():
    """Regression guard for the real bug this bench found.

    The Critic was rejecting sound analyses for lacking a recommendation — which the Analyst
    never produces, because that is the Writer's job. It was reviewing against the wrong
    contract, and every real run paid two wasted revision cycles for it.
    """
    from app.agents.prompts import CRITIC

    assert "not a report" in CRITIC.lower()
    assert "recommendation" in CRITIC.lower()
    assert "writer" in CRITIC.lower()
