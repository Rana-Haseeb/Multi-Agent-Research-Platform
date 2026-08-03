"""
Critic Detection Rate benchmark (§29).

Phase 5 proved the revision loop *terminates*. This measures whether it is worth having: given
an analysis with a known defect, does the wired Critic reject it, and for the right reason?

**The control case is the whole design.** A Critic that rejects everything scores 100% detection
and is useless — it burns two revision cycles on every run and teaches the Analyst nothing. So
the bench includes a deliberately clean analysis, and a Critic that rejects it fails. Detection
rate is only meaningful reported alongside the false-positive rate, and this harness refuses to
report one without the other.

That is the §7.3 lesson applied to the metric that most invites gaming: any scorer where "reject
everything" wins is a scorer that measures nothing.

Each scenario carries its defect in ``expects``, so a scenario cannot silently stop testing what
it claims to test. Scoring is deterministic — marker terms in the Critic's own prose, plus the
structured fields — with no LLM judge.

    python eval/critic_bench.py                # all scenarios, live
    python eval/critic_bench.py --only clean   # one scenario
    python eval/critic_bench.py --runs 2       # repeat for stability
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents import reviewers  # noqa: E402
from app.schemas.common import (  # noqa: E402
    AgentId,
    ClaimType,
    Confidence,
    ReviewCriterion,
)
from app.schemas.evidence import Evidence, EvidenceGap  # noqa: E402
from app.schemas.handoffs import (  # noqa: E402
    AnalysisHandoff,
    ComparisonRow,
    Conclusion,
    ResearchHandoff,
)
from app.schemas.request import RequestBrief  # noqa: E402
from app.services.usage import UsageTracker  # noqa: E402

OUT = ROOT / "eval" / "critic_bench_results.json"


# --------------------------------------------------------------------------- #
# Shared fixture world
# --------------------------------------------------------------------------- #
BRIEF = RequestBrief(
    objective="Compare LangGraph and CrewAI for a small Python team",
    sub_questions=["What state management does each provide?",
                   "How does each handle human approval?"],
    evaluation_criteria=["state management", "human-in-the-loop", "total cost of ownership"],
    options_under_comparison=["LangGraph", "CrewAI"],
    deliverable="Comparison and recommendation",
)


def _ev(eid: str, claim: str, text: str, *, question: str,
        confidence=Confidence.HIGH, claim_type=ClaimType.FACT,
        source="fw-langgraph-docs", title="LangGraph Documentation") -> Evidence:
    return Evidence(
        evidence_id=eid, claim=claim, supporting_text=text, source_id=source,
        source_title=title, research_question=question, confidence=confidence,
        claim_type=claim_type, agent_id="researcher", task_id="R1",
    )


Q1, Q2 = BRIEF.sub_questions

EVIDENCE = [
    _ev("E101", "LangGraph uses typed state channels with reducers",
        "State is declared as a TypedDict whose channels may carry reducer functions.",
        question=Q1),
    _ev("E102", "CrewAI passes state between tasks as text context",
        "State between tasks is passed as text context rather than a typed structure.",
        question=Q1, source="fw-crewai-docs", title="CrewAI Documentation"),
    _ev("E103", "LangGraph supports interrupt() for human-in-the-loop",
        "The interrupt() function suspends execution and returns control to the caller.",
        question=Q2),
    _ev("E104", "CrewAI markets a 10x development speedup",
        "Teams building with CrewAI ship their first production workflow up to 10x faster.",
        question=Q2, confidence=Confidence.LOW, claim_type=ClaimType.CLAIM,
        source="fw-vendor-comparison", title="Why Teams Are Moving to CrewAI"),
    _ev("E105", "One benchmark measured CrewAI at 3.8s p50 on a single task",
        "CrewAI 3.8 s p50 latency on a fixed three-step research task.",
        question=Q2, confidence=Confidence.MEDIUM,
        source="fw-benchmark-2026", title="Independent Agent Framework Benchmark"),
]

HANDOFFS = [
    ResearchHandoff(task_id="R1", research_question=Q1, findings="State models differ.",
                    evidence_ids=["E101", "E102"], confidence=Confidence.HIGH),
    ResearchHandoff(task_id="R2", research_question=Q2, findings="Approval support differs.",
                    evidence_ids=["E103", "E104", "E105"], confidence=Confidence.MEDIUM,
                    gaps=[EvidenceGap(research_question="What is the total cost of ownership?",
                                      reason="No cost data in the corpus.")]),
]


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #
@dataclass
class Scenario:
    """One analysis with a known property, and what a competent Critic should say about it."""

    name: str
    description: str
    analysis: AnalysisHandoff
    should_reject: bool
    criterion: ReviewCriterion | None = None
    markers: list[str] = field(default_factory=list)

    def detected(self, verdict) -> bool:
        """Did the Critic find *this* defect, not merely reject for some other reason?

        A rejection that names the wrong problem is not a detection — the Analyst would then fix
        something that was never broken and the real defect would survive the revision.
        """
        if not self.should_reject:
            return verdict.approved            # the control: correct behaviour is approval
        if verdict.approved:
            return False
        blob = " ".join(
            [p.location + " " + p.issue for p in verdict.problems]
            + verdict.required_revisions + verdict.missing_evidence
        ).lower()
        by_marker = sum(m.lower() in blob for m in self.markers) >= 1
        by_criterion = self.criterion in {p.criterion for p in verdict.problems}
        return by_marker or by_criterion


def _c(cid, statement, ids, *, major=True, conf=Confidence.HIGH) -> Conclusion:
    return Conclusion(conclusion_id=cid, statement=statement, evidence_ids=ids,
                      confidence=conf, is_major=major)


def scenarios() -> list[Scenario]:
    # The control has to be genuinely sound, and getting it there took two corrections that the
    # bench itself surfaced on its first run:
    #
    #   1. The brief lists THREE evaluation criteria. An analysis that silently omits one is
    #      incomplete under §18, however well-evidenced the other two are — so C3 addresses
    #      total cost of ownership explicitly by reporting that the evidence does not support an
    #      assessment. Naming a gap is a complete answer; ignoring the criterion is not.
    #   2. C2 originally said LangGraph "offers built-in" human-in-the-loop where the evidence
    #      says it "supports" interrupt(). The Critic called that overstatement, and it was
    #      right — the wording now matches the evidence.
    #   3. The first attempt at C3 cited E105 (a latency benchmark) in support of a statement
    #      about *cost*. The Critic flagged the citation as irrelevant — correctly, and it is
    #      the same defect this bench plants deliberately elsewhere. A conclusion asserting that
    #      evidence is ABSENT must cite nothing; it is minor by construction, which is why the
    #      schema permits it to be uncited.
    #
    # All three were defects in the fixture, not the Critic. Left uncorrected, each would have
    # shown up as a false positive and been misread as the Critic being too aggressive.
    clean = AnalysisHandoff(
        summary="The two frameworks differ in state model and approval support. Cost of "
                "ownership could not be assessed from the available evidence.",
        conclusions=[
            _c("C1", "LangGraph provides typed state channels; CrewAI passes text context",
               ["E101", "E102"]),
            _c("C2", "LangGraph supports human-in-the-loop pauses via interrupt()", ["E103"]),
            _c("C3", "Total cost of ownership cannot be compared: no cost evidence was gathered",
               [], major=False, conf=Confidence.LOW),
        ],
        comparison=[
            ComparisonRow(option="LangGraph",
                          scores={"state management": "typed channels",
                                  "human-in-the-loop": "supported via interrupt()",
                                  "total cost of ownership": "not assessed — no evidence"},
                          evidence_ids=["E101", "E103"]),
            ComparisonRow(option="CrewAI",
                          scores={"state management": "text context",
                                  "human-in-the-loop": "not established by the evidence",
                                  "total cost of ownership": "not assessed — no evidence"},
                          evidence_ids=["E102"]),
        ],
        # Defect 4 the Critic found in this fixture: the original trade-off asserted a "steeper
        # learning curve", which no evidence here mentions. A trade-off is a claim like any
        # other and needs support; this one is traceable to E101 and E102.
        trade_offs=["LangGraph's typed state channels are more explicit than CrewAI's text "
                    "context, at the cost of declaring channels up front [E101, E102]."],
        assumptions=["Total cost of ownership was not researched, so no cost comparison is "
                     "offered and none should be inferred."],
        evidence_ids_used=["E101", "E102", "E103"],
    )

    fabricated = clean.model_copy(deep=True)
    fabricated.conclusions.append(
        _c("C3", "CrewAI has the strongest community support of any framework", ["E999"]))

    contradiction = clean.model_copy(deep=True)
    contradiction.conclusions.append(
        _c("C3", "LangGraph provides no mechanism for pausing a workflow for human approval",
           ["E103"]))

    overgeneral = clean.model_copy(deep=True)
    overgeneral.conclusions.append(
        _c("C3", "CrewAI is the fastest agent framework available", ["E105"]))

    vendor_as_fact = clean.model_copy(deep=True)
    vendor_as_fact.conclusions.append(
        _c("C3", "Teams building with CrewAI are 10x more productive", ["E104"]))

    unresearched = clean.model_copy(deep=True)
    unresearched.summary += (
        " LangGraph is recommended on total cost of ownership grounds.")
    unresearched.conclusions.append(
        _c("C3", "LangGraph has the lowest total cost of ownership of the two", ["E101"]))

    irrelevant = clean.model_copy(deep=True)
    irrelevant.conclusions.append(
        _c("C3", "CrewAI has better documentation than LangGraph", ["E101"]))

    return [
        Scenario("clean", "A sound analysis. The Critic must APPROVE it.", clean,
                 should_reject=False),
        Scenario("fabricated_citation", "Cites E999, which does not exist.", fabricated,
                 should_reject=True, criterion=ReviewCriterion.UNSUPPORTED_CLAIMS,
                 markers=["e999", "does not exist", "fabricat", "no such"]),
        Scenario("contradiction", "C2 and C3 cannot both be true.", contradiction,
                 should_reject=True, criterion=ReviewCriterion.CONTRADICTIONS,
                 markers=["contradict", "conflict", "inconsist", "c3", "both"]),
        Scenario("overgeneralisation", "'Fastest available' from one benchmark on one task.",
                 overgeneral, should_reject=True, criterion=ReviewCriterion.UNSUPPORTED_CLAIMS,
                 markers=["fastest", "single", "one benchmark", "generalis", "generaliz",
                          "e105"]),
        Scenario("vendor_claim_as_fact", "Vendor marketing stated as established fact.",
                 vendor_as_fact, should_reject=True,
                 criterion=ReviewCriterion.UNSUPPORTED_CLAIMS,
                 markers=["vendor", "marketing", "10x", "e104", "interested", "claim"]),
        Scenario("unresearched_criterion",
                 "Recommends on total cost of ownership, which was never researched.",
                 unresearched, should_reject=True, criterion=ReviewCriterion.EVIDENCE_COVERAGE,
                 markers=["cost", "ownership", "tco", "not researched", "gap", "e101"]),
        Scenario("irrelevant_citation", "C3 cites evidence about state, not documentation.",
                 irrelevant, should_reject=True, criterion=ReviewCriterion.RELEVANCE,
                 markers=["documentation", "e101", "does not support", "not support",
                          "irrelevant", "unrelated"]),
    ]


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_scenario(scenario: Scenario, usage: UsageTracker) -> dict:
    fact_check = reviewers.fact_check(scenario.analysis, EVIDENCE, usage=usage)
    outcome = reviewers.review(BRIEF, scenario.analysis, EVIDENCE,
                               fact_check.output, HANDOFFS, usage=usage, cycle=0)
    verdict = outcome.output
    if verdict is None:
        return {"scenario": scenario.name, "ok": False,
                "error": outcome.error.message if outcome.error else "no verdict"}

    scored = set(verdict.scores)
    return {
        "scenario": scenario.name,
        "ok": True,
        "should_reject": scenario.should_reject,
        "rejected": not verdict.approved,
        "correct": scenario.detected(verdict),
        "n_problems": len(verdict.problems),
        "criteria_scored": len(scored),
        "all_six_criteria_scored": scored >= set(ReviewCriterion),
        "actionable": bool(verdict.required_revisions or verdict.missing_evidence)
                      if not verdict.approved else True,
        "problems": [f"[{p.criterion.value}/{p.severity.value}] {p.issue[:120]}"
                     for p in verdict.problems],
    }


def summarise(rows: list[dict]) -> dict:
    """Detection rate and false-positive rate. Neither is reported without the other."""
    ok = [r for r in rows if r.get("ok")]
    defective = [r for r in ok if r["should_reject"]]
    clean = [r for r in ok if not r["should_reject"]]

    detected = sum(r["correct"] for r in defective)
    false_positives = sum(1 for r in clean if r["rejected"])

    return {
        "scenarios_run": len(ok),
        "defective_scenarios": len(defective),
        "detected": detected,
        "detection_rate": round(detected / len(defective), 3) if defective else None,
        "clean_scenarios": len(clean),
        "false_positives": false_positives,
        "false_positive_rate": round(false_positives / len(clean), 3) if clean else None,
        "rejections_actionable": (
            round(sum(r["actionable"] for r in ok if r["rejected"])
                  / max(1, sum(1 for r in ok if r["rejected"])), 3)
        ),
        "runs_scoring_all_six_criteria": sum(r["all_six_criteria_scored"] for r in ok),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single scenario by name")
    ap.add_argument("--runs", type=int, default=1, help="repeat each scenario N times")
    args = ap.parse_args()

    chosen = [s for s in scenarios() if not args.only or s.name == args.only]
    if not chosen:
        print(f"No scenario named {args.only!r}. Available: "
              f"{', '.join(s.name for s in scenarios())}")
        return 1

    usage = UsageTracker(run_id="critic-bench")
    rows: list[dict] = []

    print(f"{'scenario':<24} {'expect':<8} {'got':<8} {'correct':<8} {'crit':<5} problems")
    print("-" * 96)
    for scenario in chosen:
        for _ in range(args.runs):
            row = run_scenario(scenario, usage)
            rows.append(row)
            if not row["ok"]:
                print(f"{scenario.name:<24} FAILED   {row['error'][:50]}")
                continue
            print(
                f"{scenario.name:<24} "
                f"{'reject' if row['should_reject'] else 'approve':<8} "
                f"{'reject' if row['rejected'] else 'approve':<8} "
                f"{'yes' if row['correct'] else 'NO':<8} "
                f"{row['criteria_scored']}/6  "
                f"{row['problems'][0][:56] if row['problems'] else '-'}"
            )

    stats = summarise(rows)
    print()
    print(f"  Critic detection rate   : {stats['detected']}/{stats['defective_scenarios']}"
          f"  ({stats['detection_rate']})")
    print(f"  False-positive rate     : {stats['false_positives']}/{stats['clean_scenarios']}"
          f"  ({stats['false_positive_rate']})   <- a critic that rejects everything fails here")
    print(f"  Rejections actionable   : {stats['rejections_actionable']}")
    print(f"  Scored all six criteria : {stats['runs_scoring_all_six_criteria']}"
          f"/{stats['scenarios_run']}")
    print(f"  Model calls             : {usage.billable_calls} billable "
          f"({usage.total_calls} attempted), {usage.total_input_tokens} in / "
          f"{usage.total_output_tokens} out")

    # §7.4: merge, never clobber a good results file with an empty run.
    if rows:
        prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        prev.update({"summary": stats, "rows": rows, "usage": usage.summary()})
        OUT.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
