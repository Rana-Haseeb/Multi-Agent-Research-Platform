"""
Experiments 1, 2, 4 and 5 (§30). Experiment 3 lives in ``exp3_parallel_research.py``.

**Isolate the variable, don't re-run the world.** A full workflow costs 80-120k tokens. Running
every arm of every experiment end to end is ~1M tokens — three days of free-tier quota spent
mostly re-measuring stages the variable does not touch, and burying the effect under variance
from those stages. So each experiment runs the narrowest slice that actually differs:

    Exp 1  single agent vs multi-agent    FULL workflow both arms — irreducible, and the
                                          graded question. The whole point is end-to-end
                                          comparison, so there is nothing to isolate.
    Exp 2  critic enabled vs disabled     Analyst -> [Critic] -> Writer over a FIXED evidence
                                          set. Removes research variance, which would otherwise
                                          dominate the quality difference being measured.
    Exp 4  full vs role-specific context  Context assembly is deterministic, so sizes are
                                          measured exactly with ZERO API calls, plus one paired
                                          live Analyst call to compare output.
    Exp 5  revision limits 0 / 1 / 2      Analyst <-> Critic loop over the same fixed evidence,
                                          three caps.

Experiment 2 and 5 share a fixture whose analysis contains **known planted defects**, so "did the
critic help?" is answerable by counting which defects survived rather than by reading prose.

    python experiments/run_experiments.py --experiment 2
    python experiments/run_experiments.py --experiment all --budget 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agents import reviewers  # noqa: E402
from app.agents.context import analyst_context, critic_context, full_context_control  # noqa: E402
from app.baseline.single_agent import baseline_fabricated_citations, run_baseline  # noqa: E402
from app.graph.workflow import run_workflow  # noqa: E402
from app.schemas.common import ClaimType, Confidence, ReviewCriterion  # noqa: E402
from app.schemas.evidence import Evidence, EvidenceGap  # noqa: E402
from app.schemas.handoffs import AnalysisHandoff, Conclusion, ResearchHandoff  # noqa: E402
from app.schemas.request import RequestBrief  # noqa: E402
from app.config import settings  # noqa: E402
from app.services.usage import UsageTracker  # noqa: E402
from app.storage.corpus import get_index  # noqa: E402

OUT = ROOT / "experiments" / "results.json"

REQUEST = ("Compare LangGraph and CrewAI for a 4-person Python team building a production "
           "multi-agent support system. Recommend one.")


# --------------------------------------------------------------------------- #
# Shared fixture — a realistic evidence set with a defective analysis over it
# --------------------------------------------------------------------------- #
BRIEF = RequestBrief(
    objective="Compare LangGraph and CrewAI for a small Python team",
    sub_questions=["What state management does each provide?",
                   "How does each handle human approval?"],
    evaluation_criteria=["state management", "human-in-the-loop", "total cost of ownership"],
    options_under_comparison=["LangGraph", "CrewAI"],
    deliverable="Comparison and recommendation",
)


def _ev(eid, claim, text, *, q, conf=Confidence.HIGH, ct=ClaimType.FACT,
        src="fw-langgraph-docs", title="LangGraph Documentation") -> Evidence:
    return Evidence(evidence_id=eid, claim=claim, supporting_text=text, source_id=src,
                    source_title=title, research_question=q, confidence=conf, claim_type=ct,
                    agent_id="researcher", task_id="R1")


Q1, Q2 = BRIEF.sub_questions
EVIDENCE = [
    _ev("E101", "LangGraph uses typed state channels with reducers",
        "State is declared as a TypedDict whose channels may carry reducer functions.", q=Q1),
    _ev("E102", "CrewAI passes state between tasks as text context",
        "State between tasks is passed as text context rather than a typed structure.",
        q=Q1, src="fw-crewai-docs", title="CrewAI Documentation"),
    _ev("E103", "LangGraph supports interrupt() for human-in-the-loop",
        "The interrupt() function suspends execution and returns control to the caller.", q=Q2),
    _ev("E104", "CrewAI markets a 10x development speedup",
        "Teams building with CrewAI ship their first production workflow up to 10x faster.",
        q=Q2, conf=Confidence.LOW, ct=ClaimType.CLAIM,
        src="fw-vendor-comparison", title="Why Teams Are Moving to CrewAI"),
    _ev("E105", "One benchmark measured CrewAI at 3.8s p50 on a single task",
        "CrewAI 3.8 s p50 latency on a fixed three-step research task.",
        q=Q2, conf=Confidence.MEDIUM,
        src="fw-benchmark-2026", title="Independent Agent Framework Benchmark"),
]

HANDOFFS = [
    ResearchHandoff(task_id="R1", research_question=Q1, findings="State models differ.",
                    evidence_ids=["E101", "E102"], confidence=Confidence.HIGH),
    ResearchHandoff(task_id="R2", research_question=Q2, findings="Approval support differs.",
                    evidence_ids=["E103", "E104", "E105"], confidence=Confidence.MEDIUM,
                    gaps=[EvidenceGap(research_question="What is the total cost of ownership?",
                                      reason="No cost data in the corpus.")]),
]

# Four defects planted in the analysis both arms start from. Counting which survive is what makes
# "did the Critic help?" a number rather than an impression.
#
# Measured STRUCTURALLY on the analysis, not by string-matching the report prose. The first
# version searched the report for "E999", "10x", "fastest" and "total cost" — and scored the
# Critic as catching NOTHING, while it had in fact removed every defect. The markers were present
# only inside the report's own disclaimers: "No fabricated citations (e.g., E999) are used; any
# prior references to such sources have been removed." A detector that cannot tell an assertion
# from a statement that the assertion was avoided measures the opposite of what it claims.
DEFECT_NAMES = ["fabricated_citation", "vendor_claim_as_fact", "overgeneralised",
                "unresearched_criterion"]

DEFECTIVE_ANALYSIS = AnalysisHandoff(
    summary="LangGraph is the stronger choice and has the lowest total cost of ownership.",
    conclusions=[
        Conclusion(conclusion_id="C1",
                   statement="LangGraph provides typed state channels; CrewAI passes text context",
                   evidence_ids=["E101", "E102"], confidence=Confidence.HIGH, is_major=True),
        Conclusion(conclusion_id="C2",
                   statement="Teams using CrewAI are 10x more productive",
                   evidence_ids=["E104"], confidence=Confidence.HIGH, is_major=True),
        Conclusion(conclusion_id="C3",
                   statement="CrewAI is the fastest agent framework available",
                   evidence_ids=["E105"], confidence=Confidence.HIGH, is_major=True),
        Conclusion(conclusion_id="C4",
                   statement="CrewAI has the strongest community support of any framework",
                   evidence_ids=["E999"], confidence=Confidence.HIGH, is_major=True),
    ],
    trade_offs=["LangGraph has the lowest total cost of ownership."],
    assumptions=[],
    evidence_ids_used=["E101", "E102", "E104", "E105", "E999"],
)


def defects_surviving(analysis) -> list[str]:
    """Which planted defects are still ASSERTED by the analysis.

    A conclusion that says "no evidence on X exists" is the defect being *fixed*, not surviving,
    so each check looks for the defective assertion rather than for a keyword.
    """
    if analysis is None:
        return list(DEFECT_NAMES)

    known = {e.evidence_id for e in EVIDENCE}
    survived: list[str] = []

    def asserted(*needles: str) -> bool:
        """A MAJOR conclusion positively claiming this, not denying or qualifying it."""
        for c in analysis.conclusions:
            s = c.statement.lower()
            if any(n in s for n in needles) and not any(
                neg in s for neg in ("no evidence", "not supported", "cannot", "does not",
                                     "insufficient", "was not", "no data")
            ):
                return True
        return False

    if any(eid not in known for c in analysis.conclusions for eid in c.evidence_ids):
        survived.append("fabricated_citation")
    if asserted("10x", "10 x", "more productive"):
        survived.append("vendor_claim_as_fact")
    if asserted("fastest", "the fastest"):
        survived.append("overgeneralised")
    # The TCO claim is planted in the summary and trade-offs, not only in a conclusion, so this
    # one searches those fields too. A calibration assertion (see below) catches the omission if
    # a future edit moves the defect somewhere the detector does not look.
    tco_needles = ("lowest total cost", "total cost of ownership is", "best total cost",
                   "lowest cost of ownership")
    tco_text = " ".join([analysis.summary, *analysis.trade_offs,
                         *(c.statement for c in analysis.conclusions)]).lower()
    if any(n in tco_text for n in tco_needles) and not any(
        neg in tco_text for neg in ("no evidence on total cost", "cost of ownership for either",
                                    "no cost evidence", "not assessed")
    ):
        survived.append("unresearched_criterion")
    return survived


def _assert_detector_sees_the_planted_defects() -> None:
    """The detector must see all four in the fixture, or it is measuring nothing.

    Runs at import. The previous prose-matching detector reported the Critic catching zero
    defects when it had in fact caught all four; a calibration check is the cheapest guard
    against shipping that number a second time.
    """
    found = set(defects_surviving(DEFECTIVE_ANALYSIS))
    missing = set(DEFECT_NAMES) - found
    if missing:
        raise AssertionError(
            f"defect detector is blind to {sorted(missing)} in the planted fixture — "
            f"any 'caught by critic' figure it produces would be understated"
        )


_assert_detector_sees_the_planted_defects()


# --------------------------------------------------------------------------- #
# Experiment 1 — single agent vs multi-agent (§30.1)
# --------------------------------------------------------------------------- #
def experiment_1() -> dict:
    """The graded question, answered end to end.

    Both arms get the same request, the same corpus and the same tools. The single agent is NOT
    handicapped — it holds the widest permission set any one role has. The only difference is
    that nobody checks its work.
    """
    index = get_index()

    b_usage = UsageTracker(run_id="exp1-baseline")
    started = time.perf_counter()
    b_outcome, b_evidence, _ = run_baseline(REQUEST, index, usage=b_usage, run_id="exp1-baseline")
    b_wall = time.perf_counter() - started
    b_report = b_outcome.output

    # SEQUENTIAL research, deliberately. The first attempt used the parallel fan-out and every
    # researcher was refused: three branches burst against a per-MINUTE token ceiling at the same
    # instant, and because they hammer the whole fallback chain simultaneously, none of the
    # alternates has headroom either. A 4-token health check passes while three concurrent
    # ~2k-token researchers do not.
    #
    # This is not a workaround that weakens the comparison — Experiment 3 measures parallel
    # versus sequential in isolation and found identical evidence and token counts, differing
    # only in wall clock. Using the reliable arm here keeps Experiment 1 about its actual
    # variable (one agent versus six) instead of about rate-limit luck.
    # Let the per-minute token bucket refill before the second arm. Without this the arms
    # compete: the single agent spends ~14k tokens, then the multi-agent arm starts into an
    # empty bucket and fails for reasons that have nothing to do with the variable under test.
    time.sleep(settings.inter_arm_pause_seconds)

    multi = run_workflow(REQUEST, index=index, store=None, parallel_research=False)
    m_report = multi.state.get("report")
    m_usage = multi.deps.usage.summary()

    b_md = b_report.to_markdown() if b_report else ""
    m_md = multi.report_markdown

    # A comparison in which an arm never produced a report is not a result. Saying so is the
    # difference between a finding and a table of zeros that looks like one.
    valid = bool(b_report) and multi.completed
    return {
        "request": REQUEST,
        "comparison_valid": valid,
        "invalid_reason": "" if valid else (
            f"single_agent report={'yes' if b_report else 'NO'}, "
            f"multi_agent status={multi.status.value} "
            f"({multi.state.get('abort_reason', '')[:120]}). "
            f"Arms that did not finish are not comparable; re-run when both complete."
        ),
        "single_agent": {
            "completed": b_outcome.ok,
            "wall_seconds": round(b_wall, 1),
            "billable_calls": b_usage.billable_calls,
            "input_tokens": b_usage.total_input_tokens,
            "output_tokens": b_usage.total_output_tokens,
            "evidence_stored": len(b_evidence),
            "evidence_cited": len(b_report.evidence_used) if b_report else 0,
            # The measurement that matters: the multi-agent path structurally cannot do this.
            "fabricated_citations": baseline_fabricated_citations(b_report, b_evidence)
                                    if b_report else [],
            "report_chars": len(b_md),
            "limitations_declared": len(b_report.risks_and_limitations) if b_report else 0,
            "had_review": False,
        },
        "multi_agent": {
            "completed": multi.completed,
            "wall_seconds": round(multi.wall_seconds, 1),
            "billable_calls": m_usage["billable_calls"],
            "input_tokens": m_usage["input_tokens"],
            "output_tokens": m_usage["output_tokens"],
            "evidence_stored": len(multi.state.get("evidence", [])),
            "evidence_cited": len(m_report.evidence_used) if m_report else 0,
            "fabricated_citations": (list(multi.state["fact_check"].fabricated)
                                     if multi.state.get("fact_check") else []),
            "report_chars": len(m_md),
            "limitations_declared": len(m_report.risks_and_limitations) if m_report else 0,
            "had_review": bool(multi.state.get("critic_verdicts")),
            "revisions": multi.state.get("revision_count", 0),
            "critic_problems": sum(len(v.problems) for v in multi.state.get("critic_verdicts", [])),
        },
    }


# --------------------------------------------------------------------------- #
# Experiment 2 — with vs without the Critic (§30.2)
# --------------------------------------------------------------------------- #
def _analysis_to_report(analysis, verdict, usage) -> tuple[str, int]:
    outcome = reviewers.write_report(BRIEF, analysis, EVIDENCE, HANDOFFS,
                                     verdict=verdict, usage=usage)
    report = outcome.output
    return (report.to_markdown() if report else ""), (len(report.evidence_used) if report else 0)


def experiment_2() -> dict:
    """Does the Critic change the output, and by how much?

    Both arms start from the SAME defective analysis, so the research stage cannot introduce
    variance. Four defects are planted; the measurement is how many reach the final report.
    """
    arms = {}

    for label, critic_on in (("critic_disabled", False), ("critic_enabled", True)):
        usage = UsageTracker(run_id=f"exp2-{label}")
        started = time.perf_counter()
        analysis = DEFECTIVE_ANALYSIS.model_copy(deep=True)

        fact_check = reviewers.fact_check(analysis, EVIDENCE, usage=usage)
        verdict = None
        revisions = 0
        problems = 0

        if critic_on:
            outcome = reviewers.review(BRIEF, analysis, EVIDENCE, fact_check.output,
                                       HANDOFFS, usage=usage, cycle=0)
            verdict = outcome.output
            problems = len(verdict.problems) if verdict else 0
            if verdict and not verdict.approved:
                note = ("\n".join(f"  - [{p.location}] {p.issue}" for p in verdict.problems)
                        + "\nRequired changes:\n"
                        + "\n".join(f"  - {r}" for r in verdict.required_revisions))
                revised = reviewers.analyse(BRIEF, EVIDENCE, HANDOFFS, usage=usage,
                                            revision_note=note, revision=1)
                if revised.ok and revised.output:
                    analysis = revised.output
                    revisions = 1

        markdown, cited = _analysis_to_report(analysis, verdict, usage)
        arms[label] = {
            "wall_seconds": round(time.perf_counter() - started, 1),
            "billable_calls": usage.billable_calls,
            "input_tokens": usage.total_input_tokens,
            "output_tokens": usage.total_output_tokens,
            "critic_problems_found": problems,
            "revisions": revisions,
            "fabricated_citations_detected": list(fact_check.output.fabricated)
                                             if fact_check.output else [],
            "defects_surviving": defects_surviving(analysis),
            "evidence_cited": cited,
            "report_chars": len(markdown),
        }

    off = arms["critic_disabled"]["defects_surviving"]
    on = arms["critic_enabled"]["defects_surviving"]
    arms["delta"] = {
        "planted_defects": list(DEFECT_NAMES),
        "caught_by_critic": sorted(set(off) - set(on)),
        "survived_both": sorted(set(off) & set(on)),
        "extra_calls_for_the_critic": (arms["critic_enabled"]["billable_calls"]
                                       - arms["critic_disabled"]["billable_calls"]),
        "extra_seconds_for_the_critic": round(arms["critic_enabled"]["wall_seconds"]
                                              - arms["critic_disabled"]["wall_seconds"], 1),
    }
    return arms


# --------------------------------------------------------------------------- #
# Experiment 4 — full context vs role-specific context (§30.4)
# --------------------------------------------------------------------------- #
def experiment_4() -> dict:
    """Context size measured exactly; output compared on one paired live call.

    Context assembly is deterministic, so the token question needs no API calls at all — running
    the model to measure a string length would be spending quota to learn something arithmetic
    already knows. Only "does the smaller context hurt the output?" needs a live comparison.
    """
    full = full_context_control(REQUEST, BRIEF, None, EVIDENCE, HANDOFFS,
                                DEFECTIVE_ANALYSIS, None)
    scoped = {
        "analyst": analyst_context(BRIEF, EVIDENCE, HANDOFFS),
        "critic": critic_context(BRIEF, DEFECTIVE_ANALYSIS, EVIDENCE, None, HANDOFFS),
    }

    # ~4 characters per token is the standard rough conversion; exact counts come from the
    # live calls below, so this is only for the offline comparison table.
    sizes = {
        "full_context_chars": len(full),
        "full_context_est_tokens": len(full) // 4,
        "per_agent": {
            name: {"chars": len(text), "est_tokens": len(text) // 4,
                   "reduction_vs_full": round(1 - len(text) / len(full), 3)}
            for name, text in scoped.items()
        },
    }

    # One paired live call: same Analyst, same evidence, different context assembly.
    live = {}
    for label, use_full in (("role_specific", False), ("full_context", True)):
        usage = UsageTracker(run_id=f"exp4-{label}")
        started = time.perf_counter()
        if use_full:
            from app.agents.base import structured_step
            from app.agents.prompts import ANALYST
            from app.schemas.common import AgentId

            outcome = structured_step(
                agent_id=AgentId.ANALYST, node="analyst", system=ANALYST,
                user=full_context_control(REQUEST, BRIEF, None, EVIDENCE, HANDOFFS, None, None),
                schema=AnalysisHandoff, usage=usage)
        else:
            outcome = reviewers.analyse(BRIEF, EVIDENCE, HANDOFFS, usage=usage)

        analysis = outcome.output
        known = {e.evidence_id for e in EVIDENCE}
        cited = analysis.cited_ids() if analysis else set()
        live[label] = {
            "ok": outcome.ok,
            "wall_seconds": round(time.perf_counter() - started, 1),
            "input_tokens": usage.total_input_tokens,
            "output_tokens": usage.total_output_tokens,
            "conclusions": len(analysis.conclusions) if analysis else 0,
            "major_conclusions": len(analysis.major_conclusions()) if analysis else 0,
            # Relevance proxy: citing evidence that does not exist is the clearest sign the
            # extra context degraded rather than helped.
            "cited_ids": sorted(cited),
            "fabricated_ids": sorted(cited - known),
        }

    if live["role_specific"]["input_tokens"] and live["full_context"]["input_tokens"]:
        live["token_reduction"] = round(
            1 - live["role_specific"]["input_tokens"] / live["full_context"]["input_tokens"], 3)
    return {"measured_offline": sizes, "measured_live": live}


# --------------------------------------------------------------------------- #
# Experiment 5 — revision limits 0 / 1 / 2 (§30.5)
# --------------------------------------------------------------------------- #
def experiment_5() -> dict:
    """Quality against cost as the revision cap rises.

    The same defective analysis enters every arm, so the only variable is how many times the
    Critic is allowed to send it back.
    """
    arms = {}
    for cap in (0, 1, 2):
        usage = UsageTracker(run_id=f"exp5-cap{cap}")
        started = time.perf_counter()
        analysis = DEFECTIVE_ANALYSIS.model_copy(deep=True)
        verdict = None
        cycles = 0
        problems_per_cycle = []

        for cycle in range(cap):
            fact_check = reviewers.fact_check(analysis, EVIDENCE, usage=usage)
            outcome = reviewers.review(BRIEF, analysis, EVIDENCE, fact_check.output,
                                       HANDOFFS, usage=usage, cycle=cycle)
            verdict = outcome.output
            problems_per_cycle.append(len(verdict.problems) if verdict else 0)
            if not verdict or verdict.approved:
                break
            note = ("\n".join(f"  - [{p.location}] {p.issue}" for p in verdict.problems)
                    + "\nRequired changes:\n"
                    + "\n".join(f"  - {r}" for r in verdict.required_revisions))
            revised = reviewers.analyse(BRIEF, EVIDENCE, HANDOFFS, usage=usage,
                                        revision_note=note, revision=cycle + 1)
            if not (revised.ok and revised.output):
                break
            analysis = revised.output
            cycles += 1

        markdown, cited = _analysis_to_report(analysis, verdict, usage)
        arms[f"max_revisions_{cap}"] = {
            "wall_seconds": round(time.perf_counter() - started, 1),
            "billable_calls": usage.billable_calls,
            "input_tokens": usage.total_input_tokens,
            "output_tokens": usage.total_output_tokens,
            "revisions_used": cycles,
            "problems_per_cycle": problems_per_cycle,
            "defects_surviving": defects_surviving(analysis),
            "final_verdict_approved": verdict.approved if verdict else None,
            "evidence_cited": cited,
        }
    return arms


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "1": ("exp1_single_vs_multi_agent", experiment_1, 40),
    "2": ("exp2_with_without_critic", experiment_2, 10),
    "4": ("exp4_context_strategy", experiment_4, 4),
    "5": ("exp5_revision_limits", experiment_5, 14),
}


def save(key: str, payload: dict) -> None:
    """Merge into the results file — never clobber other experiments (§7.4)."""
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev[key] = payload
    OUT.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="all", choices=[*EXPERIMENTS, "all"])
    ap.add_argument("--budget", type=int, default=300,
                    help="approximate model-call ceiling for this invocation")
    args = ap.parse_args()

    chosen = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    estimate = sum(EXPERIMENTS[k][2] for k in chosen)
    print(f"Running experiment(s) {', '.join(chosen)} — estimated ~{estimate} model calls "
          f"(budget {args.budget})\n")
    if estimate > args.budget:
        print("Estimate exceeds the budget; run fewer experiments or raise --budget.\n")
        return 1

    for key in chosen:
        name, fn, cost = EXPERIMENTS[key]
        print(f"--- Experiment {key}: {name} (~{cost} calls) ---")
        started = time.perf_counter()
        try:
            payload = fn()
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}\n")
            continue
        payload["_wall_seconds"] = round(time.perf_counter() - started, 1)
        save(name, payload)
        print(json.dumps(payload, indent=2, default=str)[:2200])
        print(f"    saved -> {OUT.relative_to(ROOT)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
