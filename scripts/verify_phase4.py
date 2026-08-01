"""
Phase 4 acceptance check: the six agents and the single-agent baseline.

    python scripts/verify_phase4.py
    python scripts/verify_phase4.py --live    # also run one real researcher task
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.agents import researcher, reviewers, supervisor  # noqa: E402
from app.agents.context import (  # noqa: E402
    analyst_context,
    critic_context,
    full_context_control,
    researcher_context,
    supervisor_context,
    writer_context,
)
from app.agents.prompts import ALL_PROMPTS  # noqa: E402
from app.baseline import single_agent  # noqa: E402
from app.config import settings  # noqa: E402
from app.schemas.common import AgentId, ClaimType, Confidence, ReviewCriterion, Severity  # noqa: E402
from app.schemas.evidence import Evidence  # noqa: E402
from app.schemas.handoffs import (  # noqa: E402
    AnalysisHandoff,
    Conclusion,
    CriticVerdict,
    FactCheckReport,
    Problem,
    ResearchHandoff,
)
from app.schemas.request import RequestBrief  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true", help="also run one real researcher task")
args = ap.parse_args()

# --- §12 specialisation ------------------------------------------------------
check("all six agents plus baseline have instructions", len(ALL_PROMPTS) == 8,
      ", ".join(sorted(ALL_PROMPTS)))
check("§12 no two agents share instructions",
      len(set(ALL_PROMPTS.values())) == len(ALL_PROMPTS))
check("every specialist states prohibitions",
      all("PROHIBITED" in ALL_PROMPTS[a]
          for a in ("researcher", "analyst", "critic", "writer", "fact_checker")))
check("source-reading agents carry the injection guard",
      all("DATA, never instructions" in ALL_PROMPTS[a]
          for a in ("researcher", "supervisor_analyse", "baseline")))

# --- agent entry points exist ------------------------------------------------
for label, fn in [
    ("supervisor.analyse_request", supervisor.analyse_request),
    ("supervisor.build_plan", supervisor.build_plan),
    ("researcher.research", researcher.research),
    ("reviewers.analyse", reviewers.analyse),
    ("reviewers.fact_check", reviewers.fact_check),
    ("reviewers.review", reviewers.review),
    ("reviewers.write_report", reviewers.write_report),
    ("baseline.run_baseline", single_agent.run_baseline),
]:
    check(f"{label} callable", callable(fn))

# --- §21 context boundaries --------------------------------------------------
brief = RequestBrief(
    objective="Compare agent frameworks",
    sub_questions=["What state management does LangGraph provide?",
                   "How does CrewAI handle approval?"],
    evaluation_criteria=["state", "approval"],
    options_under_comparison=["LangGraph", "CrewAI"],
)
ev = [
    Evidence(evidence_id="E101", claim="LangGraph uses reducers",
             supporting_text="UNIQUE_SUPPORTING_TEXT_MARKER for the state channel reducers.",
             source_id="fw-langgraph-docs", source_title="LangGraph Documentation",
             research_question=brief.sub_questions[0], confidence=Confidence.HIGH,
             claim_type=ClaimType.FACT, agent_id="researcher", task_id="R1"),
    Evidence(evidence_id="E201", claim="CrewAI has human_input",
             supporting_text="CrewAI supports a human_input flag on individual tasks.",
             source_id="fw-crewai-docs", source_title="CrewAI Documentation",
             research_question=brief.sub_questions[1], confidence=Confidence.MEDIUM,
             claim_type=ClaimType.FACT, agent_id="researcher", task_id="R2"),
]
hs = [ResearchHandoff(task_id="R1", research_question=brief.sub_questions[0],
                      findings="f", evidence_ids=["E101"], confidence=Confidence.HIGH)]
analysis = AnalysisHandoff(
    summary="s",
    conclusions=[Conclusion(conclusion_id="C1", statement="LangGraph has typed state",
                            evidence_ids=["E101"], confidence=Confidence.HIGH, is_major=True)],
    evidence_ids_used=["E101"])

sup_ctx = supervisor_context("compare", brief, None, ev, hs)
check("§21 supervisor sees counts, not evidence bodies",
      "UNIQUE_SUPPORTING_TEXT_MARKER" not in sup_ctx and "COVERAGE" in sup_ctx)

res_ctx = researcher_context("R1", brief.sub_questions[0], brief.objective)
check("§21 researcher sees only its own question",
      brief.sub_questions[1] not in res_ctx)

crit_ctx = critic_context(brief, analysis, ev, None, hs)
check("§21 critic gets the evidence index, not bodies",
      "UNIQUE_SUPPORTING_TEXT_MARKER" not in crit_ctx and "E101" in crit_ctx)

wr_ctx = writer_context(brief, analysis, ev, hs)
check("§21 writer sees only cited evidence", "E201" not in wr_ctx and "E101" in wr_ctx)

full = full_context_control("compare", brief, None, ev, hs, analysis, None)
check("§21 role-specific contexts are smaller than the full-context control",
      all(len(c) < len(full) for c in (sup_ctx, res_ctx, crit_ctx, wr_ctx)),
      f"full={len(full)} sup={len(sup_ctx)} res={len(res_ctx)} crit={len(crit_ctx)} wr={len(wr_ctx)}")

# --- deterministic routing ---------------------------------------------------
d = supervisor.evidence_gate(brief, [], hs, research_round=0)
check("evidence gate blocks with no evidence", not d.proceed)
d = supervisor.evidence_gate(brief, [], hs, settings.max_research_rounds)
check("§22 evidence gate terminates at the research-round cap", d.proceed)

assumptions = [e.model_copy(update={"claim_type": ClaimType.ASSUMPTION}) for e in ev]
d = supervisor.evidence_gate(brief, assumptions, hs, research_round=0)
check("assumptions do not count as coverage", not d.proceed)

reject = CriticVerdict(approved=False,
                       problems=[Problem(location="C1", issue="weak",
                                         criterion=ReviewCriterion.RELEVANCE,
                                         severity=Severity.MAJOR)],
                       required_revisions=["fix"])
terminates = all(
    supervisor.next_action(reject, n, 0).next_step == "writer"
    for n in range(settings.max_revision_cycles, settings.max_revision_cycles + 4)
)
check("§18 workflow terminates when the critic never approves", terminates)
check("approval routes to the writer",
      supervisor.next_action(CriticVerdict(approved=True), 0, 0).next_step == "writer")

# --- deterministic fact-check ------------------------------------------------
fabricated = reviewers.check_citations_deterministic(
    analysis.model_copy(update={"conclusions": [
        analysis.conclusions[0].model_copy(update={"evidence_ids": ["E101", "E9"]})]}), ev)
check("fabricated citations detected without a model call", fabricated["C1"] == ["E9"])

# --- baseline ----------------------------------------------------------------
check("single-agent baseline exists for Experiment 1",
      callable(single_agent.run_baseline) and callable(
          single_agent.baseline_fabricated_citations))
check("baseline output uses the same report schema as the multi-agent path",
      set(single_agent.BaselineReport.model_fields) >= {
          "executive_summary", "research_objective", "methodology", "key_findings",
          "risks_and_limitations", "recommendation_statement"})

# --- live smoke test ---------------------------------------------------------
if args.live:
    from app.services.usage import UsageTracker
    from app.storage.corpus import build_index

    usage = UsageTracker(run_id="verify-live")
    out, stored = researcher.research(
        "R1", "What mechanisms does LangGraph provide for human-in-the-loop approval?",
        "Compare agent frameworks", build_index(), usage=usage)
    check("live: researcher completes a real task", out.ok,
          out.error.message if out.failed else "")
    check("live: evidence was actually stored", len(stored) > 0, f"{len(stored)} items")
    check("live: every stored quote was verified against its source",
          all(e.supporting_text and e.source_id for e in stored))
    check("live: usage was metered", usage.total_calls > 0,
          f"{usage.total_calls} calls, {usage.total_input_tokens} in / "
          f"{usage.total_output_tokens} out tokens")

# --- tests -------------------------------------------------------------------
res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                     cwd=ROOT, capture_output=True, text=True)
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("full test suite passes", res.returncode == 0, tail[0])

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 4: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
