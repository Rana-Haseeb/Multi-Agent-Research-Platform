"""
Phase 6 acceptance check: the quality-control loop and its measurement.

The loop itself (cap, routing, termination) landed with the graph in Phase 5 and is re-checked
here so a regression in either phase is caught. What Phase 6 adds is the *measurement*: §29's
Critic Detection Rate, and the false-positive control that makes it meaningful.

    python scripts/verify_phase6.py
    python scripts/verify_phase6.py --live   # also run the benchmark against the real model
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.agents import reviewers  # noqa: E402
from app.agents.prompts import CRITIC  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph import routing as R  # noqa: E402
from app.graph.nodes import WorkflowDeps  # noqa: E402
from app.schemas.common import ReviewCriterion, Severity, WorkflowStatus  # noqa: E402
from app.schemas.handoffs import (  # noqa: E402
    AnalysisHandoff,
    Conclusion,
    CriticVerdict,
    FactCheckReport,
    Problem,
)
from app.storage.corpus import build_index  # noqa: E402
from eval.critic_bench import BRIEF, EVIDENCE, scenarios, summarise  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true", help="run the benchmark against the real model")
args = ap.parse_args()

deps = WorkflowDeps(index=build_index(), store=None)

# --- §18 the loop itself -----------------------------------------------------
check("§18 revision cap is finite", 0 <= settings.max_revision_cycles <= 3,
      f"{settings.max_revision_cycles}")
check("§18 all six review criteria defined", len(ReviewCriterion) == 6)

route = R.make_route_after_critic(deps)
reject = CriticVerdict(
    approved=False,
    problems=[Problem(location="C1", issue="weak", criterion=ReviewCriterion.RELEVANCE,
                      severity=Severity.MAJOR)],
    required_revisions=["fix"])
check("§18 terminates however long the critic rejects",
      all(route({"critic_verdicts": [reject], "revision_count": n,
                 "status": WorkflowStatus.REVISING}) == R.WRITER
          for n in range(settings.max_revision_cycles, settings.max_revision_cycles + 5)))
check("§18 rejection under the cap loops back",
      route({"critic_verdicts": [reject], "revision_count": 0,
             "status": WorkflowStatus.REVISING}) == R.REVISION)

# --- rejections must be actionable ------------------------------------------
try:
    CriticVerdict(approved=False, problems=[
        Problem(location="C1", issue="weak", criterion=ReviewCriterion.RELEVANCE,
                severity=Severity.MAJOR)])
    actionable_enforced = False
except Exception:  # noqa: BLE001
    actionable_enforced = True
check("a rejection without required_revisions is rejected by the schema", actionable_enforced)

# --- deterministic overrides -------------------------------------------------
analysis = AnalysisHandoff(
    summary="s",
    conclusions=[Conclusion(conclusion_id="C1", statement="X", evidence_ids=["E101"],
                            confidence="high", is_major=True)],
    evidence_ids_used=["E101"])
fabricated = FactCheckReport.model_validate({"checks": [
    {"conclusion_id": "C1", "citation_exists": False, "fabricated_ids": ["E9"],
     "evidence_supports": False}]})
verdict = reviewers._fallback_verdict(analysis, fabricated, 0,
                                      type("O", (), {"trace": [], "error": None,
                                                     "duration_seconds": 0.0})()).output
check("a critic outage never becomes an approval when citations are fabricated",
      not verdict.approved)
check("a critic outage on clean work discloses that no review happened",
      any("unavailable" in m.lower()
          for m in reviewers._fallback_verdict(
              analysis, None, 0,
              type("O", (), {"trace": [], "error": None,
                             "duration_seconds": 0.0})()).output.missing_evidence))

# --- the bug this phase found ------------------------------------------------
low = CRITIC.lower()
check("critic prompt scopes review to the analysis, not the report",
      "not a report" in low and "writer" in low)
check("critic prompt permits a declared gap as completeness",
      "gap" in low and "cannot be assessed" in low)

# --- §29 the benchmark -------------------------------------------------------
names = {s.name for s in scenarios()}
check("benchmark has a clean control", any(not s.should_reject for s in scenarios()))
check("benchmark covers six defect families",
      {"fabricated_citation", "contradiction", "overgeneralisation", "vendor_claim_as_fact",
       "unresearched_criterion", "irrelevant_citation"} <= names, f"{len(names)} scenarios")
check("every defective scenario declares its expected criterion and markers",
      all(s.criterion and s.markers for s in scenarios() if s.should_reject))

# the scorer must be able to fail, in both directions
approve_all = [{"scenario": s.name, "ok": True, "should_reject": s.should_reject,
                "rejected": False, "correct": s.detected(CriticVerdict(approved=True)),
                "actionable": True, "all_six_criteria_scored": True} for s in scenarios()]
reject_all = [{"scenario": s.name, "ok": True, "should_reject": s.should_reject,
               "rejected": True, "correct": s.detected(reject),
               "actionable": True, "all_six_criteria_scored": True} for s in scenarios()]
check("§7.3 an always-approving critic scores 0 detection",
      summarise(approve_all)["detection_rate"] == 0.0)
check("§7.3 an always-rejecting critic scores 100% false positives",
      summarise(reject_all)["false_positive_rate"] == 1.0)
check("§7.3 empty run yields no score rather than a perfect one",
      summarise([])["detection_rate"] is None)

# --- control fixture integrity ----------------------------------------------
clean = next(s for s in scenarios() if not s.should_reject).analysis
known = {e.evidence_id for e in EVIDENCE}
check("control cites only evidence that exists", clean.cited_ids() <= known)
check("control addresses every evaluation criterion",
      all(not (set(BRIEF.evaluation_criteria) - set(row.scores)) for row in clean.comparison))

# --- recorded results --------------------------------------------------------
results = ROOT / "eval" / "critic_bench_results.json"
check("benchmark results recorded", results.is_file())
if results.is_file():
    data = json.loads(results.read_text(encoding="utf-8"))
    stats = data.get("summary", {})
    check("recorded detection rate is perfect", stats.get("detection_rate") == 1.0,
          f"{stats.get('detected')}/{stats.get('defective_scenarios')}")
    check("recorded false-positive rate is zero", stats.get("false_positive_rate") == 0.0,
          f"{stats.get('false_positives')}/{stats.get('clean_scenarios')}")
    check("every rejection was actionable", stats.get("rejections_actionable") == 1.0)
    check("critic scored all six criteria on every run",
          stats.get("runs_scoring_all_six_criteria") == stats.get("scenarios_run"),
          f"{stats.get('runs_scoring_all_six_criteria')}/{stats.get('scenarios_run')}")

# --- live re-measurement -----------------------------------------------------
if args.live:
    res = subprocess.run([sys.executable, "eval/critic_bench.py"], cwd=ROOT,
                         capture_output=True, text=True)
    check("live benchmark run completes", res.returncode == 0,
          res.stdout.strip().splitlines()[-1] if res.stdout else res.stderr[:120])

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
print(f"\nPhase 6: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
