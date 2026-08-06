"""
Phase 9 acceptance check: the evaluation dataset, metrics and runner (§28, §29).

    python scripts/verify_phase9.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from eval.dataset import (  # noqa: E402
    DATASET,
    MINIMUMS,
    Category,
    Depth,
    by_category,
    by_id,
    counts,
)
from eval.metrics import CaseResult, against_targets, aggregate, score_case  # noqa: E402

# --- §28 dataset --------------------------------------------------------------
actual = counts()
check("§28 at least 25 scenarios", len(DATASET) >= 25, f"{len(DATASET)} cases")
for category, minimum in MINIMUMS.items():
    check(f"§28 {category.value} minimum {minimum}", actual[category.value] >= minimum,
          f"{actual[category.value]}")
check("case ids are unique", len({c.case_id for c in DATASET}) == len(DATASET))
check("§28 every case documents expectations and notes",
      all(c.notes and c.expected_checkpoint for c in DATASET))
check("ambiguous category has a must-NOT-clarify control",
      any(not c.expect_clarification for c in by_category(Category.AMBIGUOUS)),
      "without it, always-clarify scores 100%")
check("full-depth cases cover comparison and insufficient-evidence",
      {Category.COMPARISON, Category.INSUFFICIENT_EVIDENCE}
      <= {c.category for c in DATASET if c.depth is Depth.FULL})
check("adversarial input is in the dataset",
      any("injection" in c.tags for c in DATASET))

# --- §29 metrics are falsifiable ---------------------------------------------
empty = aggregate(DATASET, [])
check("§7.3 an empty run scores None, not 1.0",
      empty["pass_rate"] is None
      and empty["task_planning_accuracy"]["rate"] is None
      and empty["workflow_completion_rate"] is None)
check("§7.3 unmeasured targets report as unmeasured",
      all(r["met"] is None for r in against_targets(empty)))


def _perfect(case) -> CaseResult:
    return CaseResult(
        case_id=case.case_id, category=case.category.value, depth=case.depth.value,
        status="completed", clarification_requested=case.expect_clarification,
        research_tasks=0 if case.expect_clarification else case.expect_min_research_tasks,
        agents_assigned=["researcher", "analyst", "critic", "writer"],
        options_found=list(case.expect_options), plan_valid=True,
        evidence_count=5 if case.expect_evidence else 0,
        gaps_declared=2 if case.expect_gaps else 0,
        handoffs=case.expect_min_research_tasks,
        handoffs_expected=case.expect_min_research_tasks,
        critic_ran=True, report_produced=case.expect_report, report_has_limitations=True,
        checkpoints_recorded=["plan_approval", "final_review"],
        agent_calls=0 if case.expect_failure else 12, wall_seconds=30.0)


c1 = by_id("C1")
good = _perfect(c1)
good.checks = score_case(c1, good)
check("a correct run passes every applicable check", good.passed,
      ", ".join(k for k, v in good.checks.items() if not v))

# Each metric must notice a specific defect.
for label, kwargs, expected_check in [
    ("planning accuracy", {"research_tasks": 99}, "task_count_reasonable"),
    ("option identification", {"options_found": ["unrelated"]}, "identified_the_options"),
    ("routing accuracy", {"agents_assigned": ["wizard"]}, "only_known_agents_assigned"),
    ("approval compliance", {"checkpoints_recorded": []}, "approval_checkpoint_recorded"),
    ("handoff success", {"handoffs": 0, "handoffs_expected": 3}, "handoffs_completed"),
    ("fabricated citations", {"fabricated_citations": ["E9"]},
     "no_fabricated_citations_survived"),
    ("uncited major claims", {"uncited_major": 1}, "no_uncited_major_conclusions"),
    ("workflow completion", {"status": "failed"}, "workflow_completed"),
]:
    bad = _perfect(c1)
    for key, value in kwargs.items():
        setattr(bad, key, value)
    bad.checks = score_case(c1, bad)
    check(f"§7.3 {label} can fail", bad.checks.get(expected_check) is False)

a1, a4 = by_id("A1"), by_id("A4")
over = _perfect(a4)
over.clarification_requested = True
over.checks = score_case(a4, over)
check("§7.3 over-asking for clarification costs something",
      over.checks["clarification_decision"] is False)

under = _perfect(a1)
under.clarification_requested = False
under.research_tasks = 3
under.checks = score_case(a1, under)
check("§7.3 failing to clarify an ambiguous request costs something",
      under.checks["clarification_decision"] is False)

i1 = by_id("I1")
invented = _perfect(i1)
invented.evidence_count, invented.gaps_declared = 9, 0
invented.checks = score_case(i1, invented)
check("§7.3 inventing an answer to an unanswerable question fails",
      invented.checks["gaps_declared"] is False)

# --- depth accounting ---------------------------------------------------------
plan_case = by_id("S2")
plan_result = _perfect(plan_case)
plan_result.checks = score_case(plan_case, plan_result)
summary = aggregate(DATASET, [plan_result])
check("plan-depth cases do not pollute the completion rate",
      summary["workflow_completion_rate"] is None)
check("per-category breakdown is reported", "by_category" in summary)
check("unrun cases are listed rather than ignored", bool(summary["not_run"]))

# --- runner -------------------------------------------------------------------
runner = ROOT / "eval" / "run_eval.py"
check("evaluation runner exists", runner.is_file())
if runner.is_file():
    src = runner.read_text(encoding="utf-8")
    check("§7.4 runner saves after every case", "save(rows)" in src)
    check("§7.4 runner refuses to overwrite good results with an empty run",
          "Nothing to save" in src)
    check("§7.4 runner skips completed cases (resumable)",
          "c.case_id not in previous" in src)
    check("§7.5 runner budgets the quota before spending", "--budget" in src)
    check("runner can regenerate the report without running anything", "--report" in src)

# --- recorded results ---------------------------------------------------------
results = ROOT / "eval" / "results.json"
report = ROOT / "eval" / "A6_evaluation.md"
check("evaluation results recorded", results.is_file())
if results.is_file():
    data = json.loads(results.read_text(encoding="utf-8"))
    rows = data.get("results", [])
    stored = data.get("summary", {})
    check("results contain scored cases", bool(rows), f"{len(rows)} case(s)")
    check("stored summary reports a pass rate",
          stored.get("pass_rate") is not None,
          f"{stored.get('cases_passed')}/{stored.get('cases_run')}")
    # A provider-blocked case legitimately carries no checks — scoring an outage would
    # attribute the provider's quota to the system. What must never happen is a case with no
    # checks being counted as scored, so the two sets are asserted to be disjoint and complete.
    blocked = set(stored.get("blocked_by_provider", []))
    unscored = {r["case_id"] for r in rows if not r.get("checks")}
    check("every scored case carries its checks", unscored <= blocked,
          f"unscored but not blocked: {sorted(unscored - blocked)}")
    check("blocked cases are reported, not silently dropped",
          blocked <= {r["case_id"] for r in rows},
          f"blocked: {sorted(blocked)}" if blocked else "none blocked")
    check("no blocked case is counted as passed",
          not (blocked & {f["case_id"] for f in stored.get("failing_cases", [])})
          and stored.get("cases_run", 0) == len(rows) - len(blocked),
          f"{stored.get('cases_run')} scored of {len(rows)} stored, {len(blocked)} blocked")
check("evaluation report generated", report.is_file())

# --- tests --------------------------------------------------------------------
res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                     cwd=ROOT, capture_output=True, text=True)
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("full test suite passes", res.returncode == 0, tail[0])
check("§33 at least 15 automated tests",
      "passed" in tail[0] and int(tail[0].split()[0]) >= 15, tail[0])

# --- report -------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 9: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
