"""
Phase 1 acceptance check: schemas, shared state, handoff contracts.

Checks the spec requirements against the actual code rather than against a build note.

    python scripts/verify_phase1.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.graph.state import (  # noqa: E402
    FIELD_PERMISSIONS,
    PARALLEL_WRITE_CHANNELS,
    WorkflowState,
    initial_state,
)
from app.schemas.common import AgentId, ReviewCriterion, WorkflowStatus  # noqa: E402
from app.schemas.evidence import Evidence  # noqa: E402
from app.schemas.handoffs import (  # noqa: E402
    AnalysisHandoff,
    CriticVerdict,
    FactCheckReport,
    ResearchHandoff,
)
from app.schemas.reports import FinalReport, TraceEvent  # noqa: E402
from app.schemas.request import RequestBrief  # noqa: E402
from app.schemas.tasks import TaskPlan  # noqa: E402

# --- §15 evidence model: the nine required fields ----------------------------
REQUIRED_EVIDENCE = {
    "evidence_id", "claim", "supporting_text", "source_id", "source_title",
    "retrieved_at", "research_question", "confidence", "agent_id",
}
missing = REQUIRED_EVIDENCE - set(Evidence.model_fields)
check("§15 evidence model has all required fields", not missing, f"missing: {missing}" if missing else "")
check("§2 evidence distinguishes fact/claim/assumption/missing", "claim_type" in Evidence.model_fields)

# --- §16 shared state: the required contents ---------------------------------
REQUIRED_STATE = {
    "user_request", "clarifications", "brief", "plan", "task_status", "evidence",
    "analysis", "critic_verdicts", "revision_count", "report", "errors", "status",
}
missing = REQUIRED_STATE - set(WorkflowState.__annotations__)
check("§16 state has all required fields", not missing, f"missing: {missing}" if missing else "")
check("§27 permission table covers every field",
      set(FIELD_PERMISSIONS) == set(WorkflowState.__annotations__))

# --- §19 parallel safety -----------------------------------------------------
hints = get_type_hints(WorkflowState, include_extras=True)
no_reducer = [c for c in PARALLEL_WRITE_CHANNELS if not hasattr(hints.get(c), "__metadata__")]
check("§19 every parallel channel has a reducer", not no_reducer, f"no reducer: {no_reducer}" if no_reducer else "")

# --- §17 the seven handoff contracts exist and validate ----------------------
for label, model, required in [
    ("RequestBrief", RequestBrief, {"objective", "sub_questions", "needs_clarification"}),
    ("TaskPlan", TaskPlan, {"tasks"}),
    ("ResearchHandoff", ResearchHandoff,
     {"research_question", "findings", "evidence_ids", "confidence", "gaps"}),
    ("AnalysisHandoff", AnalysisHandoff, {"conclusions", "assumptions", "summary"}),
    ("FactCheckReport", FactCheckReport, {"checks"}),
    ("CriticVerdict", CriticVerdict,
     {"approved", "problems", "missing_evidence", "required_revisions"}),
    ("FinalReport", FinalReport,
     {"executive_summary", "research_objective", "methodology", "key_findings",
      "risks_and_limitations", "recommendation", "evidence_used"}),
]:
    miss = required - set(model.model_fields)
    check(f"§17 {label} contract complete", not miss, f"missing: {miss}" if miss else "")

# --- §18 the six review criteria ---------------------------------------------
check("§18 critic has all six review criteria", len(ReviewCriterion) == 6,
      ", ".join(c.value for c in ReviewCriterion))

# --- §23 trace stores operations, not reasoning ------------------------------
banned = {"reasoning", "thought", "thinking", "chain_of_thought", "rationale", "scratchpad"}
leaked = banned & set(TraceEvent.model_fields)
check("§23 trace stores no chain-of-thought field", not leaked, f"found: {leaked}" if leaked else "")

# --- §25 report has the eight required sections ------------------------------
report = FinalReport(title="T", executive_summary="e", research_objective="r", methodology="m")
md = report.to_markdown()
sections = ["Executive Summary", "Research Objective", "Methodology", "Key Findings",
            "Risks and Limitations", "Recommendation", "Evidence and References"]
absent = [s for s in sections if f"## {s}" not in md]
check("§25 markdown export has required sections", not absent, f"missing: {absent}" if absent else "")

# --- agents / statuses -------------------------------------------------------
check("six specialist agents defined", len(AgentId.llm_agents()) == 6,
      ", ".join(sorted(a.value for a in AgentId.llm_agents())))
check("three human checkpoints are distinct states",
      sum(s.is_waiting_for_human() for s in WorkflowStatus) == 3)

# --- state initialises cleanly -----------------------------------------------
s = initial_state("compare three things")
check("initial state has no None collections",
      all(s[k] == [] for k in ("evidence", "trace", "errors", "clarifications")))
check("initial status is PENDING", s["status"] is WorkflowStatus.PENDING)

# --- generated docs exist and are current ------------------------------------
for doc in ("docs/A2_state_specification.md", "docs/A3_handoff_contracts.md"):
    p = ROOT / doc
    check(f"{doc} generated", p.is_file() and "GENERATED by scripts/gen_specs.py" in
          p.read_text(encoding="utf-8"))

# --- the test suite actually passes ------------------------------------------
res = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_schemas.py", "-q", "--no-header"],
    cwd=ROOT, capture_output=True, text=True,
)
passed = "failed" not in res.stdout.lower() and res.returncode == 0
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("schema test suite passes", passed, tail[0])

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 1: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
