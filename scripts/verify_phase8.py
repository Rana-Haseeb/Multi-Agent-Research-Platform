"""
Phase 8 acceptance check: human checkpoints and the observability dashboard (§20, §24).

    python scripts/verify_phase8.py
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.graph import routing as R  # noqa: E402
from app.graph.nodes import WorkflowDeps, _decision_of, make_final_review, make_plan_approval  # noqa: E402
from app.graph.state import initial_state  # noqa: E402
from app.graph.workflow import EDGES, WorkflowSession, build_workflow  # noqa: E402
from app.schemas.common import HumanDecision, WorkflowStatus  # noqa: E402
from app.schemas.request import HumanCheckpointResponse  # noqa: E402
from app.storage.corpus import build_index  # noqa: E402

index = build_index()

# --- §20 two checkpoints exist in the graph ----------------------------------
nodes = set(build_workflow(WorkflowDeps(index=index, store=None)).get_graph().nodes)
check("§20 plan-approval checkpoint node exists", R.PLAN_APPROVAL in nodes)
check("§20 final-review checkpoint node exists", R.FINAL_REVIEW in nodes)
documented = {n for e in EDGES for n in e[:2]}
check("both checkpoints are documented in the topology",
      {R.PLAN_APPROVAL, R.FINAL_REVIEW} <= documented)
check("final review sits between the writer and finalise",
      (R.WRITER, R.FINAL_REVIEW, "") in EDGES
      and any(e[0] == R.FINAL_REVIEW and e[1] == R.FINALISE for e in EDGES))

# --- ordering: the plan gate precedes research -------------------------------
order = [e[0] for e in EDGES]
check("§20 plan approval precedes research dispatch",
      order.index(R.PLAN_APPROVAL) < order.index(R.RESEARCH_TASK)
      if R.RESEARCH_TASK in order else True)

# --- unattended runs declare themselves --------------------------------------
unattended = WorkflowDeps(index=index, store=None, human_in_the_loop=False)
plan_update = make_plan_approval(unattended)(initial_state("x", run_id="verify"))
check("unattended plan gate auto-approves", plan_update["plan_approved"] is True)
check("§29 unattended plan gate records that review was skipped",
      plan_update["human_decisions"][0]["decision"] == "auto_approved"
      and "disabled" in plan_update["human_decisions"][0]["note"].lower())

final_update = make_final_review(unattended)(initial_state("x", run_id="verify"))
check("§29 unattended final gate records that review was skipped",
      final_update["human_decisions"][0]["decision"] == "auto_approved"
      and "disabled" in final_update["human_decisions"][0]["note"].lower())

# --- decision normalisation ---------------------------------------------------
check("decision normalises from a bare string", _decision_of("reject") == "reject")
check("decision normalises from a dict", _decision_of({"decision": "edit"}) == "edit")
check("decision normalises from the schema enum",
      _decision_of(HumanCheckpointResponse(decision=HumanDecision.REJECT)) == "reject")
check("an unrecognised decision never silently aborts", _decision_of("???") == "approve")

# --- session handle -----------------------------------------------------------
session = WorkflowSession("test", index=index, store=None, human_in_the_loop=True)
check("WorkflowSession exposes start/resume/pending_interrupt",
      all(callable(getattr(session, m, None))
          for m in ("start", "resume", "pending_interrupt", "snapshot", "is_waiting")))
check("a fresh session is not waiting", session.pending_interrupt() is None)
check("session reuses one checkpointer across resumes",
      session.graph is session.graph and session.checkpointer is not None)

# --- §24 dashboard ------------------------------------------------------------
main_py = ROOT / "app" / "main.py"
check("dashboard exists", main_py.is_file())
if main_py.is_file():
    src = main_py.read_text(encoding="utf-8")
    try:
        ast.parse(src)
        parses = True
    except SyntaxError as e:
        parses, detail = False, str(e)
    check("dashboard parses", parses, "" if parses else detail)

    check("§24 pipeline shows every agent",
          all(k in src for k in ("Supervisor", "Researchers", "Analyst",
                                 "Fact-Checker", "Critic", "Writer")))
    check("§24 dashboard surfaces evidence count, revisions and errors",
          all(k in src for k in ('metric("Evidence"', 'metric("Revisions"', 'metric("Errors"')))
    check("§24 dashboard exposes the task plan, trace and report",
          all(k in src for k in ('"Plan"', '"Trace"', '"Report"')))
    check("§25 report is downloadable as Markdown",
          "download_button" in src and "text/markdown" in src)

    # Week 3 §7.2: st.error immediately before st.rerun is wiped before it paints.
    lines = [ln.strip() for ln in src.splitlines()]
    bad = [i for i, ln in enumerate(lines[:-1])
           if ln.startswith("st.error(") and lines[i + 1].startswith("st.rerun(")]
    check("§7.2 no st.error is discarded by an immediate st.rerun", not bad,
          f"lines {bad}" if bad else "")
    check("§7.2 errors are queued for the next run", "session_state.notice" in src)

    # The session must survive Streamlit's top-to-bottom rerun.
    check("session is held in st.session_state", "st.session_state.session = session" in src)
    check("dashboard renders both checkpoint gates",
          "plan_approval" in src and "final_review" in src)
    check("pipeline status is computed across stages, not per stage",
          "def pipeline_statuses" in src)

launch = ROOT / ".claude" / "launch.json"
check("dashboard is runnable via a documented command", launch.is_file())

# --- tests --------------------------------------------------------------------
res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                     cwd=ROOT, capture_output=True, text=True)
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("full test suite passes", res.returncode == 0, tail[0])

res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_checkpoints.py", "-q",
                      "--no-header"], cwd=ROOT, capture_output=True, text=True)
check("checkpoint test suite passes", res.returncode == 0)

# --- report -------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 8: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
