"""
Phase 7 acceptance check: parallel research fan-out (§19).

    python scripts/verify_phase7.py
    python scripts/verify_phase7.py --live   # also re-run Experiment 3 against the API
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import Send  # noqa: E402

from app.config import settings  # noqa: E402
from app.graph import routing as R  # noqa: E402
from app.graph.nodes import WorkflowDeps, make_research_task  # noqa: E402
from app.graph.state import WorkflowState, evidence_id_for  # noqa: E402
from app.graph.workflow import EDGES, build_workflow  # noqa: E402
from app.schemas.common import AgentId, TaskStatus, WorkflowStatus  # noqa: E402
from app.schemas.request import RequestBrief  # noqa: E402
from app.schemas.tasks import Task, TaskPlan  # noqa: E402
from app.services.usage import UsageTracker  # noqa: E402
from app.storage.corpus import build_index  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--live", action="store_true")
args = ap.parse_args()

index = build_index()
QS = ["What state management does LangGraph provide?",
      "How does CrewAI handle human approval?",
      "What did the benchmark measure?"]


def state(n=3, done=None):
    tasks = [Task(task_id=f"R{i+1}", description="r", assigned_agent=AgentId.RESEARCHER,
                  research_question=q) for i, q in enumerate(QS[:n])]
    return {"status": WorkflowStatus.RESEARCHING, "plan_approved": True,
            "plan": TaskPlan(tasks=tasks),
            "brief": RequestBrief(objective="Compare frameworks", sub_questions=QS[:n]),
            "task_status": done or {}, "evidence": [], "errors": []}


# --- topology ----------------------------------------------------------------
nodes = set(build_workflow(WorkflowDeps(index=index, store=None)).get_graph().nodes)
check("fan-out target node exists", R.RESEARCH_TASK in nodes)
check("sequential dispatch node retained as the experiment control", R.RESEARCH in nodes)
documented = {n for e in EDGES for n in e[:2]}
check("both research arms are documented", {R.RESEARCH, R.RESEARCH_TASK} <= documented)

shapes = {tuple(sorted(build_workflow(
    WorkflowDeps(index=index, store=None, parallel_research=f)).get_graph().nodes))
    for f in (False, True)}
check("§30 both arms are the same program under a flag", len(shapes) == 1)

# --- routing -----------------------------------------------------------------
seq_route = R.make_route_after_plan_approval(WorkflowDeps(index=index, parallel_research=False))
par_route = R.make_route_after_plan_approval(WorkflowDeps(index=index, parallel_research=True))

check("sequential flag routes to the dispatch node", seq_route(state()) == R.RESEARCH)
sends = par_route(state(3))
check("§19 parallel flag emits one Send per research task",
      isinstance(sends, list) and len(sends) == 3
      and all(isinstance(s, Send) for s in sends), f"{len(sends) if isinstance(sends, list) else sends}")
check("every Send targets the research task node",
      all(s.node == R.RESEARCH_TASK for s in sends))
check("§21 each Send carries only its own question, no shared state",
      all(set(s.arg) == {"task_id", "research_question", "objective"} for s in sends))
check("branches receive distinct questions",
      len({s.arg["research_question"] for s in sends}) == 3)
check("completed tasks are not re-dispatched",
      {s.arg["task_id"] for s in par_route(state(3, {"R1": TaskStatus.COMPLETED}))} == {"R2", "R3"})
check("fan-out respects the parallel researcher cap",
      len(par_route(state(3))) <= settings.max_parallel_researchers)
check("a failed run never dispatches",
      par_route({**state(), "status": WorkflowStatus.FAILED}) == END)

# --- concurrency, measured ---------------------------------------------------
import app.graph.nodes as nodes_mod  # noqa: E402

original = nodes_mod.research_one
starts: list[float] = []
lock = threading.Lock()


def slow(_deps, task_id, question, objective):
    with lock:
        starts.append(time.perf_counter())
    time.sleep(0.4)
    return {"evidence": [], "research_handoffs": [], "errors": [], "trace": [],
            "task_status": {task_id: TaskStatus.COMPLETED}}


nodes_mod.research_one = slow
try:
    deps = WorkflowDeps(index=index, store=None, parallel_research=True)
    g = StateGraph(WorkflowState)
    g.add_node("gate", lambda s: {})
    g.add_node(R.RESEARCH_TASK, make_research_task(deps))
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", R.make_route_after_plan_approval(deps), [R.RESEARCH_TASK, END])
    g.add_edge(R.RESEARCH_TASK, END)
    t0 = time.perf_counter()
    out = g.compile().invoke(state(3))
    elapsed = time.perf_counter() - t0
finally:
    nodes_mod.research_one = original

check("§19 fan-out is genuinely concurrent, not sequential dispatch",
      elapsed < 0.9, f"{elapsed:.2f}s for 3 x 0.4s tasks (sequential would be >=1.2s)")
check("all branches started together",
      len(starts) == 3 and (max(starts) - min(starts)) < 0.3)
check("every branch completed", len(out["task_status"]) == 3)

# --- reducers hold under the real fan-out ------------------------------------
from tests.conftest import make_evidence  # noqa: E402

nodes_mod.research_one = lambda _d, tid, q, o: {
    "evidence": [make_evidence(evidence_id_for(tid, i), question=q) for i in (1, 2)],
    "research_handoffs": [], "errors": [], "trace": [],
    "task_status": {tid: TaskStatus.COMPLETED}}
try:
    merged = g.compile().invoke(state(3))
finally:
    nodes_mod.research_one = original
check("concurrent evidence writes all survive the merge",
      len(merged["evidence"]) == 6, f"{len(merged['evidence'])} of 6")
check("evidence ids from concurrent branches do not collide",
      len({e.evidence_id for e in merged["evidence"]}) == 6)

# --- thread safety -----------------------------------------------------------
usage = UsageTracker(run_id="verify")


def hammer():
    for _ in range(200):
        usage.record(agent_id="researcher", provider="groq", model="m",
                     input_tokens=1, output_tokens=1)


threads = [threading.Thread(target=hammer) for _ in range(6)]
for t in threads:
    t.start()
for t in threads:
    t.join()
check("usage tracker loses no writes under concurrency", usage.total_calls == 1200,
      f"{usage.total_calls} of 1200")

# --- experiment harness ------------------------------------------------------
exp = ROOT / "experiments" / "exp3_parallel_research.py"
check("Experiment 3 harness exists", exp.is_file())
res = subprocess.run([sys.executable, str(exp), "--simulate", "--tasks", "3"],
                     cwd=ROOT, capture_output=True, text=True)
check("Experiment 3 harness self-validates with no API calls", res.returncode == 0)

results = ROOT / "experiments" / "results.json"
if results.is_file():
    data = json.loads(results.read_text(encoding="utf-8"))
    sim = data.get("exp3_parallel_research_simulated", {}).get("summary", {})
    if sim:
        check("simulated speedup approaches the theoretical ceiling",
              sim.get("speedup", 0) >= sim.get("theoretical_max", 3) * 0.8,
              f"{sim.get('speedup')}x of {sim.get('theoretical_max')}x")
    live = data.get("exp3_parallel_research", {}).get("summary", {})
    check("Experiment 3 measured against the live API", bool(live),
          f"{live.get('speedup')}x speedup" if live else "not yet run")
    if live:
        check("parallelism changed latency, not evidence gathered",
              abs(live.get("evidence_parallel", 0) - live.get("evidence_sequential", 0))
              <= max(2, 0.5 * max(1, live.get("evidence_sequential", 1))),
              f"{live.get('evidence_sequential')} seq vs {live.get('evidence_parallel')} par")

if args.live:
    res = subprocess.run([sys.executable, str(exp), "--tasks", "3"],
                         cwd=ROOT, capture_output=True, text=True)
    check("live Experiment 3 run completes", res.returncode == 0)

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
print(f"\nPhase 7: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
