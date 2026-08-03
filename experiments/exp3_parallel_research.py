"""
Experiment 3 — sequential versus parallel research (§30).

Measures the research phase **in isolation** rather than two whole workflows. That is not a
shortcut, it is the better design: research dispatch is the only stage the flag changes, so
timing the full pipeline would bury a real effect under variance from planning, analysis, the
critic loop and revision count — stages whose latency has nothing to do with the variable and
which differ run to run by more than the effect being measured.

It also costs roughly a quarter as many tokens, which matters on a rate-limited free tier where
a careless experiment design is the difference between a measurement and an exhausted quota.

Both arms run the identical researchers over the identical questions against the identical
corpus. The only difference is whether the branches are dispatched through ``Send()``.

    python experiments/exp3_parallel_research.py
    python experiments/exp3_parallel_research.py --tasks 2 --repeats 2
    python experiments/exp3_parallel_research.py --simulate   # no API calls, proves the harness
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.graph import END, START, StateGraph  # noqa: E402

from app.graph import routing as R  # noqa: E402
from app.graph.nodes import WorkflowDeps, make_research_dispatch, make_research_task  # noqa: E402
from app.graph.state import WorkflowState  # noqa: E402
from app.schemas.common import AgentId, TaskStatus, WorkflowStatus  # noqa: E402
from app.schemas.request import RequestBrief  # noqa: E402
from app.schemas.tasks import Task, TaskPlan  # noqa: E402
from app.services.usage import UsageTracker  # noqa: E402
from app.storage.corpus import build_index  # noqa: E402

OUT = ROOT / "experiments" / "results.json"

QUESTIONS = [
    "What mechanisms does LangGraph provide for human-in-the-loop approval?",
    "How does CrewAI handle state between tasks?",
    "What did the independent benchmark measure for framework latency?",
    "What compliance certifications do the platform-as-a-service providers hold?",
]

OBJECTIVE = "Compare agent frameworks for a small Python team"


def build_state(n_tasks: int) -> dict:
    tasks = [
        Task(task_id=f"R{i + 1}", description=f"Research: {q}",
             assigned_agent=AgentId.RESEARCHER, research_question=q)
        for i, q in enumerate(QUESTIONS[:n_tasks])
    ]
    return {
        "status": WorkflowStatus.RESEARCHING,
        "plan_approved": True,
        "plan": TaskPlan(tasks=tasks),
        "brief": RequestBrief(objective=OBJECTIVE, sub_questions=QUESTIONS[:n_tasks]),
        "task_status": {},
        "evidence": [],
        "research_handoffs": [],
        "trace": [],
        "errors": [],
    }


def build_arm(deps: WorkflowDeps):
    """A minimal graph containing only the stage under test."""
    g = StateGraph(WorkflowState)
    g.add_node("dispatch", lambda s: {})
    g.add_node(R.RESEARCH, make_research_dispatch(deps))
    g.add_node(R.RESEARCH_TASK, make_research_task(deps))
    g.add_edge(START, "dispatch")
    g.add_conditional_edges("dispatch", R.make_route_after_plan_approval(deps),
                            [R.RESEARCH, R.RESEARCH_TASK, END])
    g.add_edge(R.RESEARCH, END)
    g.add_edge(R.RESEARCH_TASK, END)
    return g.compile()


def run_arm(parallel: bool, n_tasks: int, index, simulate: bool) -> dict:
    usage = UsageTracker(run_id=f"exp3-{'par' if parallel else 'seq'}")
    deps = WorkflowDeps(index=index, store=None, usage=usage,
                        run_id=usage.run_id, parallel_research=parallel)

    if simulate:
        # Proves the harness measures what it claims without spending a single token: a fixed
        # 0.5s of "work" per task makes the expected speedup exactly known in advance.
        import app.graph.nodes as nodes

        original = nodes.research_one

        def stub(_deps, task_id, question, objective):
            time.sleep(0.5)
            return {"evidence": [], "research_handoffs": [], "errors": [], "trace": [],
                    "task_status": {task_id: TaskStatus.COMPLETED}}

        nodes.research_one = stub
        try:
            started = time.perf_counter()
            out = build_arm(deps).invoke(build_state(n_tasks))
            elapsed = time.perf_counter() - started
        finally:
            nodes.research_one = original
    else:
        started = time.perf_counter()
        out = build_arm(deps).invoke(build_state(n_tasks))
        elapsed = time.perf_counter() - started

    completed = sum(1 for s in out.get("task_status", {}).values()
                    if s is TaskStatus.COMPLETED)
    return {
        "parallel": parallel,
        "wall_seconds": round(elapsed, 2),
        "tasks": n_tasks,
        "tasks_completed": completed,
        "evidence": len(out.get("evidence", [])),
        "billable_calls": usage.billable_calls,
        "attempted_calls": usage.total_calls,
        "refused_calls": usage.refused_calls,
        "input_tokens": usage.total_input_tokens,
        "output_tokens": usage.total_output_tokens,
        "errors": len(out.get("errors", [])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--simulate", action="store_true",
                    help="fixed-cost stub researchers; no API calls")
    args = ap.parse_args()

    index = build_index()
    rows: list[dict] = []

    label = "SIMULATED (0.5s/task stub)" if args.simulate else "LIVE"
    print(f"Experiment 3 — sequential vs parallel research  [{label}]")
    print(f"{args.tasks} research tasks, {args.repeats} repeat(s) per arm\n")
    print(f"{'arm':<12} {'wall':>8} {'tasks':>7} {'evidence':>9} {'calls':>7} "
          f"{'refused':>8} {'tokens in':>10}")
    print("-" * 70)

    for repeat in range(args.repeats):
        for parallel in (False, True):
            row = run_arm(parallel, args.tasks, index, args.simulate)
            row["repeat"] = repeat + 1
            rows.append(row)
            print(f"{'parallel' if parallel else 'sequential':<12} "
                  f"{row['wall_seconds']:>7.2f}s {row['tasks_completed']:>7} "
                  f"{row['evidence']:>9} {row['billable_calls']:>7} "
                  f"{row['refused_calls']:>8} {row['input_tokens']:>10}")

    seq = [r["wall_seconds"] for r in rows if not r["parallel"]]
    par = [r["wall_seconds"] for r in rows if r["parallel"]]
    seq_med, par_med = statistics.median(seq), statistics.median(par)
    speedup = round(seq_med / par_med, 2) if par_med else None

    seq_ev = sum(r["evidence"] for r in rows if not r["parallel"])
    par_ev = sum(r["evidence"] for r in rows if r["parallel"])
    seq_tok = sum(r["input_tokens"] for r in rows if not r["parallel"])
    par_tok = sum(r["input_tokens"] for r in rows if r["parallel"])

    print()
    print(f"  Sequential median : {seq_med:.2f}s")
    print(f"  Parallel median   : {par_med:.2f}s")
    print(f"  Speedup           : {speedup}x   (theoretical ceiling {args.tasks}x)")
    print(f"  Evidence gathered : {seq_ev} sequential vs {par_ev} parallel")
    print(f"  Input tokens      : {seq_tok} sequential vs {par_tok} parallel")
    print()
    print("  Parallelism should change latency, not output. A large evidence or token gap")
    print("  between the arms means something other than dispatch order differed.")

    summary = {
        "tasks": args.tasks, "repeats": args.repeats, "simulated": args.simulate,
        "sequential_median_s": round(seq_med, 2), "parallel_median_s": round(par_med, 2),
        "speedup": speedup, "theoretical_max": args.tasks,
        "evidence_sequential": seq_ev, "evidence_parallel": par_ev,
        "input_tokens_sequential": seq_tok, "input_tokens_parallel": par_tok,
    }
    if rows:
        prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        key = "exp3_parallel_research" + ("_simulated" if args.simulate else "")
        prev[key] = {"summary": summary, "rows": rows}
        OUT.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
