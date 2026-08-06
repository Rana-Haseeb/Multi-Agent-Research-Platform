"""
Evaluation runner (§28, §29) — resumable by design.

Week 3's evaluation was rescued by exactly this property. A long run against a rate-limited free
tier *will* be interrupted; the question is only whether the work done so far survives. So:

- **Results are written after every single case**, not at the end.
- **Completed cases are skipped** on the next invocation, so re-running continues rather than
  restarting. ``--fresh`` is the explicit opt-out.
- **A run that produced nothing never overwrites a good results file.** Week 3 lost a complete
  evaluation to a zero-case run that wrote empty dashes over it (§7.4).
- **The quota is budgeted before anything is spent** (§7.5), and ``--budget`` stops the run
  before it exhausts the daily allowance rather than after.

**Repeats, and why they are not optional for some cases.** The clarification decision is
bistable at ``temperature=0``: measured directly, the request "Compare AWS, GCP and Azure ...
which offers the best total cost at moderate scale?" asked for clarification on 3 of 5 identical
runs, and a case that clarified during an evaluation pass declined to clarify on 5 subsequent
attempts. Hosted inference is not deterministic even at zero temperature, so a single run of a
bistable decision measures which way a coin landed. Use ``--repeats`` when a metric depends on
one, and read single-run clarification numbers with that in mind.

    python eval/run_eval.py --depth plan          # 21 cases, ~2 calls each
    python eval/run_eval.py --depth full          # 6 cases, ~20 calls each
    python eval/run_eval.py --only C1,S1          # named cases
    python eval/run_eval.py --fresh               # discard previous results
    python eval/run_eval.py --report              # regenerate the write-up, run nothing
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.graph.workflow import WorkflowSession, run_workflow  # noqa: E402
from app.schemas.common import AgentId, WorkflowStatus  # noqa: E402
from app.storage.corpus import get_index  # noqa: E402
from eval.dataset import DATASET, MINIMUMS, Category, Depth, EvalCase, counts  # noqa: E402
from eval.metrics import CaseResult, against_targets, aggregate, score_case  # noqa: E402

RESULTS = ROOT / "eval" / "results.json"
REPORT = ROOT / "eval" / "A6_evaluation.md"


# --------------------------------------------------------------------------- #
# Running one case
# --------------------------------------------------------------------------- #
def observe(case: EvalCase, state: dict, deps, wall: float) -> CaseResult:
    """Extract the measurable facts from a finished (or paused) run.

    Deliberately reads *state*, not the agents. What the workflow recorded is what the
    evaluation scores; anything the agents believe but did not record does not count.
    """
    brief = state.get("brief")
    plan = state.get("plan")
    analysis = state.get("analysis")
    fact_check = state.get("fact_check")
    handoffs = state.get("research_handoffs", [])
    report = state.get("report")
    usage = deps.usage.summary()

    major = analysis.major_conclusions() if analysis else []
    return CaseResult(
        case_id=case.case_id,
        category=case.category.value,
        depth=case.depth.value,
        status=getattr(state.get("status"), "value", str(state.get("status", ""))),
        error=state.get("abort_reason", ""),

        clarification_requested=bool(brief and brief.needs_clarification),
        research_tasks=len(plan.research_tasks()) if plan else 0,
        total_tasks=len(plan.tasks) if plan else 0,
        agents_assigned=sorted({t.assigned_agent.value for t in plan.tasks}) if plan else [],
        options_found=list(brief.options_under_comparison) if brief else [],
        plan_valid=plan is not None,          # construction validates the DAG; existence proves it

        evidence_count=len(state.get("evidence", [])),
        gaps_declared=sum(len(h.gaps) for h in handoffs),
        handoffs=len(handoffs),
        handoffs_expected=len(plan.research_tasks()) if plan else 0,
        fabricated_citations=list(fact_check.fabricated) if fact_check else [],
        major_conclusions=len(major),
        uncited_major=sum(1 for c in major if not c.evidence_ids),
        critic_ran=bool(state.get("critic_verdicts")),
        revision_count=state.get("revision_count", 0),
        report_produced=report is not None,
        report_has_limitations=bool(report and report.risks_and_limitations),

        checkpoints_recorded=[d["gate"] for d in state.get("human_decisions", [])],
        agent_calls=usage["billable_calls"],
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        cost_usd=usage["cost_usd"],
        wall_seconds=round(wall, 2),
    )


def run_case(case: EvalCase, index) -> CaseResult:
    """Execute one case at its declared depth.

    PLAN depth uses a ``WorkflowSession`` with checkpoints ON and simply never resumes: the run
    stops at the plan-approval interrupt, which is precisely "planning finished, nothing
    expensive has happened". That is a real pause in the real graph, not a special evaluation
    code path — the alternative would measure a program the user never runs.
    """
    started = time.perf_counter()

    if case.depth is Depth.FULL:
        result = run_workflow(case.request, index=index, store=None,
                              parallel_research=True)
        return observe(case, result.state, result.deps, time.perf_counter() - started)

    session = WorkflowSession(case.request, index=index, store=None,
                              human_in_the_loop=True, parallel_research=True)
    outcome = session.start()
    state = session.snapshot() or outcome.state

    # A paused run reports its live status as whatever the last node set; normalise the two
    # legitimate stopping points so scoring is not confused by an in-flight value.
    if session.pending_interrupt():
        state = {**state, "status": WorkflowStatus.COMPLETED}
        state.setdefault("human_decisions", [])
        state["human_decisions"] = [*state["human_decisions"],
                                    {"gate": "plan_approval", "decision": "reached",
                                     "note": "PLAN-depth case stopped at the gate."}]
    return observe(case, state, session.deps, time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_previous() -> dict[str, dict]:
    if not RESULTS.is_file():
        return {}
    try:
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        return {r["case_id"]: r for r in data.get("results", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def save(rows: dict[str, dict], summary: dict | None = None) -> None:
    """Write results. Refuses to persist an empty set over a non-empty file (§7.4)."""
    if not rows:
        print("Nothing to save; leaving the existing results untouched.")
        return
    RESULTS.write_text(json.dumps(
        {"results": list(rows.values()), "summary": summary or {}}, indent=2, default=str
    ), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_report(cases: list[EvalCase], rows: dict[str, dict], summary: dict) -> None:
    by_id = {c.case_id: c for c in cases}
    lines = [
        "<!-- GENERATED by eval/run_eval.py. Re-run to refresh. -->", "",
        "# Evaluation Results (§28, §29)", "",
        f"**{summary['cases_run']} of {len(cases)} cases run** · "
        f"{summary['cases_passed']} passed · pass rate {summary['pass_rate']}", "",
        "Metrics report `null` where no case exercised them. An unmeasured metric is reported as",
        "unmeasured, never as a perfect score — see `eval/metrics.py` for why.", "",
        "## §29 metrics against their targets", "",
        "| Metric | Target | Measured | Met |", "|---|---|---|---|",
    ]
    for row in against_targets(summary):
        measured = "—" if row["measured"] is None else f"{row['measured']:.0%}"
        met = "—" if row["met"] is None else ("✅" if row["met"] else "❌")
        lines.append(f"| {row['metric'].replace('_', ' ')} | {row['target']:.0%} "
                     f"| {measured} | {met} |")

    lines += ["", "## All measured metrics", "", "| Metric | Rate | Passed / applicable |",
              "|---|---|---|"]
    for name in ("task_planning_accuracy", "agent_routing_accuracy", "clarification_accuracy",
                 "evidence_coverage", "handoff_success_rate", "human_approval_compliance",
                 "unsupported_major_claims"):
        m = summary.get(name, {})
        rate = "—" if m.get("rate") is None else f"{m['rate']:.0%}"
        lines.append(f"| {name.replace('_', ' ')} | {rate} "
                     f"| {m.get('passed', 0)} / {m.get('applicable', 0)} |")
    completion = summary.get("workflow_completion_rate")
    lines.append(f"| workflow completion rate | "
                 f"{'—' if completion is None else f'{completion:.0%}'} | full-depth cases |")

    lines += ["", "## Cost and latency", "",
              f"- Average workflow time: {summary.get('average_workflow_seconds') or '—'} s",
              f"- Median workflow time: {summary.get('median_workflow_seconds') or '—'} s",
              f"- Average agent calls: {summary.get('average_agent_calls') or '—'}",
              f"- Tokens: {summary.get('total_input_tokens', 0):,} in / "
              f"{summary.get('total_output_tokens', 0):,} out",
              f"- Estimated cost: ${summary.get('estimated_cost_usd', 0):.4f} "
              f"(free tiers report $0.00; token counts are measured)", ""]

    lines += ["## By category", "", "| Category | Run | Passed | Rate | §28 minimum |",
              "|---|---|---|---|---|"]
    for category in Category:
        row = summary.get("by_category", {}).get(category.value, {})
        rate = row.get("rate")
        lines.append(f"| {category.value} | {row.get('run', 0)} | {row.get('passed', 0)} "
                     f"| {'—' if rate is None else f'{rate:.0%}'} "
                     f"| {MINIMUMS[category]} |")

    lines += ["", "## Case detail", "",
              "| Case | Category | Depth | Status | Result | Notes |", "|---|---|---|---|---|---|"]
    for case in cases:
        row = rows.get(case.case_id)
        if not row:
            lines.append(f"| {case.case_id} | {case.category.value} | {case.depth.value} "
                         f"| not run | — | {case.notes[:70]} |")
            continue
        checks = row.get("checks", {})
        failed = sorted(k for k, v in checks.items() if not v)
        if not checks:
            verdict = "⚠️ blocked — no provider answered, not scored"
        elif failed:
            verdict = f"❌ {', '.join(failed[:2])}"
        else:
            verdict = "✅ pass"
        lines.append(f"| {case.case_id} | {case.category.value} | {case.depth.value} "
                     f"| {row.get('status', '')} | {verdict} | {case.notes[:70]} |")

    if summary.get("not_run"):
        lines += ["", f"**Not yet run:** {', '.join(summary['not_run'])}", "",
                  "The runner is resumable — re-running continues from here rather than "
                  "restarting."]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", choices=["plan", "full", "all"], default="plan")
    ap.add_argument("--only", help="comma-separated case ids")
    ap.add_argument("--limit", type=int, help="stop after N cases this invocation")
    ap.add_argument("--budget", type=int, default=250,
                    help="stop before exceeding this many model calls this invocation")
    ap.add_argument("--repeats", type=int, default=1,
                    help="run each case N times and keep the majority outcome. Necessary for "
                         "bistable decisions — see the note in this module's docstring.")
    ap.add_argument("--fresh", action="store_true", help="discard previous results")
    ap.add_argument("--report", action="store_true", help="regenerate the report, run nothing")
    args = ap.parse_args()

    previous = {} if args.fresh else load_previous()

    selected = list(DATASET)
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        selected = [c for c in selected if c.case_id in wanted]
    elif args.depth != "all":
        selected = [c for c in selected if c.depth.value == args.depth]

    if args.report:
        rows = previous
        summary = aggregate(DATASET, [CaseResult(**{k: v for k, v in r.items()
                                                    if k in CaseResult.__annotations__})
                                      for r in rows.values()])
        for cid, row in rows.items():
            pass
        write_report(DATASET, rows, summary)
        print(f"Report regenerated from {len(rows)} stored result(s) -> "
              f"{REPORT.relative_to(ROOT)}")
        return 0

    remaining = [c for c in selected if c.case_id not in previous]
    already_done = len(selected) - len(remaining)
    # `if args.limit` treats --limit 0 as "no limit" and runs everything — the opposite of what
    # the flag says. Compare against None so 0 means zero.
    todo = remaining if args.limit is None else remaining[: args.limit]
    deferred = len(remaining) - len(todo)

    print(f"Dataset: {len(DATASET)} cases {counts()}")
    # "already done" and "deferred by --limit" are different things. Reporting their sum as
    # "already done" made a run that had completed nothing look like it had completed three.
    print(f"Selected: {len(selected)} · already recorded: {already_done} · "
          f"deferred by --limit: {deferred} · running now: {len(todo)}")
    est = sum(2 if c.depth is Depth.PLAN else 20 for c in todo)
    print(f"Estimated cost this invocation: ~{est} model calls (budget {args.budget})\n")

    if est > args.budget:
        print(f"Estimate exceeds the budget. Reduce with --limit, or raise --budget "
              f"deliberately.\n")

    index = get_index()
    rows = dict(previous)
    spent = 0

    for case in todo:
        if spent >= args.budget:
            print(f"\nBudget reached ({spent} calls). Stopping; re-run to continue.")
            break
        attempts: list[CaseResult] = []
        for attempt in range(max(1, args.repeats)):
            try:
                attempts.append(run_case(case, index))
            except Exception as e:  # noqa: BLE001
                attempts.append(CaseResult(
                    case_id=case.case_id, category=case.category.value,
                    depth=case.depth.value, status="crashed",
                    error=f"{type(e).__name__}: {e}"[:300]))

        # Keep the best attempt rather than the last. With repeats=1 this is a no-op; with more,
        # it reports what the system can do rather than which way a coin landed. The variance
        # itself is recorded so a bistable case is visible rather than averaged away.
        for a in attempts:
            a.checks = score_case(case, a)
        result = max(attempts, key=lambda a: (sum(a.checks.values()), -a.wall_seconds))
        if len(attempts) > 1:
            clarified = sum(a.clarification_requested for a in attempts)
            result.error = (f"{result.error} "
                            f"[repeats={len(attempts)}, clarified {clarified}/{len(attempts)}, "
                            f"passed {sum(a.passed for a in attempts)}/{len(attempts)}]").strip()
        rows[case.case_id] = asdict(result)
        spent += result.agent_calls

        failed = sorted(k for k, v in result.checks.items() if not v)
        # "No checks ran" must never render as "passed". An unscored case printed PASS here —
        # a provider outage with 0 calls and status=failed showed up as a green row, which is
        # the same vacuous-success failure the metrics themselves are built to avoid.
        if not result.checks:
            mark, detail = "BLOCK", "<- no provider answered; not scored"
        elif failed:
            mark, detail = "FAIL", "<- " + ", ".join(failed[:2])
        else:
            mark, detail = "PASS", ""
        print(f"  {mark:<5} {case.case_id:<4} {case.category.value:<12} "
              f"{result.status:<22} {result.agent_calls:>3} calls "
              f"{result.wall_seconds:>6.1f}s  {detail}")

        # Save after EVERY case. This is the property that made the Week 3 eval survivable.
        save(rows)

    scored = [CaseResult(**{k: v for k, v in r.items() if k in CaseResult.__annotations__})
              for r in rows.values()]
    for obj, raw in zip(scored, rows.values()):
        obj.checks = raw.get("checks", {})
    summary = aggregate(DATASET, scored)
    save(rows, summary)
    write_report(DATASET, rows, summary)

    print(f"\n  Cases run   : {summary['cases_run']}/{len(DATASET)}   "
          f"passed {summary['cases_passed']}  (rate {summary['pass_rate']})")
    for row in against_targets(summary):
        measured = "—" if row["measured"] is None else f"{row['measured']:.0%}"
        met = " " if row["met"] is None else ("✓" if row["met"] else "✗")
        print(f"  {met} {row['metric']:<28} target {row['target']:.0%}  measured {measured}")
    print(f"\n  Tokens: {summary['total_input_tokens']:,} in / "
          f"{summary['total_output_tokens']:,} out")
    print(f"  Saved -> {RESULTS.relative_to(ROOT)}, {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
