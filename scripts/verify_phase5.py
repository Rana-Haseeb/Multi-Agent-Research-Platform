"""
Phase 5 acceptance check: the orchestration graph.

    python scripts/verify_phase5.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.config import settings  # noqa: E402
from app.graph import routing as R  # noqa: E402
from app.graph.nodes import WorkflowDeps  # noqa: E402
from app.graph.workflow import build_workflow, describe_topology  # noqa: E402
from app.schemas.common import ReviewCriterion, Severity, WorkflowStatus  # noqa: E402
from app.schemas.handoffs import CriticVerdict, Problem  # noqa: E402
from app.storage.corpus import build_index  # noqa: E402

index = build_index()
deps = WorkflowDeps(index=index, store=None)

# --- topology ----------------------------------------------------------------
graph = build_workflow(deps)
nodes = set(graph.get_graph().nodes)
expected = {R.INTAKE, R.ANALYSE, R.PLAN, R.PLAN_APPROVAL, R.RESEARCH, R.EVIDENCE_GATE,
            R.ANALYST, R.FACT_CHECKER, R.CRITIC, R.REVISION, R.WRITER, R.FINALISE}
missing = expected - nodes
check("graph compiles with all 12 nodes", not missing, f"missing: {missing}" if missing else "")
check("topology is documented", all(n in describe_topology() for n in expected))

# --- §8 orchestration: every agent has a node --------------------------------
for label, node in [("supervisor (analyse)", R.ANALYSE), ("supervisor (plan)", R.PLAN),
                    ("researcher", R.RESEARCH), ("analyst", R.ANALYST),
                    ("fact-checker", R.FACT_CHECKER), ("critic", R.CRITIC),
                    ("writer", R.WRITER)]:
    check(f"§5 {label} wired into the graph", node in nodes)

# --- §20 checkpoint nodes exist even before Phase 8 --------------------------
check("§20 plan-approval checkpoint node exists", R.PLAN_APPROVAL in nodes)
check("§20 three human-wait states are distinct",
      sum(s.is_waiting_for_human() for s in WorkflowStatus) == 3)

# --- routing is deterministic and terminating --------------------------------
route_critic = R.make_route_after_critic(deps)
route_gate = R.make_route_after_gate(deps)
route_revision = R.make_route_after_revision(deps)

reject = CriticVerdict(
    approved=False,
    problems=[Problem(location="C1", issue="weak", criterion=ReviewCriterion.RELEVANCE,
                      severity=Severity.MAJOR)],
    required_revisions=["fix"])

check("approval routes to the writer",
      route_critic({"critic_verdicts": [CriticVerdict(approved=True)],
                    "status": WorkflowStatus.REVIEWING}) == R.WRITER)
check("rejection under the cap routes to revision",
      route_critic({"critic_verdicts": [reject], "revision_count": 0,
                    "status": WorkflowStatus.REVISING}) == R.REVISION)

terminates = all(
    route_critic({"critic_verdicts": [reject], "revision_count": n,
                  "status": WorkflowStatus.REVISING}) == R.WRITER
    for n in range(settings.max_revision_cycles, settings.max_revision_cycles + 5)
)
check("§18 loop terminates at the revision cap however long the critic rejects", terminates)

check("§22 gate never re-plans past the research round cap",
      route_gate({"status": WorkflowStatus.PLANNING,
                  "research_round": settings.max_research_rounds}) == R.ANALYST)
check("a failed status short-circuits every router",
      all(fn({"status": WorkflowStatus.FAILED}) == "__end__"
          for fn in (R.route_after_intake, R.route_after_analyse, R.route_after_plan,
                     R.route_after_analyst, R.route_after_fact_check, R.route_after_writer,
                     route_critic, route_gate, route_revision)))
check("budget backstop routes to the writer",
      route_critic({"critic_verdicts": [reject], "revision_count": 0,
                    "agent_calls": settings.max_agent_calls_per_run,
                    "status": WorkflowStatus.REVISING}) == R.WRITER)

# --- experiment levers are configuration, not code ---------------------------
shapes = set()
for kwargs in ({}, {"critic_enabled": False}, {"max_revisions": 0},
               {"parallel_research": True}, {"full_context": True}):
    shapes.add(tuple(sorted(build_workflow(
        WorkflowDeps(index=index, store=None, **kwargs)).get_graph().nodes)))
check("§30 experiment flags do not change the graph shape", len(shapes) == 1,
      f"{len(shapes)} distinct topologies")
check("§30 revision cap is overridable per run",
      WorkflowDeps(index=index, max_revisions=0).revision_cap() == 0
      and WorkflowDeps(index=index).revision_cap() == settings.max_revision_cycles)

# --- reliability settings are sane -------------------------------------------
check("revision cap is finite", 0 <= settings.max_revision_cycles <= 3)
check("research round cap is finite", 1 <= settings.max_research_rounds <= 3)
check("call budget is finite", 10 <= settings.max_agent_calls_per_run <= 200)
check("wall-clock budget is finite", 60 <= settings.max_run_seconds <= 1800)
check("rate-limit backoff is configured", settings.rate_limit_backoff_seconds >= 0,
      f"{settings.rate_limit_backoff_seconds}s")

# --- §23 persistence: the finalise write path -------------------------------
from app.graph.nodes import make_finalise, make_intake  # noqa: E402
from app.graph.state import initial_state  # noqa: E402
from app.services.usage import UsageTracker  # noqa: E402
from app.storage.evidence_store import EvidenceStore  # noqa: E402


class _BrokenStore:
    enabled = True

    def _boom(self, *a, **k):
        raise RuntimeError("connection refused")

    start_run = save_evidence = save_trace = save_errors = save_report = finish_run = _boom


broken = WorkflowDeps(index=index, store=_BrokenStore(), usage=UsageTracker(), run_id="verify")
check("persistence failure cannot fail a completed run",
      make_finalise(broken)(initial_state("x", run_id="verify"))["status"]
      is WorkflowStatus.COMPLETED)
check("persistence failure cannot fail intake",
      make_intake(broken)(initial_state("x", run_id="verify"))["status"]
      is WorkflowStatus.ANALYSING_REQUEST)

store = EvidenceStore()
check("evidence store is configured for durable runs", store.enabled,
      "" if store.enabled else "DATABASE_URL unset — runs will not persist")

# --- generated architecture doc ---------------------------------------------
subprocess.run([sys.executable, "scripts/gen_specs.py"], cwd=ROOT, capture_output=True)
arch = ROOT / "docs" / "A5_architecture.md"
check("A5 architecture doc generated", arch.is_file())
if arch.is_file():
    text = arch.read_text(encoding="utf-8")
    check("A5 is generated, not hand-written", "GENERATED by scripts/gen_specs.py" in text)
    check("A5 contains a mermaid diagram", "```mermaid" in text)
    check("A5 documents the termination caps", "always terminates" in text)
    check("A5 documents the §22 failure table", "Failure handling" in text)
    from app.graph.workflow import EDGES

    documented = {n for e in EDGES for n in e[:2]}
    check("documented edges match the compiled graph", documented == nodes | {"__start__", "__end__"},
          f"diff: {documented ^ (nodes | {'__start__', '__end__'})}")
    block = text.split("```mermaid", 1)[1].split("```", 1)[0] if "```mermaid" in text else ""
    reserved = [ln.strip().split("[")[0].split("(")[0].split("{")[0].split()[0]
                for ln in block.splitlines()
                if ln.strip() and not ln.strip().startswith("flowchart")]
    check("mermaid diagram avoids reserved keywords",
          not ({"end", "graph", "subgraph"} & set(reserved)))

# --- tests -------------------------------------------------------------------
res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                     cwd=ROOT, capture_output=True, text=True)
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("full test suite passes", res.returncode == 0, tail[0])

res = subprocess.run([sys.executable, "-m", "pytest", "tests/test_workflow.py", "-q",
                      "--no-header", "-k", "terminates"], cwd=ROOT, capture_output=True, text=True)
check("adversarial critic termination test passes", res.returncode == 0)

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 5: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
