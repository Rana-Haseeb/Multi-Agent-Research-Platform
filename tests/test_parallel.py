"""
Phase 7 tests: the parallel research fan-out (§19).

The claim "research runs in parallel" is easy to make and easy to get wrong — a `Send()` list
that LangGraph happens to execute one after another looks identical in the trace to one that
runs concurrently. So :func:`test_fan_out_is_actually_concurrent` measures wall clock against a
deliberately slow researcher: three tasks that each sleep 0.4s must finish in well under the 1.2s
a sequential run would take.

The rest guard the properties parallelism can silently break: lost writes when reducers are
missing, a researcher seeing a sibling's work, and a budget cap overshooting because two
branches passed the same check.
"""
from __future__ import annotations

import threading
import time

import pytest

from app.config import settings
from app.graph import routing as R
from app.graph.nodes import WorkflowDeps, make_research_task
from app.graph.workflow import build_workflow, run_workflow
from app.schemas.common import AgentId, TaskStatus, WorkflowStatus
from app.schemas.tasks import Task, TaskPlan
from tests.test_workflow import a_brief, a_plan, full_script, researcher_script


def _plan(n: int = 3) -> TaskPlan:
    draft = a_plan(n)
    return TaskPlan(tasks=draft.tasks, rationale=draft.rationale)


def _approved_state(n: int = 3) -> dict:
    return {
        "status": WorkflowStatus.RESEARCHING,
        "plan_approved": True,
        "plan": _plan(n),
        "brief": a_brief(n),
        "task_status": {},
    }


# --------------------------------------------------------------------------- #
# Routing: one edge, two arms
# --------------------------------------------------------------------------- #
def test_sequential_flag_routes_to_the_dispatch_node(corpus_index):
    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=False)
    assert R.make_route_after_plan_approval(deps)(_approved_state()) == R.RESEARCH


def test_parallel_flag_emits_one_send_per_research_task(corpus_index):
    from langgraph.types import Send

    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)
    sends = R.make_route_after_plan_approval(deps)(_approved_state(3))

    assert isinstance(sends, list) and len(sends) == 3
    assert all(isinstance(s, Send) and s.node == R.RESEARCH_TASK for s in sends)
    assert {s.arg["task_id"] for s in sends} == {"R1", "R2", "R3"}


def test_each_send_carries_only_its_own_question(corpus_index):
    """§21 isolation, made structural: a branch physically cannot read a sibling's question."""
    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)
    sends = R.make_route_after_plan_approval(deps)(_approved_state(3))

    for send in sends:
        payload = send.arg
        assert set(payload) == {"task_id", "research_question", "objective"}
        others = {s.arg["research_question"] for s in sends} - {payload["research_question"]}
        assert payload["research_question"] not in others
        assert "evidence" not in payload and "plan" not in payload


def test_fan_out_respects_the_parallel_researcher_cap(corpus_index):
    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)
    wide = _approved_state(3)
    tasks = list(wide["plan"].tasks)
    for i in range(4, 4 + settings.max_parallel_researchers + 2):
        tasks.append(Task(task_id=f"R{i}", description="extra",
                          assigned_agent=AgentId.RESEARCHER,
                          research_question=f"Extra question {i}?"))
    wide["plan"] = TaskPlan(tasks=tasks)
    sends = R.make_route_after_plan_approval(deps)(wide)
    assert len(sends) == settings.max_parallel_researchers


def test_completed_tasks_are_not_re_dispatched(corpus_index):
    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)
    state = _approved_state(3)
    state["task_status"] = {"R1": TaskStatus.COMPLETED}
    sends = R.make_route_after_plan_approval(deps)(state)
    assert {s.arg["task_id"] for s in sends} == {"R2", "R3"}


def test_both_arms_produce_the_same_graph_shape(corpus_index):
    shapes = {
        tuple(sorted(build_workflow(
            WorkflowDeps(index=corpus_index, store=None, parallel_research=flag)
        ).get_graph().nodes))
        for flag in (False, True)
    }
    assert len(shapes) == 1, "the experiment arms are different programs, not one variable"


# --------------------------------------------------------------------------- #
# Concurrency, measured
# --------------------------------------------------------------------------- #
def test_fan_out_is_actually_concurrent(monkeypatch, corpus_index):
    """Wall clock is the only honest proof that `Send` runs branches together.

    Three researchers each sleep 0.4s. Sequential execution needs >=1.2s; genuine concurrency
    finishes near 0.4s. The threshold is deliberately loose (0.9s) so the test measures
    concurrency rather than machine speed.
    """
    import app.graph.nodes as nodes

    started: list[float] = []
    lock = threading.Lock()

    def slow_research_one(deps, task_id, question, objective):
        with lock:
            started.append(time.perf_counter())
        time.sleep(0.4)
        return {"evidence": [], "research_handoffs": [], "errors": [], "trace": [],
                "task_status": {task_id: TaskStatus.COMPLETED}}

    monkeypatch.setattr(nodes, "research_one", slow_research_one)

    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)
    task_node = make_research_task(deps)

    from langgraph.graph import END, START, StateGraph
    from app.graph.state import WorkflowState

    g = StateGraph(WorkflowState)
    g.add_node("gate", lambda s: {})
    g.add_node(R.RESEARCH_TASK, task_node)
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", R.make_route_after_plan_approval(deps), [R.RESEARCH_TASK, END])
    g.add_edge(R.RESEARCH_TASK, END)

    began = time.perf_counter()
    out = g.compile().invoke(_approved_state(3))
    elapsed = time.perf_counter() - began

    assert len(started) == 3, f"expected 3 branches, {len(started)} ran"
    assert out["task_status"] == {t: TaskStatus.COMPLETED for t in ("R1", "R2", "R3")}
    assert elapsed < 0.9, (
        f"fan-out took {elapsed:.2f}s for 3x0.4s tasks — branches ran sequentially"
    )
    assert max(started) - min(started) < 0.3, "branches did not start together"


def test_concurrent_branches_do_not_lose_evidence(monkeypatch, corpus_index):
    """The reducer property, exercised through the real fan-out rather than a fixture graph."""
    import app.graph.nodes as nodes
    from app.graph.state import evidence_id_for
    from tests.conftest import make_evidence

    def research_one_stub(deps, task_id, question, objective):
        return {
            "evidence": [make_evidence(evidence_id_for(task_id, i), question=question)
                         for i in (1, 2)],
            "research_handoffs": [], "errors": [], "trace": [],
            "task_status": {task_id: TaskStatus.COMPLETED},
        }

    monkeypatch.setattr(nodes, "research_one", research_one_stub)

    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)

    from langgraph.graph import END, START, StateGraph
    from app.graph.state import WorkflowState

    g = StateGraph(WorkflowState)
    g.add_node("gate", lambda s: {})
    g.add_node(R.RESEARCH_TASK, make_research_task(deps))
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", R.make_route_after_plan_approval(deps), [R.RESEARCH_TASK, END])
    g.add_edge(R.RESEARCH_TASK, END)

    out = g.compile().invoke({**_approved_state(3), "evidence": []})
    ids = {e.evidence_id for e in out["evidence"]}
    assert len(out["evidence"]) == 6, f"lost writes: only {len(out['evidence'])} of 6 survived"
    assert ids == {"E101", "E102", "E201", "E202", "E301", "E302"}


def test_usage_tracker_survives_concurrent_writers():
    """The lock added for the fan-out, exercised directly."""
    from app.services.usage import UsageTracker

    usage = UsageTracker(run_id="race")

    def worker():
        for _ in range(150):
            usage.record(agent_id="researcher", provider="groq", model="m",
                         input_tokens=1, output_tokens=1)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert usage.total_calls == 900
    assert usage.total_input_tokens == 900


# --------------------------------------------------------------------------- #
# End to end, both arms
# --------------------------------------------------------------------------- #
def test_parallel_workflow_completes(fake_llm_factory, corpus_index):
    fake_llm_factory(full_script(n_research=2))
    r = run_workflow("Compare LangGraph and CrewAI", index=corpus_index, store=None,
                     parallel_research=True)
    assert r.completed, r.state.get("abort_reason")
    assert len(r.state["evidence"]) == 2


def test_both_arms_reach_the_same_outcome(fake_llm_factory, corpus_index):
    """Parallelism must change timing, not results."""
    outcomes = {}
    for flag in (False, True):
        fake_llm_factory(full_script(n_research=2))
        r = run_workflow("Compare LangGraph and CrewAI", index=corpus_index, store=None,
                         parallel_research=flag)
        outcomes[flag] = (r.status, len(r.state["evidence"]),
                          {e.evidence_id for e in r.state["evidence"]},
                          r.state["report"] is not None)
    assert outcomes[False] == outcomes[True], f"arms diverged: {outcomes}"


def test_one_branch_failing_does_not_stop_the_others(monkeypatch, corpus_index):
    import app.graph.nodes as nodes
    from app.schemas.reports import ErrorRecord
    from app.graph.state import evidence_id_for
    from tests.conftest import make_evidence

    def flaky(deps, task_id, question, objective):
        if task_id == "R2":
            return {"evidence": [], "research_handoffs": [], "trace": [],
                    "errors": [ErrorRecord(agent_id=AgentId.RESEARCHER, node="research",
                                           task_id=task_id, kind="api_failure",
                                           message="provider down")],
                    "task_status": {task_id: TaskStatus.FAILED}}
        return {"evidence": [make_evidence(evidence_id_for(task_id, 1), question=question)],
                "research_handoffs": [], "errors": [], "trace": [],
                "task_status": {task_id: TaskStatus.COMPLETED}}

    monkeypatch.setattr(nodes, "research_one", flaky)
    deps = WorkflowDeps(index=corpus_index, store=None, parallel_research=True)

    from langgraph.graph import END, START, StateGraph
    from app.graph.state import WorkflowState

    g = StateGraph(WorkflowState)
    g.add_node("gate", lambda s: {})
    g.add_node(R.RESEARCH_TASK, make_research_task(deps))
    g.add_edge(START, "gate")
    g.add_conditional_edges("gate", R.make_route_after_plan_approval(deps), [R.RESEARCH_TASK, END])
    g.add_edge(R.RESEARCH_TASK, END)

    out = g.compile().invoke({**_approved_state(3), "evidence": [], "errors": []})
    assert out["task_status"]["R2"] is TaskStatus.FAILED
    assert out["task_status"]["R1"] is TaskStatus.COMPLETED
    assert len(out["evidence"]) == 2, "a failing branch took its siblings' evidence with it"
    assert len(out["errors"]) == 1
