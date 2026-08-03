"""
Phase 5 tests: the orchestration graph.

The whole graph runs here with a scripted LLM, which makes paths that are rare or expensive
against a live API — a Critic that never approves, a provider outage mid-run, an empty corpus —
cheap and deterministic to assert.

The most important test in this file is
:func:`test_workflow_terminates_when_critic_rejects_forever`. §18 requires termination even under
an adversarial reviewer, and that guarantee is the one thing a prompt can never provide.
"""
from __future__ import annotations

import pathlib

import pytest

from app.config import settings
from app.graph.nodes import WorkflowDeps
from app.graph.routing import make_route_after_critic, make_route_after_gate
from app.graph.workflow import build_workflow, describe_topology, run_workflow
from app.schemas.common import (
    AgentId,
    ClaimType,
    Confidence,
    ReviewCriterion,
    Severity,
    TaskStatus,
    WorkflowStatus,
)
from app.schemas.handoffs import (
    AnalysisHandoff,
    Conclusion,
    CriticVerdict,
    Problem,
)
from app.schemas.request import ClarificationQuestion, RequestBrief
from app.schemas.tasks import Task
from app.services.llm_service import LLMError
from app.agents.reviewers import ReportDraft, SupportJudgement, SupportJudgements
from app.agents.researcher import ResearchSummary
from app.agents.supervisor import PlanDraft
from tests.conftest import FakeAIMessage


# --------------------------------------------------------------------------- #
# Scripts
# --------------------------------------------------------------------------- #
def a_brief(n_questions: int = 2) -> RequestBrief:
    qs = ["What state management does LangGraph provide?",
          "How does CrewAI handle human approval?",
          "What are the licensing terms?"][:n_questions]
    return RequestBrief(
        objective="Compare agent frameworks for a small Python team",
        sub_questions=qs, evaluation_criteria=["state", "approval"],
        options_under_comparison=["LangGraph", "CrewAI"],
        deliverable="Comparison and recommendation",
    )


def a_plan(n_research: int = 2) -> PlanDraft:
    qs = ["What state management does LangGraph provide?",
          "How does CrewAI handle human approval?",
          "What are the licensing terms?"][:n_research]
    tasks = [
        Task(task_id=f"R{i+1}", description=f"Research {q}",
             assigned_agent=AgentId.RESEARCHER, research_question=q)
        for i, q in enumerate(qs)
    ]
    ids = [t.task_id for t in tasks]
    tasks += [
        Task(task_id="A1", description="Compare", assigned_agent=AgentId.ANALYST,
             depends_on=ids),
        Task(task_id="C1", description="Review", assigned_agent=AgentId.CRITIC,
             depends_on=["A1"]),
        Task(task_id="W1", description="Write", assigned_agent=AgentId.WRITER,
             depends_on=["C1"]),
    ]
    return PlanDraft(tasks=tasks, rationale="derived from the brief")


def a_store_call(claim: str, text: str, doc: str, call_id: str) -> dict:
    return {"name": "store_evidence", "id": call_id,
            "args": {"claim": claim, "supporting_text": text, "source_doc_id": doc,
                     "claim_type": "fact", "confidence": "high"}}


# Real verbatim spans from the corpus, so the anti-fabrication guard passes.
LANGGRAPH_SPAN = "Channels without a reducer raise InvalidUpdateError if written concurrently."
CREWAI_SPAN = "CrewAI supports a human_input flag on individual tasks."


def researcher_script(n_tasks: int) -> list:
    """Per researcher: one tool-calling turn, one empty turn, one summary."""
    script: list = []
    spans = [("LangGraph merges parallel writes via reducers", LANGGRAPH_SPAN,
              "fw-langgraph-docs", "What state management does LangGraph provide?"),
             ("CrewAI supports task-level human input", CREWAI_SPAN,
              "fw-crewai-docs", "How does CrewAI handle human approval?"),
             ("LangGraph merges parallel writes via reducers", LANGGRAPH_SPAN,
              "fw-langgraph-docs", "What are the licensing terms?")]
    for i in range(n_tasks):
        claim, text, doc, question = spans[i % len(spans)]
        script.append(FakeAIMessage(tool_calls=[a_store_call(claim, text, doc, f"c{i}")]))
        script.append(FakeAIMessage(tool_calls=[]))          # done gathering
        script.append(ResearchSummary(findings=f"Found material for task {i+1}.",
                                      confidence=Confidence.HIGH, gaps=[],
                                      sources_consulted=[doc]))
    return script


def an_analysis(revision: int = 0) -> AnalysisHandoff:
    return AnalysisHandoff(
        summary="Both frameworks meet the requirements with different trade-offs.",
        conclusions=[
            Conclusion(conclusion_id="C1",
                       statement="LangGraph provides explicit typed state management",
                       evidence_ids=["E101"], confidence=Confidence.HIGH, is_major=True),
        ],
        evidence_ids_used=["E101"], revision=revision)


def a_report() -> ReportDraft:
    return ReportDraft(
        title="Agent Framework Comparison", executive_summary="Summary.",
        research_objective="Compare frameworks", methodology="Corpus research and review.",
        key_findings=["LangGraph has typed state"], risks_and_limitations=[],
        recommendation_statement="Adopt LangGraph", recommendation_rationale="Explicit state.",
        recommendation_confidence=Confidence.MEDIUM)


def a_rejection(cycle: int = 0) -> CriticVerdict:
    return CriticVerdict(
        approved=False,
        problems=[Problem(location="C1", issue="Overgeneralised from a single source",
                          criterion=ReviewCriterion.UNSUPPORTED_CLAIMS,
                          severity=Severity.MAJOR)],
        required_revisions=["Narrow the claim or cite more evidence."], cycle=cycle)


def full_script(*, n_research=2, verdicts=None, judgements=1):
    """Assemble a complete run script."""
    verdicts = verdicts if verdicts is not None else [CriticVerdict(approved=True)]
    analyses = [an_analysis(i) for i in range(len(verdicts))]
    return {
        "supervisor": [a_brief(n_research), a_plan(n_research)],
        "researcher": researcher_script(n_research),
        "analyst": analyses,
        "fact_checker": [SupportJudgements(judgements=[
            SupportJudgement(conclusion_id="C1", evidence_supports=True)])] * len(verdicts),
        "critic": list(verdicts),
        "writer": [a_report()],
    }


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #
def test_graph_compiles(corpus_index):
    assert build_workflow(WorkflowDeps(index=corpus_index, store=None)) is not None


def test_topology_lists_every_edge():
    text = describe_topology()
    for node in ("intake", "supervisor_analyse", "supervisor_plan", "plan_approval",
                 "research_dispatch", "evidence_gate", "analyst", "fact_checker",
                 "critic", "revision", "writer", "finalise"):
        assert node in text


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_full_workflow_completes(fake_llm_factory, corpus_index):
    fake_llm_factory(full_script())
    r = run_workflow("Compare LangGraph and CrewAI", index=corpus_index, store=None)

    assert r.completed, r.state.get("abort_reason")
    assert r.state["report"] is not None
    assert len(r.state["evidence"]) == 2
    assert r.state["revision_count"] == 0
    assert "## Executive Summary" in r.report_markdown


def test_every_agent_is_invoked_in_order(fake_llm_factory, corpus_index):
    fake_llm_factory(full_script())
    r = run_workflow("Compare LangGraph and CrewAI", index=corpus_index, store=None)

    handoffs = [e for e in r.state["trace"] if e.event == "handoff"]
    order = [e.agent_id for e in handoffs]
    for agent in (AgentId.SUPERVISOR, AgentId.RESEARCHER, AgentId.ANALYST,
                  AgentId.FACT_CHECKER, AgentId.CRITIC, AgentId.WRITER):
        assert agent in order, f"{agent.value} never handed off"
    assert order.index(AgentId.RESEARCHER) < order.index(AgentId.ANALYST)
    assert order.index(AgentId.ANALYST) < order.index(AgentId.CRITIC)
    assert order.index(AgentId.CRITIC) < order.index(AgentId.WRITER)


def test_summary_reports_measured_numbers(fake_llm_factory, corpus_index):
    fake_llm_factory(full_script())
    s = run_workflow("Compare", index=corpus_index, store=None).summary()
    assert s["status"] == "completed"
    assert s["research_tasks"] == 2 and s["evidence_count"] == 2
    assert s["has_report"] and s["fabricated_citations"] == []
    assert s["trace_events"] > 10


# --------------------------------------------------------------------------- #
# Clarification (§10)
# --------------------------------------------------------------------------- #
def test_ambiguous_request_stops_for_clarification(fake_llm_factory, corpus_index):
    vague = RequestBrief(
        objective="Find the best framework", needs_clarification=True,
        clarifying_questions=[
            ClarificationQuestion(question="Which language?", why_it_matters="changes candidates"),
            ClarificationQuestion(question="Prototype or production?",
                                  why_it_matters="changes the criteria")],
    )
    created = fake_llm_factory({"supervisor": [vague]})
    r = run_workflow("Find the best AI framework", index=corpus_index, store=None)

    assert r.status is WorkflowStatus.AWAITING_CLARIFICATION
    assert r.state["awaiting"] == "clarification"
    assert r.state["report"] is None
    assert len(r.state["brief"].clarifying_questions) == 2
    # §10: no expensive work may run before the ambiguity is resolved. Exactly one call — the
    # request analysis — and no researcher, analyst, critic or writer was ever constructed.
    assert len(created["supervisor"].calls) == 1
    assert set(created) == {"supervisor"}, f"other agents ran: {sorted(created)}"


def test_supplying_answers_lets_the_run_proceed(fake_llm_factory, corpus_index):
    from app.schemas.request import Clarification

    fake_llm_factory(full_script())
    r = run_workflow("Find the best AI framework", index=corpus_index, store=None,
                     clarifications=[Clarification(question="Which language?", answer="Python")])
    assert r.completed


# --------------------------------------------------------------------------- #
# §18 quality-control loop — and its termination
# --------------------------------------------------------------------------- #
def test_rejection_triggers_one_revision(fake_llm_factory, corpus_index):
    fake_llm_factory(full_script(verdicts=[a_rejection(0), CriticVerdict(approved=True)]))
    r = run_workflow("Compare", index=corpus_index, store=None)

    assert r.completed
    assert r.state["revision_count"] == 1
    assert len(r.state["critic_verdicts"]) == 2
    assert r.state["analysis"].revision == 1


def test_workflow_terminates_when_critic_rejects_forever(fake_llm_factory, corpus_index):
    """§18's hard guarantee, under an adversarial reviewer.

    The Critic is scripted to reject every time. The run must still finish, produce a report,
    and stop at exactly the configured cap.
    """
    verdicts = [a_rejection(i) for i in range(10)]
    script = full_script(verdicts=verdicts)
    script["analyst"] = [an_analysis(i) for i in range(10)]
    script["writer"] = [a_report()]
    fake_llm_factory(script)

    r = run_workflow("Compare", index=corpus_index, store=None)

    assert r.completed, r.state.get("abort_reason")
    assert r.state["report"] is not None
    assert r.state["revision_count"] == settings.max_revision_cycles


def test_unresolved_objections_appear_in_the_report(fake_llm_factory, corpus_index):
    verdicts = [a_rejection(i) for i in range(10)]
    script = full_script(verdicts=verdicts)
    script["analyst"] = [an_analysis(i) for i in range(10)]
    fake_llm_factory(script)

    r = run_workflow("Compare", index=corpus_index, store=None)
    limitations = r.state["report"].risks_and_limitations
    assert any("Overgeneralised" in lim for lim in limitations), (
        "the report shipped without disclosing the reviewer's unresolved objection"
    )


def test_revision_limit_of_zero_is_respected(fake_llm_factory, corpus_index):
    """Experiment 5's zero-revision arm."""
    script = full_script(verdicts=[a_rejection(0)])
    fake_llm_factory(script)
    r = run_workflow("Compare", index=corpus_index, store=None, max_revisions=0)
    assert r.completed and r.state["revision_count"] == 0


def test_critic_can_be_disabled_and_says_so(fake_llm_factory, corpus_index):
    """Experiment 2's control arm must be self-identifying in the output."""
    script = full_script()
    script["critic"] = []
    fake_llm_factory(script)

    r = run_workflow("Compare", index=corpus_index, store=None, critic_enabled=False)
    assert r.completed
    verdict = r.state["critic_verdicts"][-1]
    assert verdict.approved
    assert any("disabled" in m.lower() for m in verdict.missing_evidence)


# --------------------------------------------------------------------------- #
# §22 failure handling
# --------------------------------------------------------------------------- #
def test_supervisor_failure_ends_the_run_cleanly(fake_llm_factory, corpus_index):
    fake_llm_factory({"supervisor": [LLMError("All providers failed.")]})
    r = run_workflow("Compare", index=corpus_index, store=None)

    assert r.status is WorkflowStatus.FAILED
    assert r.state["abort_reason"]
    assert r.state["errors"], "a failure with no error record is invisible to the operator"
    assert r.state["report"] is None


def test_invalid_plan_fails_without_raising(fake_llm_factory, corpus_index):
    cyclic = PlanDraft(tasks=[
        Task(task_id="R1", description="a", assigned_agent=AgentId.RESEARCHER,
             research_question="q?", depends_on=["A1"]),
        Task(task_id="A1", description="b", assigned_agent=AgentId.ANALYST, depends_on=["R1"])])
    fake_llm_factory({"supervisor": [a_brief(), cyclic]})
    r = run_workflow("Compare", index=corpus_index, store=None)

    assert r.status is WorkflowStatus.FAILED
    assert any(e.kind == "invalid_output" for e in r.state["errors"])


def test_one_researcher_failing_does_not_stop_the_others(fake_llm_factory, corpus_index):
    script = full_script(n_research=2)
    # First researcher's tool turn fails outright; the second runs normally.
    script["researcher"] = [LLMError("provider down")] + researcher_script(2)[3:]
    fake_llm_factory(script)

    r = run_workflow("Compare", index=corpus_index, store=None)
    statuses = r.state["task_status"]
    assert TaskStatus.FAILED in statuses.values()
    assert TaskStatus.COMPLETED in statuses.values()
    assert len(r.state["evidence"]) >= 1


def test_graph_failure_returns_a_result_rather_than_raising(corpus_index, monkeypatch):
    """The eval runner must record a catastrophic failure, not die on it."""
    import app.graph.workflow as wf

    class Boom:
        def invoke(self, *a, **k):
            raise RuntimeError("recursion limit reached")

    monkeypatch.setattr(wf, "build_workflow", lambda *a, **k: Boom())
    r = wf.run_workflow("Compare", index=corpus_index, store=None)
    assert r.status is WorkflowStatus.FAILED
    assert "recursion limit" in r.state["abort_reason"]


# --------------------------------------------------------------------------- #
# Routing determinism
# --------------------------------------------------------------------------- #
def test_routing_never_calls_a_model(corpus_index):
    """Routing must be a pure function of state — no fake_llm_factory needed here.

    The autouse guard in conftest raises on any model call, so this test passing at all is the
    assertion: none of these routers touched an LLM.
    """
    deps = WorkflowDeps(index=corpus_index, store=None)
    route_critic = make_route_after_critic(deps)
    route_gate = make_route_after_gate(deps)

    assert route_critic({"critic_verdicts": [CriticVerdict(approved=True)],
                         "status": WorkflowStatus.REVIEWING}) == "writer"
    assert route_critic({"critic_verdicts": [a_rejection()], "revision_count": 0,
                         "status": WorkflowStatus.REVISING}) == "revision"
    assert route_critic({"critic_verdicts": [a_rejection()],
                         "revision_count": settings.max_revision_cycles,
                         "status": WorkflowStatus.REVISING}) == "writer"
    assert route_gate({"status": WorkflowStatus.ANALYSING_EVIDENCE}) == "analyst"
    assert route_gate({"status": WorkflowStatus.FAILED}) == "__end__"


def test_gate_never_replans_past_the_round_cap(corpus_index):
    deps = WorkflowDeps(index=corpus_index, store=None)
    route_gate = make_route_after_gate(deps)
    state = {"status": WorkflowStatus.PLANNING,
             "research_round": settings.max_research_rounds}
    assert route_gate(state) == "analyst", "re-planned past the research round cap"


def test_experiment_flags_do_not_change_the_graph_shape(corpus_index):
    """Experiments must differ by configuration, not by being a different program."""
    shapes = set()
    for kwargs in ({}, {"critic_enabled": False}, {"max_revisions": 0},
                   {"parallel_research": True}, {"full_context": True}):
        deps = WorkflowDeps(index=corpus_index, store=None, **kwargs)
        g = build_workflow(deps)
        shapes.add(tuple(sorted(g.get_graph().nodes)))
    assert len(shapes) == 1, "an experiment flag altered the node set"

def test_documented_edges_match_the_graph():
    """The generated architecture doc must describe the graph that actually runs.

    ``EDGES`` feeds both the docs and the mermaid diagram. If an edge is added to
    ``build_workflow`` without a matching entry here, the published diagram silently becomes a
    lie — which is worse than having no diagram, because a reviewer will trust it.
    """
    from app.graph.workflow import EDGES
    from app.storage.corpus import build_index

    deps = WorkflowDeps(index=build_index(), store=None)
    graph_nodes = set(build_workflow(deps).get_graph().nodes)
    documented = {n for edge in EDGES for n in edge[:2]}

    assert documented == graph_nodes, (
        f"doc/graph mismatch — undocumented: {graph_nodes - documented}, "
        f"documented but absent: {documented - graph_nodes}"
    )


def test_mermaid_diagram_avoids_reserved_keywords():
    """`end` is reserved in mermaid and silently breaks the diagram.

    LangGraph's terminal node is named `__end__`, so naive underscore-stripping produces exactly
    that id. Caught only by rendering the generated doc.
    """
    doc = pathlib.Path("docs/A5_architecture.md")
    assert doc.is_file(), "run scripts/gen_specs.py"
    body = doc.read_text(encoding="utf-8")
    block = body.split("```mermaid", 1)[1].split("```", 1)[0]
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("flowchart"):
            continue
        first = stripped.split("[")[0].split("(")[0].split("{")[0].split()[0]
        assert first not in {"end", "graph", "subgraph", "class", "click", "style"}, (
            f"mermaid reserved keyword used as a node id: {first!r}"
        )


def test_store_none_really_disables_persistence(fake_llm_factory, corpus_index):
    """`store=None` must mean *no database*, not "use the default one".

    It previously fell through to a real EvidenceStore, so the entire test suite was writing
    rows into the live database on every run.
    """
    fake_llm_factory(full_script())
    r = run_workflow("Compare", index=corpus_index, store=None)
    assert r.deps.store is None


def test_omitting_store_uses_the_configured_default(fake_llm_factory, corpus_index):
    from app.storage.evidence_store import EvidenceStore

    fake_llm_factory(full_script())
    r = run_workflow("Compare", index=corpus_index)
    assert isinstance(r.deps.store, EvidenceStore)
