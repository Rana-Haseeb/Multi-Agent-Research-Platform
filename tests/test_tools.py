"""
Phase 3 tests: tool registry, permission boundaries, and tool behaviour.

The centrepiece is :func:`test_every_forbidden_pairing_is_refused`, which enumerates the entire
tool x agent grid and asserts every disallowed cell raises. §13 asks for tool boundaries; a
matrix in a README is a claim, and this is the proof.
"""
from __future__ import annotations

import pytest

from app.schemas.common import AgentId, ClaimType, Confidence
from app.schemas.evidence import Evidence
from app.storage.corpus import build_index
from app.tools import (
    ToolContext,
    ToolError,
    ToolPermissionError,
    ToolValidationError,
    all_specs,
    openai_tool_schemas,
    run_tool,
    tools_for,
)


@pytest.fixture(scope="module")
def index():
    return build_index()


@pytest.fixture
def ctx(index):
    return ToolContext(
        run_id="test-run", task_id="R1",
        research_question="How does LangGraph handle parallel writes?",
        index=index,
        evidence=[
            Evidence(
                evidence_id="E101", claim="LangGraph uses reducers",
                supporting_text="Channels may carry reducer functions.",
                source_id="fw-langgraph-docs", source_title="LangGraph Documentation",
                research_question="How does LangGraph handle parallel writes?",
                confidence=Confidence.HIGH, claim_type=ClaimType.FACT,
                agent_id="researcher", task_id="R1",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# §13 permission boundaries
# --------------------------------------------------------------------------- #
def test_at_least_four_tools_exist():
    """§14 requires a minimum of four research tools."""
    assert len(all_specs()) >= 4


def test_every_forbidden_pairing_is_refused(ctx):
    """The whole grid. Every cell the matrix says 'no' to must actually raise."""
    refused = 0
    for spec in all_specs():
        for agent in AgentId.llm_agents():
            if spec.permits(agent):
                continue
            with pytest.raises(ToolPermissionError):
                run_tool(spec.name, {}, ctx, agent_id=agent)
            refused += 1
    assert refused >= 20, f"only {refused} forbidden pairings exercised"


def test_writer_cannot_search(ctx):
    """The headline boundary: the Writer structurally cannot research."""
    with pytest.raises(ToolPermissionError, match="not permitted"):
        run_tool("search_corpus", {"query": "LangGraph"}, ctx, agent_id=AgentId.WRITER)


def test_writer_cannot_store_evidence(ctx):
    with pytest.raises(ToolPermissionError):
        run_tool("store_evidence", {}, ctx, agent_id=AgentId.WRITER)


def test_analyst_cannot_search(ctx):
    with pytest.raises(ToolPermissionError):
        run_tool("search_corpus", {"query": "cost"}, ctx, agent_id=AgentId.ANALYST)


def test_analyst_cannot_validate_its_own_citations(ctx):
    """An author verifying its own citations is not verification."""
    with pytest.raises(ToolPermissionError):
        run_tool("validate_citations", {"evidence_ids": ["E101"]}, ctx,
                 agent_id=AgentId.ANALYST)


def test_researcher_cannot_export(ctx):
    with pytest.raises(ToolPermissionError):
        run_tool("export_report", {"title": "t", "markdown": "m"}, ctx,
                 agent_id=AgentId.RESEARCHER)


def test_permission_is_checked_before_argument_validation(ctx):
    """A forbidden call must fail as forbidden even when its arguments are also invalid."""
    with pytest.raises(ToolPermissionError):
        run_tool("search_corpus", {"nonsense": True}, ctx, agent_id=AgentId.CRITIC)


def test_agents_are_only_shown_tools_they_may_call():
    for agent in AgentId.llm_agents():
        shown = {s["function"]["name"] for s in openai_tool_schemas(agent)}
        allowed = {s.name for s in tools_for(agent)}
        assert shown == allowed


def test_every_tool_documents_its_rationale():
    """§13 asks why each agent has each permission, not just which."""
    for spec in all_specs():
        assert len(spec.rationale) > 40, f"{spec.name} rationale is too thin"
        assert spec.allowed_agents, f"{spec.name} permits nobody"


def test_unknown_tool_raises():
    with pytest.raises(ToolError, match="Unknown tool"):
        run_tool("definitely_not_a_tool", {}, ToolContext(), agent_id=AgentId.RESEARCHER)


# --------------------------------------------------------------------------- #
# search_corpus
# --------------------------------------------------------------------------- #
def test_search_returns_hits_with_reliability(ctx):
    out = run_tool("search_corpus", {"query": "LangGraph checkpointer resume"}, ctx,
                   agent_id=AgentId.RESEARCHER)
    assert out.result_count > 0
    assert all(h.reliability in {"high", "medium", "low"} for h in out.hits)


def test_search_empty_result_tells_the_agent_to_record_a_gap(ctx):
    out = run_tool("search_corpus", {"query": "quantum blockchain toaster"}, ctx,
                   agent_id=AgentId.RESEARCHER)
    assert out.result_count == 0
    assert "gap" in out.note.lower()


def test_search_rejects_invalid_arguments(ctx):
    with pytest.raises(ToolValidationError, match="Invalid arguments"):
        run_tool("search_corpus", {"query": "x", "top_k": 500}, ctx,
                 agent_id=AgentId.RESEARCHER)


# --------------------------------------------------------------------------- #
# store_evidence — the anti-fabrication guard
# --------------------------------------------------------------------------- #
def test_store_evidence_rejects_fabricated_supporting_text(ctx):
    """The most damaging failure a research agent has: inventing the quote."""
    with pytest.raises(ToolError, match="not found in"):
        run_tool("store_evidence", {
            "claim": "LangGraph guarantees exactly-once delivery",
            "supporting_text": "LangGraph guarantees exactly-once delivery across all nodes.",
            "source_doc_id": "fw-langgraph-docs",
            "claim_type": "fact", "confidence": "high",
        }, ctx, agent_id=AgentId.RESEARCHER)


def test_store_evidence_rejects_unknown_source(ctx):
    with pytest.raises(ToolError, match="no document"):
        run_tool("store_evidence", {
            "claim": "A claim about a source that does not exist",
            "supporting_text": "some supporting text here",
            "source_doc_id": "fw-does-not-exist",
            "claim_type": "fact", "confidence": "high",
        }, ctx, agent_id=AgentId.RESEARCHER)


def test_store_evidence_accepts_a_verbatim_quote(ctx):
    out = run_tool("store_evidence", {
        "claim": "LangGraph merges concurrent writes via reducers",
        "supporting_text": "Channels without a reducer raise InvalidUpdateError if written concurrently.",
        "source_doc_id": "fw-langgraph-docs",
        "claim_type": "fact", "confidence": "high",
    }, ctx, agent_id=AgentId.RESEARCHER)
    assert out.stored and out.evidence_id == "E101"
    assert len(ctx.collected) == 1


def test_store_evidence_tolerates_whitespace_differences(ctx):
    out = run_tool("store_evidence", {
        "claim": "Reducers decide how writes combine",
        "supporting_text": "the   reducer\n  decides how the writes combine",
        "source_doc_id": "fw-langgraph-docs",
        "claim_type": "fact", "confidence": "medium",
    }, ctx, agent_id=AgentId.RESEARCHER)
    assert out.stored


def test_marketing_source_cannot_yield_high_confidence(ctx):
    """PD3: a low-reliability source is capped, however assertive its wording."""
    out = run_tool("store_evidence", {
        "claim": "CrewAI is 10x faster to develop with",
        "supporting_text": "ship their first production workflow up to 10x faster",
        "source_doc_id": "fw-vendor-comparison",
        "claim_type": "claim", "confidence": "high",
    }, ctx, agent_id=AgentId.RESEARCHER)
    assert out.confidence is Confidence.LOW
    assert out.adjusted and "reliability" in out.note


def test_assumptions_are_always_low_confidence(ctx):
    out = run_tool("store_evidence", {
        "claim": "LangGraph is probably better for large teams",
        "supporting_text": "Teams report a steeper initial learning curve",
        "source_doc_id": "fw-langgraph-docs",
        "claim_type": "assumption", "confidence": "high",
    }, ctx, agent_id=AgentId.RESEARCHER)
    assert out.confidence is Confidence.LOW and out.adjusted


def test_parallel_tasks_produce_non_colliding_ids(index):
    """Two researcher contexts on different tasks must not mint the same id."""
    ids = set()
    for task_id in ("R1", "R2", "R3"):
        c = ToolContext(task_id=task_id, index=index, research_question="q")
        for _ in range(2):
            out = run_tool("store_evidence", {
                "claim": "LangGraph uses a typed state object",
                "supporting_text": "a directed graph of nodes over a shared, typed state",
                "source_doc_id": "fw-langgraph-docs",
                "claim_type": "fact", "confidence": "high",
            }, c, agent_id=AgentId.RESEARCHER)
            ids.add(out.evidence_id)
    assert len(ids) == 6


# --------------------------------------------------------------------------- #
# retrieve_evidence
# --------------------------------------------------------------------------- #
def test_retrieve_index_only_omits_supporting_text(ctx):
    out = run_tool("retrieve_evidence", {"index_only": True}, ctx, agent_id=AgentId.CRITIC)
    assert out.count == 1
    assert "Channels may carry" not in out.items[0]


def test_retrieve_full_includes_supporting_text(ctx):
    out = run_tool("retrieve_evidence", {}, ctx, agent_id=AgentId.ANALYST)
    assert "Channels may carry" in out.items[0]


def test_retrieve_reports_missing_ids(ctx):
    """A fabricated citation must surface as missing, never as a silent empty result."""
    out = run_tool("retrieve_evidence", {"evidence_ids": ["E101", "E999"]}, ctx,
                   agent_id=AgentId.CRITIC)
    assert out.missing_ids == ["E999"] and out.count == 1


def test_retrieve_filters_by_research_question(ctx):
    out = run_tool("retrieve_evidence", {"research_question": "a question nobody asked"},
                   ctx, agent_id=AgentId.ANALYST)
    assert out.count == 0


# --------------------------------------------------------------------------- #
# calculate
# --------------------------------------------------------------------------- #
def test_calculate_evaluates_arithmetic(ctx):
    out = run_tool("calculate", {"expression": "(96.60 - 45.00) / 96.60 * 100"}, ctx,
                   agent_id=AgentId.ANALYST)
    assert round(out.result, 1) == 53.4


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd').read()",
    "[].__class__.__mro__",
    "1 if print('x') else 2",
])
def test_calculate_refuses_non_arithmetic(ctx, expr):
    """Expressions originate from a model that just read untrusted corpus text."""
    with pytest.raises(ToolError):
        run_tool("calculate", {"expression": expr}, ctx, agent_id=AgentId.ANALYST)


def test_calculate_handles_division_by_zero(ctx):
    with pytest.raises(ToolError, match="zero"):
        run_tool("calculate", {"expression": "1/0"}, ctx, agent_id=AgentId.ANALYST)


# --------------------------------------------------------------------------- #
# validate_citations — deterministic
# --------------------------------------------------------------------------- #
def test_validate_citations_detects_fabricated_id(ctx):
    out = run_tool("validate_citations", {"evidence_ids": ["E101", "E9"]}, ctx,
                   agent_id=AgentId.FACT_CHECKER)
    assert out.fabricated_ids == ["E9"] and not out.all_valid


def test_validate_citations_passes_real_ids(ctx):
    out = run_tool("validate_citations", {"evidence_ids": ["E101"]}, ctx,
                   agent_id=AgentId.FACT_CHECKER)
    assert out.all_valid and out.valid_ids == ["E101"]


def test_validate_citations_treats_no_citations_as_not_valid(ctx):
    """§7.3 again: verifying nothing must not score as verified."""
    out = run_tool("validate_citations", {"evidence_ids": []}, ctx,
                   agent_id=AgentId.CRITIC)
    assert not out.all_valid and out.checked == 0


# --------------------------------------------------------------------------- #
# export_report
# --------------------------------------------------------------------------- #
def test_export_without_store_still_succeeds(ctx):
    out = run_tool("export_report", {"title": "T", "markdown": "# Report"}, ctx,
                   agent_id=AgentId.WRITER)
    assert out.exported and not out.persisted


def test_tool_calls_are_audited(ctx):
    run_tool("retrieve_evidence", {}, ctx, agent_id=AgentId.WRITER)
    assert ctx.calls and ctx.calls[-1]["tool"] == "retrieve_evidence"
    assert ctx.calls[-1]["agent"] == "writer" and ctx.calls[-1]["ok"] is True


def test_failed_tool_calls_are_audited_too(ctx):
    with pytest.raises(ToolError):
        run_tool("store_evidence", {
            "claim": "An invented claim with no source",
            "supporting_text": "not in the document at all, invented",
            "source_doc_id": "fw-langgraph-docs",
            "claim_type": "fact", "confidence": "high",
        }, ctx, agent_id=AgentId.RESEARCHER)
    assert ctx.calls[-1]["ok"] is False
    assert ctx.calls[-1]["outcome"] == "tool_error"


def test_refused_calls_are_audited(ctx):
    """A permission denial is the most important thing in the log, not the least (§32)."""
    with pytest.raises(ToolPermissionError):
        run_tool("search_corpus", {"query": "anything"}, ctx, agent_id=AgentId.WRITER)
    assert ctx.calls[-1] == {
        "tool": "search_corpus", "agent": "writer", "ok": False,
        "outcome": "permission_denied", "seconds": ctx.calls[-1]["seconds"],
    }


def test_invalid_arguments_are_audited(ctx):
    with pytest.raises(ToolValidationError):
        run_tool("search_corpus", {"top_k": 999}, ctx, agent_id=AgentId.RESEARCHER)
    assert ctx.calls[-1]["outcome"] == "invalid_arguments"
