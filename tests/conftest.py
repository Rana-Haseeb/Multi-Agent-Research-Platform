"""
Shared fixtures. The point of this module is that **no test touches the network.**

``FakeLLM`` is scripted: you queue the objects (or exceptions) each structured call should
return, in order. That makes agent behaviour deterministic and lets failure paths — invalid
output, provider outage, budget exhaustion — be exercised on demand rather than waited for.

Carried over from Week 3, where the same pattern let 39 tests run with no API key and no
database. Here it matters more: six agents with a revision loop have far more paths than one
agent, and most of them are error paths.
"""
from __future__ import annotations

from typing import Any

import pytest

from app.schemas.common import AgentId, ClaimType, Confidence
from app.schemas.evidence import Evidence
from app.schemas.handoffs import AnalysisHandoff, Conclusion, ResearchHandoff
from app.schemas.request import RequestBrief
from app.services.llm_service import LLMError
from app.storage.corpus import build_index


# --------------------------------------------------------------------------- #
# Scripted LLM
# --------------------------------------------------------------------------- #
class FakeLLM:
    """A scripted stand-in for ``LLMService``.

    Queue results with ``script=[...]``. Each entry is either a value to return or an exception
    instance to raise. Exhausting the script raises, so a test that triggers more calls than it
    scripted fails loudly rather than silently reusing the last response.
    """

    def __init__(self, script: list[Any] | None = None, agent_id: str = "test",
                 usage: Any = None):
        self.script = list(script or [])
        self.agent_id = agent_id
        self.usage = usage
        self.calls: list[dict] = []
        self.last_used_model = "fake-model"
        self.last_used_provider = "fake"

    def _next(self, kind: str, system: str, user: str) -> Any:
        self.calls.append({"kind": kind, "agent": self.agent_id,
                           "system": system[:80], "user": user[:200]})
        if not self.script:
            raise AssertionError(
                f"FakeLLM script exhausted on call {len(self.calls)} "
                f"(agent={self.agent_id}, kind={kind})"
            )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def structured(self, system: str, user: str, schema: type) -> Any:
        return self._next("structured", system, user)

    def complete(self, system: str, user: str) -> str:
        return self._next("complete", system, user)

    def invoke_tools(self, messages: list, tools: list) -> Any:
        self.calls.append({"kind": "tools", "agent": self.agent_id,
                           "n_messages": len(messages), "n_tools": len(tools)})
        if not self.script:
            raise AssertionError("FakeLLM script exhausted on invoke_tools")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def describe(self) -> str:
        return "fake:fake-model"


class FakeAIMessage:
    """Minimal stand-in for a LangChain ``AIMessage`` carrying tool calls."""

    def __init__(self, tool_calls: list[dict] | None = None, content: str = ""):
        self.tool_calls = tool_calls or []
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


@pytest.fixture(autouse=True)
def _no_unmocked_llm_calls(monkeypatch, request):
    """Fail loudly if a test makes a real model call.

    Added after a test that simply forgot to script its LLM quietly hit the live API and took 23
    seconds — the suite still passed, so nothing pointed at it. An unmocked call in a unit test
    is slow, costs quota, and makes the result depend on a provider being up. Now it raises.

    Tests that genuinely need the network opt in with ``@pytest.mark.live``.
    """
    if request.node.get_closest_marker("live"):
        return

    def _forbidden(agent_id: str = "system", usage: Any = None):
        raise AssertionError(
            f"Unmocked LLM call from agent '{agent_id}'. Use the fake_llm_factory fixture, "
            f"or mark the test @pytest.mark.live if it must hit the network."
        )

    for module in ("app.agents.base", "app.services.llm_service"):
        monkeypatch.setattr(f"{module}.get_llm", _forbidden, raising=False)


@pytest.fixture
def fake_llm_factory(monkeypatch):
    """Patch ``get_llm`` everywhere agents import it, returning scripted responses per agent.

    Usage::

        fake_llm_factory({"analyst": [some_analysis], "critic": [some_verdict]})
    """
    created: dict[str, FakeLLM] = {}

    def install(scripts: dict[str, list[Any]]):
        def _get_llm(agent_id: str = "system", usage: Any = None) -> FakeLLM:
            llm = FakeLLM(script=list(scripts.get(agent_id, [])), agent_id=agent_id, usage=usage)
            created[agent_id] = llm
            return llm

        for module in ("app.agents.base", "app.services.llm_service"):
            monkeypatch.setattr(f"{module}.get_llm", _get_llm, raising=False)
        return created

    return install


# --------------------------------------------------------------------------- #
# Data fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def corpus_index():
    return build_index()


@pytest.fixture
def brief() -> RequestBrief:
    return RequestBrief(
        objective="Compare agent frameworks for a small Python team",
        sub_questions=["What state management does LangGraph provide?",
                       "How does CrewAI handle human approval?"],
        evaluation_criteria=["state management", "human-in-the-loop"],
        options_under_comparison=["LangGraph", "CrewAI"],
        deliverable="Comparison report with a recommendation",
    )


def make_evidence(eid="E101", question="What state management does LangGraph provide?",
                  confidence=Confidence.HIGH, claim_type=ClaimType.FACT,
                  claim="LangGraph provides typed state channels") -> Evidence:
    return Evidence(
        evidence_id=eid, claim=claim,
        supporting_text="State is declared as a TypedDict whose channels may carry reducers.",
        source_id="fw-langgraph-docs", source_title="LangGraph Documentation",
        research_question=question, confidence=confidence, claim_type=claim_type,
        agent_id="researcher", task_id="R1",
    )


@pytest.fixture
def evidence() -> list[Evidence]:
    return [
        make_evidence("E101"),
        make_evidence("E201", question="How does CrewAI handle human approval?",
                      claim="CrewAI supports a human_input flag on tasks"),
    ]


@pytest.fixture
def handoffs() -> list[ResearchHandoff]:
    return [
        ResearchHandoff(task_id="R1",
                        research_question="What state management does LangGraph provide?",
                        findings="LangGraph uses a typed StateGraph with reducers.",
                        evidence_ids=["E101"], confidence=Confidence.HIGH),
        ResearchHandoff(task_id="R2",
                        research_question="How does CrewAI handle human approval?",
                        findings="CrewAI exposes human_input on individual tasks.",
                        evidence_ids=["E201"], confidence=Confidence.MEDIUM),
    ]


@pytest.fixture
def analysis() -> AnalysisHandoff:
    return AnalysisHandoff(
        summary="Both frameworks support the required features, with different trade-offs.",
        conclusions=[
            Conclusion(conclusion_id="C1",
                       statement="LangGraph provides explicit typed state management",
                       evidence_ids=["E101"], confidence=Confidence.HIGH, is_major=True),
            Conclusion(conclusion_id="C2",
                       statement="CrewAI offers task-level human approval",
                       evidence_ids=["E201"], confidence=Confidence.MEDIUM, is_major=True),
        ],
        assumptions=["Team size does not change within the evaluation period"],
        evidence_ids_used=["E101", "E201"],
    )


@pytest.fixture
def llm_error() -> LLMError:
    return LLMError("All configured providers and models failed.")
