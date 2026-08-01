"""
Shared agent scaffolding: structured calls, the tool loop, and failure handling (§22).

Every agent is a function returning an :class:`AgentOutcome`. Agents never raise into the graph.
A node that raises takes the whole workflow down; a node that returns a failed outcome lets the
Supervisor decide whether to retry, degrade, or stop — which is what §22 asks for and what makes
"one agent returns invalid output" a recoverable event rather than a crash.

Two execution shapes cover all six agents:

- :func:`structured_step` — one call, one validated object out. Five of the six agents.
- :func:`tool_loop` — iterative tool calling, then a structured summary. The Researcher only,
  because it is the only agent that gathers rather than transforms.

Both meter usage, emit trace events, and check the run budget *before* spending anything.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from app.config import settings
from app.schemas.common import AgentId
from app.schemas.reports import ErrorRecord, TraceEvent
from app.services.llm_service import LLMError, get_llm
from app.services.usage import BudgetExceeded, UsageTracker
from app.tools import ToolContext, ToolError, ToolPermissionError, openai_tool_schemas, run_tool

T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentOutcome(Generic[T]):
    """What every agent returns. Never an exception."""

    agent_id: AgentId
    ok: bool
    output: T | None = None
    error: ErrorRecord | None = None
    trace: list[TraceEvent] = field(default_factory=list)
    duration_seconds: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return not self.ok


def _error(agent_id: AgentId, node: str, task_id: str, kind: str, message: str,
           action: str = "") -> ErrorRecord:
    return ErrorRecord(agent_id=agent_id, node=node, task_id=task_id, kind=kind,
                       message=message[:400], recovered=False, action_taken=action)


def classify_failure(exc: Exception) -> str:
    """Map an exception to a §22 failure category, for the trace and the metrics."""
    if isinstance(exc, BudgetExceeded):
        return "budget"
    if isinstance(exc, ToolPermissionError):
        return "tool_permission"
    if isinstance(exc, ToolError):
        return "tool_failure"
    if isinstance(exc, LLMError):
        msg = str(exc).lower()
        if "structured" in msg or "structure" in msg:
            return "invalid_output"
        if "rate limit" in msg or "provider" in msg:
            return "api_failure"
        if "timed out" in msg or "timeout" in msg:
            return "timeout"
        return "api_failure"
    return "unexpected"


# --------------------------------------------------------------------------- #
# One structured call
# --------------------------------------------------------------------------- #
def structured_step(
    *,
    agent_id: AgentId,
    node: str,
    system: str,
    user: str,
    schema: type[T],
    usage: UsageTracker | None = None,
    task_id: str = "",
    detail: str = "",
) -> AgentOutcome[T]:
    """Run one agent call producing a validated ``schema`` instance.

    The budget check happens first, so a run that has already exceeded its ceiling stops without
    spending another call. Retry and cross-provider fallback live inside ``LLMService`` — by the
    time an :class:`LLMError` reaches here, every provider in the chain has already failed, and
    retrying at this level would only multiply the cost of a genuine outage.
    """
    started = time.perf_counter()
    llm = get_llm(agent_id.value, usage=usage)

    try:
        if usage:
            usage.check_budget()
        output = llm.structured(system, user, schema)
    except Exception as e:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        kind = classify_failure(e)
        return AgentOutcome(
            agent_id=agent_id, ok=False, duration_seconds=elapsed,
            error=_error(agent_id, node, task_id, kind, str(e)),
            trace=[TraceEvent(agent_id=agent_id, event="error", node=node, task_id=task_id,
                              detail=f"{kind}: {str(e)[:120]}", duration_seconds=elapsed)],
        )

    elapsed = time.perf_counter() - started
    return AgentOutcome(
        agent_id=agent_id, ok=True, output=output, duration_seconds=elapsed,
        trace=[TraceEvent(
            agent_id=agent_id, event="node_end", node=node, task_id=task_id,
            detail=detail or f"produced {schema.__name__}", duration_seconds=elapsed,
            provider=llm.last_used_provider or "", model=llm.last_used_model or "",
        )],
    )


# --------------------------------------------------------------------------- #
# Iterative tool calling
# --------------------------------------------------------------------------- #
def tool_loop(
    *,
    agent_id: AgentId,
    node: str,
    system: str,
    user: str,
    ctx: ToolContext,
    usage: UsageTracker | None = None,
    task_id: str = "",
    max_iterations: int = 6,
) -> tuple[list[str], list[TraceEvent], ErrorRecord | None]:
    """Let an agent call its permitted tools until it stops or hits the iteration cap.

    Returns the transcript of tool results (for the follow-up summary call), trace events, and a
    terminal error if one occurred.

    A tool failure is **fed back to the model rather than raised**. A researcher whose search
    returned nothing, or who cited a document that does not exist, should try a different query —
    that is recovery, and it only happens if the agent is told what went wrong. Only budget
    exhaustion and total provider failure end the loop.

    ``max_iterations`` is a hard stop against a model that calls tools forever without concluding.
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    llm = get_llm(agent_id.value, usage=usage)
    schemas = openai_tool_schemas(agent_id)
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
    transcript: list[str] = []
    trace: list[TraceEvent] = []

    for iteration in range(max_iterations):
        try:
            if usage:
                usage.check_budget()
            reply = llm.invoke_tools(messages, schemas)
        except Exception as e:  # noqa: BLE001
            kind = classify_failure(e)
            trace.append(TraceEvent(agent_id=agent_id, event="error", node=node, task_id=task_id,
                                    detail=f"{kind}: {str(e)[:120]}"))
            return transcript, trace, _error(agent_id, node, task_id, kind, str(e))

        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            break  # the agent is done gathering

        messages.append(reply if isinstance(reply, AIMessage) else AIMessage(content=""))
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", "") or name
            started = time.perf_counter()
            try:
                result = run_tool(name, args, ctx, agent_id=agent_id)
                payload = result.model_dump_json()
                ok = True
            except (ToolError, ToolPermissionError) as e:
                # Reported back to the model as a result, not raised. This is the recovery path.
                payload = f"TOOL ERROR: {e}"
                ok = False
            elapsed = time.perf_counter() - started

            messages.append(ToolMessage(content=payload[:4000], tool_call_id=call_id))
            transcript.append(f"{name}({args}) -> {payload[:600]}")
            trace.append(TraceEvent(
                agent_id=agent_id, event="tool_call", node=node, task_id=task_id,
                detail=f"{name}{'' if ok else ' [failed]'}", duration_seconds=elapsed,
            ))
    else:
        trace.append(TraceEvent(
            agent_id=agent_id, event="node_end", node=node, task_id=task_id,
            detail=f"tool loop hit the {max_iterations}-iteration cap",
        ))

    return transcript, trace, None


def summarise_tool_results(transcript: list[str], limit: int = 8000) -> str:
    """Compact the tool transcript for the follow-up structured call."""
    if not transcript:
        return "No tool calls were made and no information was gathered."
    joined = "\n\n".join(transcript)
    if len(joined) <= limit:
        return joined
    return joined[:limit] + f"\n\n[transcript truncated — {len(transcript)} tool calls total]"


def budget_note(usage: UsageTracker | None) -> str:
    if not usage:
        return ""
    return (f"{usage.total_calls}/{settings.max_agent_calls_per_run} calls, "
            f"{usage.elapsed_seconds:.0f}s elapsed")
