"""
Tool registry with per-agent permission boundaries (§13).

The Week 3 system's strongest measured result was 100% human-approval compliance, and the reason
was that approval lived in the graph rather than in a prompt. §13 is the same problem one level
down: the Writer must not be able to research, and "you are not allowed to search" in a system
prompt is a request, not a boundary.

So permission is a property of the call. :func:`run_tool` takes an ``agent_id`` and checks it
against the tool's ``allowed_agents`` **before** the tool function is reached. A Writer that
attempts ``corpus_search`` raises :class:`ToolPermissionError` — it cannot succeed, whatever the
model was persuaded to emit. That matters especially here, because this system reads a corpus
containing a live prompt-injection payload (PD6): an injected instruction cannot grant an agent
a tool it does not have.

Every tool declares:
- typed input and output models (invalid arguments become a clean error, §22)
- ``allowed_agents`` — the closed set of callers
- ``is_write`` — whether it mutates durable state
- ``rationale`` — *why* those agents and no others, surfaced in the generated docs
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.schemas.common import AgentId


# --------------------------------------------------------------------- errors
class ToolError(RuntimeError):
    """A tool failed for a user-safe, reportable reason."""


class ToolValidationError(ToolError):
    """The provided arguments did not match the tool's input schema."""


class ToolPermissionError(ToolError):
    """An agent attempted a tool outside its permission set (§13)."""


# -------------------------------------------------------------------- context
@dataclass
class ToolContext:
    """Dependencies handed to every tool.

    ``collected`` is the buffer a research node drains after its tool calls. Tools never write
    LangGraph state directly — a node returns state updates, and a tool that mutated state
    behind the node's back would bypass the reducers that make the parallel fan-out safe.
    """

    run_id: str = ""
    task_id: str = ""
    research_question: str = ""
    index: Any = None                       # app.storage.corpus.CorpusIndex
    store: Any = None                       # app.storage.evidence_store.EvidenceStore
    evidence: list[Any] = field(default_factory=list)   # evidence visible to this call
    collected: list[Any] = field(default_factory=list)  # evidence produced by this call
    llm: Any = None
    calls: list[dict] = field(default_factory=list)     # per-tool audit for the trace


# ----------------------------------------------------------------------- spec
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[[BaseModel, ToolContext], BaseModel]
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    allowed_agents: frozenset[AgentId]
    rationale: str
    is_write: bool = False

    def permits(self, agent_id: AgentId) -> bool:
        return agent_id in self.allowed_agents


TOOL_REGISTRY: dict[str, ToolSpec] = {}


def register_tool(
    *,
    name: str,
    description: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    allowed_agents: set[AgentId] | frozenset[AgentId],
    rationale: str,
    is_write: bool = False,
) -> Callable:
    """Decorator registering a tool under ``name``.

    ``allowed_agents`` and ``rationale`` are mandatory. §13 asks not only which agent gets which
    tool but *why*, and a required field is the only reliable way to keep that answer next to the
    code instead of drifting into a stale table.
    """

    def decorator(func: Callable) -> Callable:
        if name in TOOL_REGISTRY:
            raise ValueError(f"Duplicate tool name: {name}")
        if not allowed_agents:
            raise ValueError(f"Tool {name!r} must permit at least one agent")
        if not rationale.strip():
            raise ValueError(f"Tool {name!r} must document why its agents are permitted")
        TOOL_REGISTRY[name] = ToolSpec(
            name=name,
            description=description.strip(),
            func=func,
            input_model=input_model,
            output_model=output_model,
            allowed_agents=frozenset(allowed_agents),
            rationale=rationale.strip(),
            is_write=is_write,
        )
        return func

    return decorator


# ------------------------------------------------------------------ accessors
def get_spec(name: str) -> ToolSpec:
    if name not in TOOL_REGISTRY:
        raise ToolError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name]


def all_specs() -> list[ToolSpec]:
    return list(TOOL_REGISTRY.values())


def tools_for(agent_id: AgentId) -> list[ToolSpec]:
    """Every tool this agent may call. Drives what gets bound to the model."""
    return [s for s in TOOL_REGISTRY.values() if s.permits(agent_id)]


def openai_tool_schemas(agent_id: AgentId) -> list[dict]:
    """Tool schemas to bind for ``agent_id``.

    An agent is never *shown* a tool it may not call. That is defence in depth, not the
    boundary — :func:`run_tool` enforces the boundary regardless of what was bound, because a
    model can emit a tool call for a name it was never given.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.input_model.model_json_schema(),
            },
        }
        for s in tools_for(agent_id)
    ]


# ------------------------------------------------------------------ execution
def run_tool(name: str, raw_args: dict | None, ctx: ToolContext, agent_id: AgentId) -> BaseModel:
    """The single execution path for every tool call.

    Order matters: permission is checked before argument validation, and both before the tool
    function runs. A forbidden call must fail as forbidden even when its arguments are malformed,
    so that the audit trail records the permission violation rather than a schema error.
    """
    spec = get_spec(name)
    started = time.perf_counter()

    def audit(ok: bool, outcome: str) -> None:
        """Record every attempt, including refused ones.

        A refused call is the most interesting entry in the log, not the least: it is how a
        prompt-injection attempt or a misrouted agent becomes visible after the fact (§32).
        Auditing only successful calls would hide exactly the events worth reviewing.
        """
        ctx.calls.append({
            "tool": name, "agent": agent_id.value, "ok": ok, "outcome": outcome,
            "seconds": round(time.perf_counter() - started, 3),
        })

    if not spec.permits(agent_id):
        audit(False, "permission_denied")
        allowed = ", ".join(sorted(a.value for a in spec.allowed_agents))
        raise ToolPermissionError(
            f"Agent '{agent_id.value}' is not permitted to call '{name}'. "
            f"Permitted agents: {allowed}."
        )

    try:
        parsed = spec.input_model.model_validate(raw_args or {})
    except ValidationError as e:
        audit(False, "invalid_arguments")
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ToolValidationError(f"Invalid arguments for '{name}': {problems}") from e

    try:
        result = spec.func(parsed, ctx)
    except ToolError:
        audit(False, "tool_error")
        raise
    except Exception as e:  # noqa: BLE001
        audit(False, "unexpected_error")
        raise ToolError(f"Tool '{name}' failed: {type(e).__name__}") from e

    audit(True, "ok")
    return result


# ------------------------------------------------------------- documentation
def permission_matrix() -> str:
    """Render the §13 matrix from the registry, for the generated tool spec."""
    agents = [AgentId.SUPERVISOR, AgentId.RESEARCHER, AgentId.ANALYST,
              AgentId.FACT_CHECKER, AgentId.CRITIC, AgentId.WRITER]
    header = "| Tool | " + " | ".join(a.value for a in agents) + " | Writes |"
    sep = "|---" * (len(agents) + 2) + "|"
    rows = [header, sep]
    for spec in sorted(all_specs(), key=lambda s: s.name):
        marks = " | ".join("yes" if spec.permits(a) else "-" for a in agents)
        rows.append(f"| `{spec.name}` | {marks} | {'yes' if spec.is_write else '-'} |")
    return "\n".join(rows)
