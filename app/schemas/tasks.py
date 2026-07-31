"""
Dynamic task planning (§11).

The plan is generated per request, not templated, so it is the one structure most exposed to
model error. Three failures here are fatal to the graph and all three are caught by validators
rather than discovered at runtime:

- **a dependency on a task that does not exist** — the dependent task waits forever
- **a dependency cycle** — the graph deadlocks with no error
- **an unknown agent name** — the router has nowhere to send the task

§22 also lists "duplicate tasks" as a failure to handle; :meth:`TaskPlan.duplicate_groups`
detects them so the Supervisor can collapse them instead of paying for the same research twice.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.common import AgentId, TaskStatus

TASK_ID_RE = re.compile(r"^[A-Z]{1,2}\d+$")


class Task(BaseModel):
    """One unit of planned work, owned by exactly one agent."""

    task_id: str = Field(description="Short id: R1/R2 research, A1 analysis, C1 critique, W1 write")
    description: str = Field(min_length=1)
    assigned_agent: AgentId
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=2, ge=1, le=5, description="1 = highest")
    research_question: str | None = Field(
        default=None,
        description="Required for researcher tasks — the single question this task answers",
    )
    error: str = ""

    @field_validator("task_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        v = v.strip().upper()
        if not TASK_ID_RE.match(v):
            raise ValueError(f"task_id must look like R1, A1, C1, W1 (got {v!r})")
        return v

    @model_validator(mode="after")
    def _research_tasks_have_a_question(self) -> Task:
        """A researcher task without a question has nothing to search for.

        This is the contract that makes the fan-out safe: every ``Send()`` carries exactly one
        research question, so parallel branches cannot silently duplicate or drop work.
        """
        if self.assigned_agent is AgentId.RESEARCHER and not (self.research_question or "").strip():
            raise ValueError(f"researcher task {self.task_id} must set research_question")
        return self

    def is_terminal(self) -> bool:
        return self.status in TaskStatus.terminal()


class TaskPlan(BaseModel):
    """The Supervisor's execution plan. Validated as a DAG before anything runs."""

    tasks: list[Task] = Field(default_factory=list)
    rationale: str = Field(default="", description="Why the plan is shaped this way")
    created_by: AgentId = AgentId.SUPERVISOR
    revision: int = Field(default=0, description="Bumped when the user edits or the plan re-plans")

    # ------------------------------------------------------------- validation
    @model_validator(mode="after")
    def _valid_dag(self) -> TaskPlan:
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate task_id(s): {dupes}")

        known = set(ids)
        for t in self.tasks:
            unknown = [d for d in t.depends_on if d not in known]
            if unknown:
                raise ValueError(f"task {t.task_id} depends on unknown task(s): {unknown}")
            if t.task_id in t.depends_on:
                raise ValueError(f"task {t.task_id} depends on itself")

        cycle = self._find_cycle()
        if cycle:
            raise ValueError(f"dependency cycle: {' -> '.join(cycle)}")
        return self

    def _find_cycle(self) -> list[str] | None:
        """Depth-first cycle detection. Returns the cycle path, or None if the plan is a DAG."""
        deps = {t.task_id: list(t.depends_on) for t in self.tasks}
        WHITE, GREY, BLACK = 0, 1, 2
        colour = dict.fromkeys(deps, WHITE)
        stack: list[str] = []

        def visit(node: str) -> list[str] | None:
            colour[node] = GREY
            stack.append(node)
            for nxt in deps.get(node, []):
                if colour.get(nxt) == GREY:
                    return stack[stack.index(nxt):] + [nxt]
                if colour.get(nxt) == WHITE:
                    found = visit(nxt)
                    if found:
                        return found
            colour[node] = BLACK
            stack.pop()
            return None

        for node in deps:
            if colour[node] == WHITE:
                found = visit(node)
                if found:
                    return found
        return None

    # ---------------------------------------------------------------- queries
    def by_id(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.task_id == task_id), None)

    def for_agent(self, agent: AgentId) -> list[Task]:
        return [t for t in self.tasks if t.assigned_agent is agent]

    def research_tasks(self) -> list[Task]:
        return self.for_agent(AgentId.RESEARCHER)

    def ready(self) -> list[Task]:
        """Pending tasks whose dependencies have all completed.

        A task whose dependency FAILED is not ready and never will be — the caller marks it
        SKIPPED rather than leaving it pending forever (§22, failure propagation).
        """
        done = {t.task_id for t in self.tasks if t.status is TaskStatus.COMPLETED}
        return [
            t for t in self.tasks
            if t.status is TaskStatus.PENDING and all(d in done for d in t.depends_on)
        ]

    def blocked(self) -> list[Task]:
        """Pending tasks that can never run because a dependency failed or was skipped."""
        dead = {
            t.task_id for t in self.tasks
            if t.status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
        }
        return [
            t for t in self.tasks
            if t.status is TaskStatus.PENDING and any(d in dead for d in t.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(t.is_terminal() for t in self.tasks)

    def progress(self) -> tuple[int, int]:
        return sum(t.is_terminal() for t in self.tasks), len(self.tasks)

    def duplicate_groups(self, threshold: float = 0.85) -> list[list[str]]:
        """Research tasks whose questions are near-identical (§22, "duplicate tasks").

        Uses ratio similarity rather than an LLM call — detecting that two questions are the
        same string is not a judgement problem, and paying a model to do it would be exactly
        the kind of unnecessary agent call §43 warns against.
        """
        research = self.research_tasks()
        groups: list[list[str]] = []
        seen: set[str] = set()
        for i, a in enumerate(research):
            if a.task_id in seen:
                continue
            group = [a.task_id]
            for b in research[i + 1:]:
                if b.task_id in seen:
                    continue
                qa = (a.research_question or "").lower().strip()
                qb = (b.research_question or "").lower().strip()
                if qa and qb and SequenceMatcher(None, qa, qb).ratio() >= threshold:
                    group.append(b.task_id)
                    seen.add(b.task_id)
            if len(group) > 1:
                seen.add(a.task_id)
                groups.append(group)
        return groups

    def render(self) -> str:
        """Human-readable plan for the §20 approval checkpoint."""
        lines = []
        for t in sorted(self.tasks, key=lambda x: (x.priority, x.task_id)):
            dep = f" after {','.join(t.depends_on)}" if t.depends_on else ""
            lines.append(
                f"  {t.task_id:<4} [{t.assigned_agent.value:<12}] p{t.priority}{dep}\n"
                f"       {t.description}"
            )
        return "\n".join(lines)
