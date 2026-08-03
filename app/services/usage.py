"""
Per-agent token and cost accounting.

Week 4 §29 asks for *average agent calls* and *approximate cost per run*, and Experiment 4
compares token usage under full-context versus role-specific context. Neither number can be
reconstructed after the fact, so every model call is metered at the moment it happens and
attributed to the agent that made it.

A :class:`UsageTracker` is created per workflow run and threaded through the graph. It also
serves as the run's **circuit breaker**: :meth:`check_budget` raises once a run exceeds its
configured call count, wall clock, or estimated spend, which is what stops a runaway
supervisor→critic→analyst loop from quietly costing real money (§22).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import PROVIDERS, settings


class BudgetExceeded(RuntimeError):
    """A run hit its call, time, or cost ceiling and must stop."""


@dataclass
class CallRecord:
    agent_id: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float
    cost_usd: float
    ok: bool
    error: str = ""


@dataclass
class UsageTracker:
    """Accumulates one run's model usage. Not thread-safe by design — see note below.

    Parallel researchers run in separate LangGraph branches but the same Python process and
    event loop, so appends are effectively serialised. If the fan-out ever moves to real
    threads, guard :meth:`record` with a lock.
    """

    run_id: str = ""
    started_at: float = field(default_factory=time.perf_counter)
    calls: list[CallRecord] = field(default_factory=list)

    # ---------------------------------------------------------------- recording
    def record(
        self,
        *,
        agent_id: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        seconds: float = 0.0,
        ok: bool = True,
        error: str = "",
    ) -> CallRecord:
        cfg = PROVIDERS.get(provider)
        cost = 0.0
        if cfg:
            cost = (input_tokens / 1_000_000) * cfg.price_in_per_m + (
                output_tokens / 1_000_000
            ) * cfg.price_out_per_m
        rec = CallRecord(
            agent_id=agent_id, provider=provider, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            seconds=seconds, cost_usd=cost, ok=ok, error=error,
        )
        self.calls.append(rec)
        return rec

    # ----------------------------------------------------------------- totals
    @property
    def total_calls(self) -> int:
        """Every recorded attempt, successful or not. Used for reporting."""
        return len(self.calls)

    @property
    def billable_calls(self) -> int:
        """Attempts that actually reached a model.

        A provider that refuses a request on a rate limit consumes no tokens and produces no
        output — it is a routing event, not a model call. Counting refusals against the run's
        call budget made the circuit breaker fire on a healthy run: an end-to-end test tripped
        the cap at 50 "calls" of which roughly half were instant 429s that cost nothing.

        ``total_calls`` still counts everything, because the *reporting* question ("how many
        attempts did this run make?") and the *budget* question ("how much work did it actually
        do?") are different questions.
        """
        return sum(1 for c in self.calls if c.ok or c.input_tokens or c.output_tokens)

    @property
    def refused_calls(self) -> int:
        return self.total_calls - self.billable_calls

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def by_agent(self) -> dict[str, dict]:
        """Per-agent rollup — this is the table the dashboard and §29 metrics read."""
        out: dict[str, dict] = {}
        for c in self.calls:
            row = out.setdefault(
                c.agent_id,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                 "seconds": 0.0, "cost_usd": 0.0, "failures": 0},
            )
            row["calls"] += 1
            row["input_tokens"] += c.input_tokens
            row["output_tokens"] += c.output_tokens
            row["seconds"] += c.seconds
            row["cost_usd"] += c.cost_usd
            row["failures"] += 0 if c.ok else 1
        return out

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "total_calls": self.total_calls,
            "billable_calls": self.billable_calls,
            "refused_calls": self.refused_calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cost_usd": round(self.total_cost_usd, 6),
            "wall_seconds": round(self.elapsed_seconds, 2),
            "by_agent": self.by_agent(),
        }

    # ------------------------------------------------------------ circuit break
    def check_budget(self) -> None:
        """Raise :class:`BudgetExceeded` if this run has gone past any ceiling."""
        if self.billable_calls >= settings.max_agent_calls_per_run:
            raise BudgetExceeded(
                f"Run exceeded {settings.max_agent_calls_per_run} model calls "
                f"({self.billable_calls} billable of {self.total_calls} attempted). "
                f"Stopping to prevent a runaway loop."
            )
        if self.elapsed_seconds >= settings.max_run_seconds:
            raise BudgetExceeded(
                f"Run exceeded {settings.max_run_seconds}s wall clock "
                f"({self.elapsed_seconds:.0f}s). Stopping."
            )
        if self.total_cost_usd >= settings.max_cost_usd_per_run:
            raise BudgetExceeded(
                f"Run exceeded ${settings.max_cost_usd_per_run:.2f} estimated cost "
                f"(${self.total_cost_usd:.4f}). Stopping."
            )
