"""
Model-selection probe: which Groq models can actually do the Supervisor's job?

The generic probe in ``probe_providers.py`` proves a model can emit *a* structured object.
That is not the bar. The Supervisor has to emit a **nested** plan with inter-task dependencies
and correct agent assignment, and it has to notice when a request is too vague to plan at all.
Small models pass the generic probe and fail this one, so model choice is decided here.

Scored deterministically (no LLM judge) on one under-specified and one well-specified request,
5 points each:
  - tasks    : produced a sensible number of tasks (3-10)
  - deps     : at least one task declares a dependency, and every dep id exists
  - agents   : only ever assigns known agent names
  - ambig    : flagged the vague request as needing clarification, and the clear one as not
  - questions: when flagging ambiguity, actually said what was missing (>=2 questions)

``tasks`` and ``deps`` are waived when a model correctly withholds a plan pending clarification
— see :func:`score`.

    python scripts/probe_planning_quality.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

load_dotenv(ROOT / ".env")

from app.config import PROVIDERS  # noqa: E402

AGENTS = {"researcher", "analyst", "critic", "fact_checker", "writer"}
OUT = ROOT / "scripts" / "probe_results.json"


class PlannedTask(BaseModel):
    task_id: str = Field(description="Short id, e.g. R1, A1, C1, W1")
    description: str
    assigned_agent: str = Field(description=f"One of: {', '.join(sorted(AGENTS))}")
    depends_on: list[str] = Field(default_factory=list, description="task_ids that must finish first")
    priority: int = Field(description="1 = highest")


class Plan(BaseModel):
    objective: str
    needs_clarification: bool = Field(description="True only if the request is too vague to plan")
    clarifying_questions: list[str] = Field(default_factory=list)
    sub_questions: list[str]
    evaluation_criteria: list[str]
    tasks: list[PlannedTask]


SYSTEM = (
    "You are the Supervisor of a multi-agent research system. Decompose the user's request into "
    "a dependency-ordered task plan. Assign each task to exactly one of: "
    f"{', '.join(sorted(AGENTS))}. Analysis tasks must depend on research tasks; review tasks must "
    "depend on analysis; the write task must depend on review. Set needs_clarification only when "
    "the request is genuinely too vague to plan."
)

VAGUE = "Find the best AI framework."
CLEAR = (
    "Compare LangGraph, CrewAI and the OpenAI Agents SDK for a 4-person Python team building a "
    "production multi-agent support system. Recommend one, with reasoning."
)


def score(plan: Plan, expect_ambiguous: bool) -> dict:
    """Score a plan, waiving structural checks that don't apply.

    A model that flags a request as needing clarification and therefore emits **no tasks** is
    behaving correctly, not failing — deferring the plan until the objective is known is exactly
    what §10 asks for. An earlier version of this scorer demanded 3-10 tasks unconditionally and
    so punished the most careful model. Structural checks are waived (scored as passes) when the
    model correctly withheld a plan pending clarification.
    """
    ids = {t.task_id for t in plan.tasks}
    withheld = plan.needs_clarification and not plan.tasks
    deps_ok = any(t.depends_on for t in plan.tasks) and all(
        d in ids for t in plan.tasks for d in t.depends_on
    )
    return {
        "tasks": len(plan.tasks),
        "withheld_plan": withheld,
        "n_tasks_ok": True if withheld else 3 <= len(plan.tasks) <= 10,
        "deps_ok": True if withheld else deps_ok,
        "agents_ok": all(t.assigned_agent in AGENTS for t in plan.tasks),
        "ambig_ok": plan.needs_clarification == expect_ambiguous,
        # Flagging ambiguity without saying what's missing is useless to the user (§10).
        "asked_questions": (not plan.needs_clarification) or len(plan.clarifying_questions) >= 2,
        "n_subq": len(plan.sub_questions),
        "n_criteria": len(plan.evaluation_criteria),
    }


def run(make, model: str, request: str, expect_ambiguous: bool) -> dict:
    t0 = time.perf_counter()
    try:
        plan = make(model).with_structured_output(Plan, method="function_calling").invoke(
            [("system", SYSTEM), ("user", request)]
        )
        row = {"ok": True, "seconds": round(time.perf_counter() - t0, 2), **score(plan, expect_ambiguous)}
        row["points"] = sum(
            bool(row[k])
            for k in ("n_tasks_ok", "deps_ok", "agents_ok", "ambig_ok", "asked_questions")
        )
        return row
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "seconds": round(time.perf_counter() - t0, 2),
                "error": str(e)[:140], "points": 0}


def main() -> int:
    cfg = PROVIDERS["groq"]
    key = os.getenv(cfg.api_key_env)
    if not key:
        print(f"No {cfg.api_key_env} set."); return 1

    from langchain_openai import ChatOpenAI

    def make(model: str):
        return ChatOpenAI(model=model, api_key=key, base_url=cfg.base_url,
                          temperature=0, max_tokens=2048, timeout=90, max_retries=0)

    print(f"{'model':<34} {'vague req':<28} {'clear req':<28} total")
    print("-" * 100)
    results = []
    for model in cfg.probes():
        v = run(make, model, VAGUE, expect_ambiguous=True)
        c = run(make, model, CLEAR, expect_ambiguous=False)
        total = v["points"] + c["points"]

        def fmt(r: dict) -> str:
            if not r["ok"]:
                return f"FAILED {r.get('error','')[:18]}"
            flags = "".join(
                ch if r[k] else "-"
                for ch, k in (("t", "n_tasks_ok"), ("d", "deps_ok"), ("a", "agents_ok"),
                              ("?", "ambig_ok"), ("q", "asked_questions"))
            )
            return f"{r['points']}/5 [{flags}] {r['tasks']:>2}task {r['seconds']:>5.2f}s"

        print(f"{model:<34} {fmt(v):<29} {fmt(c):<29} {total}/10")
        results.append({"model": model, "vague": v, "clear": c, "total": total})

    print("\nflags: t=task count  d=valid dependencies  a=valid agent names  "
          "?=ambiguity call  q=asked >=2 questions")
    print("       t and d are waived when a model correctly withholds a plan pending clarification")
    best = max(results, key=lambda r: (r["total"], -r["clear"]["seconds"]))
    print(f"\nHighest scoring: {best['model']}  ({best['total']}/10)")

    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev["groq_planning_quality"] = results
    OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"Saved -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
