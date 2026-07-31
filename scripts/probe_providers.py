"""
Provider / model capability probe.

Every agent in this system depends on **structured output** (a validated Pydantic object) and
the Researcher additionally depends on **tool calling**. Model ids churn and free tiers differ,
so we measure those two capabilities directly instead of trusting a docs page.

For each configured provider, for each candidate model, this records:
  - plain completion works + latency
  - structured output works + latency   <- the capability the whole graph rests on
  - tool calling works + latency
  - token usage (so Phase 10 cost estimates start from measured numbers)

Never prints an API key. Providers with no key set are skipped, not failed.

    python scripts/probe_providers.py                # probe every provider that has a key
    python scripts/probe_providers.py --provider groq
"""
from __future__ import annotations

import argparse
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

OUT = ROOT / "scripts" / "probe_results.json"


# --------------------------------------------------------------- probe payloads
class ResearchBrief(BaseModel):
    """Deliberately shaped like the real Supervisor output — nested list + enum-ish field."""

    objective: str = Field(description="One-sentence research objective")
    sub_questions: list[str] = Field(description="2-4 specific sub-questions")
    deliverable: str = Field(description="What the user should receive")


SEARCH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search the local research corpus for documents matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "How many results"},
                },
                "required": ["query"],
            },
        },
    }
]


def _client(cfg, key: str):
    from langchain_openai import ChatOpenAI

    return lambda model, **kw: ChatOpenAI(
        model=model, api_key=key, base_url=cfg.base_url,
        temperature=0, max_tokens=1024, timeout=60, max_retries=0, **kw
    )


def _timed(fn) -> tuple[bool, float, str, dict]:
    """Run fn, return (ok, seconds, detail, usage)."""
    t0 = time.perf_counter()
    try:
        result = fn()
        dt = time.perf_counter() - t0
        usage = {}
        meta = getattr(result, "usage_metadata", None) if not isinstance(result, BaseModel) else None
        if meta:
            usage = {"in": meta.get("input_tokens", 0), "out": meta.get("output_tokens", 0)}
        return True, dt, "", usage
    except Exception as e:  # noqa: BLE001
        dt = time.perf_counter() - t0
        msg = str(e).replace(os.environ.get("GROQ_API_KEY", "\0"), "***")
        return False, dt, msg[:160], {}


def probe_model(make, model: str) -> dict:
    row: dict = {"model": model}

    ok, dt, err, usage = _timed(
        lambda: make(model).invoke([("system", "Answer in one short sentence."),
                                    ("user", "What is a multi-agent system?")])
    )
    row["completion"] = {"ok": ok, "seconds": round(dt, 2), "error": err, "usage": usage}

    ok, dt, err, _ = _timed(
        lambda: make(model).with_structured_output(ResearchBrief, method="function_calling").invoke(
            [("system", "You are a research supervisor. Decompose the request."),
             ("user", "Compare LangGraph, CrewAI and the OpenAI Agents SDK for a small team.")]
        )
    )
    row["structured"] = {"ok": ok, "seconds": round(dt, 2), "error": err}

    def tool_call():
        r = make(model).bind_tools(SEARCH_TOOL).invoke(
            [("system", "Use the search_corpus tool to find information. Always call the tool."),
             ("user", "Find documents about LangGraph state management.")]
        )
        if not getattr(r, "tool_calls", None):
            raise RuntimeError("model returned no tool_calls")
        return r

    ok, dt, err, _ = _timed(tool_call)
    row["tool_calling"] = {"ok": ok, "seconds": round(dt, 2), "error": err}

    row["usable"] = row["completion"]["ok"] and row["structured"]["ok"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", help="probe only this provider")
    args = ap.parse_args()

    results: dict = {}
    names = [args.provider] if args.provider else list(PROVIDERS)

    for name in names:
        cfg = PROVIDERS[name]
        key = os.getenv(cfg.api_key_env)
        if not key:
            print(f"\n[skip] {name:<11} no {cfg.api_key_env} set")
            results[name] = {"skipped": "no api key"}
            continue

        print(f"\n=== {name} ({cfg.base_url}) ===")
        make = _client(cfg, key)
        rows = []
        for model in cfg.probe_models:
            r = probe_model(make, model)
            rows.append(r)
            mark = "OK " if r["usable"] else "XX "
            print(
                f" {mark} {model:<34} "
                f"chat {r['completion']['seconds']:>5.2f}s {'y' if r['completion']['ok'] else 'n'} | "
                f"struct {r['structured']['seconds']:>5.2f}s {'y' if r['structured']['ok'] else 'n'} | "
                f"tools {r['tool_calling']['seconds']:>5.2f}s {'y' if r['tool_calling']['ok'] else 'n'}"
            )
            for stage in ("completion", "structured", "tool_calling"):
                if r[stage]["error"]:
                    print(f"      {stage}: {r[stage]['error']}")
        results[name] = {"base_url": cfg.base_url, "models": rows}

    # §7.4 lesson: merge into existing results, never clobber a good file with an empty run.
    if any("models" in v for v in results.values()):
        prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        prev.update(results)
        OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
        print(f"\nSaved -> {OUT.relative_to(ROOT)}")
    else:
        print("\nNothing probed; results file left untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
