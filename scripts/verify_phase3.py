"""
Phase 3 acceptance check: tools and permission boundaries.

    python scripts/verify_phase3.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.schemas.common import AgentId  # noqa: E402
from app.storage.corpus import build_index  # noqa: E402
from app.tools import (  # noqa: E402
    ToolContext,
    ToolError,
    ToolPermissionError,
    all_specs,
    openai_tool_schemas,
    run_tool,
    tools_for,
)

specs = all_specs()
index = build_index()
ctx = ToolContext(run_id="verify", task_id="R1", research_question="q", index=index)

# --- §14 tool count ----------------------------------------------------------
check("§14 at least four research tools", len(specs) >= 4, f"{len(specs)} registered")
check("search / extract / store / retrieve all present",
      {"search_corpus", "extract_document", "store_evidence", "retrieve_evidence"}
      <= {s.name for s in specs})

# --- §13 permissions ---------------------------------------------------------
check("every tool documents a rationale",
      all(len(s.rationale) > 40 for s in specs))
check("every tool permits at least one agent", all(s.allowed_agents for s in specs))

# the headline boundaries
for agent, tool in [(AgentId.WRITER, "search_corpus"), (AgentId.WRITER, "store_evidence"),
                    (AgentId.ANALYST, "search_corpus"), (AgentId.CRITIC, "search_corpus"),
                    (AgentId.ANALYST, "validate_citations"),
                    (AgentId.RESEARCHER, "export_report"),
                    (AgentId.SUPERVISOR, "search_corpus")]:
    try:
        run_tool(tool, {}, ctx, agent_id=agent)
        ok = False
    except ToolPermissionError:
        ok = True
    except ToolError:
        ok = False  # wrong error type: it got past the permission gate
    check(f"§13 {agent.value} cannot call {tool}", ok)

# the whole grid
refused = allowed_pairs = 0
for spec in specs:
    for agent in AgentId.llm_agents():
        if spec.permits(agent):
            allowed_pairs += 1
            continue
        try:
            run_tool(spec.name, {}, ctx, agent_id=agent)
            check(f"forbidden pairing refused: {agent.value} -> {spec.name}", False)
        except ToolPermissionError:
            refused += 1
        except ToolError:
            check(f"forbidden pairing refused: {agent.value} -> {spec.name}", False,
                  "raised the wrong error type")
check("every forbidden agent/tool pairing raises", refused > 0,
      f"{refused} refused, {allowed_pairs} permitted")

check("agents are only shown tools they may call",
      all({s["function"]["name"] for s in openai_tool_schemas(a)}
          == {s.name for s in tools_for(a)} for a in AgentId.llm_agents()))
check("refused calls are audited",
      any(c.get("outcome") == "permission_denied" for c in ctx.calls),
      f"{sum(1 for c in ctx.calls if c.get('outcome') == 'permission_denied')} denials logged")

# --- anti-fabrication guard --------------------------------------------------
fresh = ToolContext(run_id="verify", task_id="R1", research_question="q", index=index)
try:
    run_tool("store_evidence", {
        "claim": "LangGraph guarantees exactly-once delivery",
        "supporting_text": "LangGraph guarantees exactly-once delivery across every node.",
        "source_doc_id": "fw-langgraph-docs", "claim_type": "fact", "confidence": "high",
    }, fresh, agent_id=AgentId.RESEARCHER)
    check("fabricated supporting text is rejected", False)
except ToolError as e:
    check("fabricated supporting text is rejected", "not found in" in str(e))

out = run_tool("store_evidence", {
    "claim": "CrewAI markets a 10x development speedup",
    "supporting_text": "ship their first production workflow up to 10x faster",
    "source_doc_id": "fw-vendor-comparison", "claim_type": "claim", "confidence": "high",
}, fresh, agent_id=AgentId.RESEARCHER)
check("low-reliability source caps confidence", out.confidence.value == "low" and out.adjusted)

# --- empty results and untrusted input --------------------------------------
empty = run_tool("search_corpus", {"query": "quantum blockchain toaster"}, fresh,
                 agent_id=AgentId.RESEARCHER)
check("§22 empty search returns zero hits with guidance",
      empty.result_count == 0 and "gap" in empty.note.lower())

for expr in ("__import__('os').system('x')", "open('f').read()", "[].__class__"):
    try:
        run_tool("calculate", {"expression": expr}, fresh, agent_id=AgentId.ANALYST)
        check("calculate refuses non-arithmetic input", False, expr)
        break
    except ToolError:
        pass
else:
    check("calculate refuses non-arithmetic input", True)

val = run_tool("validate_citations", {"evidence_ids": []}, fresh, agent_id=AgentId.CRITIC)
check("validating nothing does not score as valid", not val.all_valid)

# --- generated docs ----------------------------------------------------------
subprocess.run([sys.executable, "scripts/gen_specs.py"], cwd=ROOT, capture_output=True)
spec_doc = ROOT / "docs" / "A4_tool_specification.md"
check("A4 tool specification generated", spec_doc.is_file())
if spec_doc.is_file():
    text = spec_doc.read_text(encoding="utf-8")
    check("A4 documents every tool", all(f"`{s.name}`" in text for s in specs))
    check("A4 includes the permission matrix", "| Tool | supervisor |" in text)

# --- tests -------------------------------------------------------------------
res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
                     cwd=ROOT, capture_output=True, text=True)
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("full test suite passes", res.returncode == 0, tail[0])

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 3: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
