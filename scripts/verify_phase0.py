"""
Phase 0 acceptance check.

Week 3's lesson (§7.6) was that "done" reported from memory is not done. Each phase therefore
ends with a script that checks the phase's claims against the actual repo and exits non-zero if
any of them is false.

    python scripts/verify_phase0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


# --- structure ---------------------------------------------------------------
for d in ("app/agents", "app/graph", "app/tools", "app/schemas", "app/services",
          "app/storage", "app/observability", "app/baseline",
          "corpus/frameworks", "corpus/cloud", "corpus/coding_assistants",
          "tests", "eval", "experiments", "docs", "scripts", "screenshots", ".streamlit"):
    check(f"dir {d}", (ROOT / d).is_dir())

for f in (".gitignore", ".env.example", "requirements.txt",
          "app/config.py", "app/theme.py", "app/services/llm_service.py",
          "app/services/usage.py", ".streamlit/config.toml"):
    check(f"file {f}", (ROOT / f).is_file())

# --- secrets hygiene (§7.1) --------------------------------------------------
gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
check("gitignore blocks .env", "\n.env\n" in gitignore)
check("gitignore keeps .env.example", "!.env.example" in gitignore)
example = (ROOT / ".env.example").read_text(encoding="utf-8")
check("no key material in .env.example",
      "gsk_" not in example and "sk-or-v1" not in example and "sk-proj" not in example)

# --- torch is genuinely gone (build-size decision) ---------------------------
# Both files *document* why torch and the file-watcher hack were dropped, so these checks must
# read effective configuration only. Matching raw text would fail on the explanatory comments.
def _active_lines(path: Path, comment: str = "#") -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(comment, 1)[0].strip()
        if line:
            out.append(line.lower())
    return out


reqs = _active_lines(ROOT / "requirements.txt")
check("no sentence-transformers/torch in requirements",
      not any("sentence-transformers" in ln or "torch" in ln for ln in reqs))
toml = _active_lines(ROOT / ".streamlit" / "config.toml")
check("hot-reload enabled (no fileWatcherType=none)",
      not any("filewatchertype" in ln for ln in toml))

# --- config wiring -----------------------------------------------------------
try:
    from app.config import AGENT_MODELS, PROVIDERS, model_for_agent, settings
    from app.services.llm_service import configured_providers, get_llm
    from app.services.usage import BudgetExceeded, UsageTracker

    # Asserted by capability, not by count. The original check was `len(PROVIDERS) == 5` and it
    # broke the moment a second and third provider organisation were added for capacity — a
    # correct change failing a test that was measuring the wrong property.
    required = {"groq", "google", "openrouter", "openai"}
    check("every required provider family is registered", required <= set(PROVIDERS),
          f"{len(PROVIDERS)} registered: {', '.join(PROVIDERS)}")
    check("at least two provider entries exist for cross-provider fallback",
          len(PROVIDERS) >= 2)
    keys = configured_providers()
    check("at least one provider key present", any(keys.values()),
          str([k for k, v in keys.items() if v]))
    check("fallback chain has >1 provider", len(settings.provider_chain()) > 1,
          str(settings.provider_chain()))
    check("all 6 agents + baseline have a model tier", len(AGENT_MODELS) == 7)
    check("supervisor/critic on the reasoning tier",
          model_for_agent("supervisor") == model_for_agent("critic") == "openai/gpt-oss-120b")
    check("researcher on the throughput tier",
          model_for_agent("researcher") == "llama-3.3-70b-versatile")
    check("revision cap is finite and <= 2", 0 < settings.max_revision_cycles <= 2)

    # The budget guard must actually raise — a cap that cannot trip is not a cap (§7.3).
    u = UsageTracker(run_id="verify")
    for _ in range(settings.max_agent_calls_per_run):
        u.record(agent_id="x", provider="groq", model="m")
    tripped = False
    try:
        u.check_budget()
    except BudgetExceeded:
        tripped = True
    check("budget circuit breaker trips at the call cap", tripped)

    llm = get_llm("supervisor")
    check("llm service builds a chain", len(llm._chain()) >= 2, llm.describe())
except Exception as e:  # noqa: BLE001
    check("config/services import", False, f"{type(e).__name__}: {e}")

# --- probe evidence exists ---------------------------------------------------
probe = ROOT / "scripts" / "probe_results.json"
check("provider probe has been run", probe.is_file())
if probe.is_file():
    import json

    data = json.loads(probe.read_text(encoding="utf-8"))
    check("planning-quality probe recorded", "groq_planning_quality" in data)

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 0: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
