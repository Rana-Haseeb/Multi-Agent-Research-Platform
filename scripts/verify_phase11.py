"""
Phase 11 acceptance check — documentation, adversarial testing, security review, deployment.

Week 3's lesson (§7.6) was that "done" reported from memory is not done. This script checks the
phase's claims against the actual repo — including running the adversarial suite rather than
trusting that it passed once — and exits non-zero if any claim is false.

    python scripts/verify_phase11.py
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def read(rel: str) -> str:
    p = ROOT / rel
    return p.read_text(encoding="utf-8") if p.is_file() else ""


# --- §36 deliverables exist ---------------------------------------------------
DELIVERABLES = {
    "docs/A1_agent_specifications.md": "agent specifications (§26)",
    "docs/A2_state_specification.md": "state specification (§16, §27)",
    "docs/A3_handoff_contracts.md": "handoff contracts (§17)",
    "docs/A4_tool_specification.md": "tool specification (§13)",
    "docs/A5_architecture.md": "architecture (§7, §8)",
    "eval/A6_evaluation.md": "evaluation results (§29)",
    "docs/A7_experiments.md": "experiment report (§30)",
    "docs/A8_adversarial_results.md": "adversarial results (§31)",
    "docs/A9_security_review.md": "security review (§32)",
    "docs/BUILDER_JOURNAL.md": "builder journal (§37)",
    "docs/ROADMAP.md": "future roadmap",
    "DEPLOYMENT.md": "deployment guide (§35)",
    "README.md": "README",
}
for rel, what in DELIVERABLES.items():
    check(f"{what}", (ROOT / rel).is_file(), rel)

# --- §26 agent specifications cover every agent -------------------------------
try:
    from app.schemas.common import AgentId

    specs = read("docs/A1_agent_specifications.md")
    # SYSTEM is the pseudo-identity used by deterministic nodes and holds no tool grants.
    # BASELINE is Experiment 1's single-agent control arm, deliberately outside the workflow.
    # Neither is one of the six specialised agents §26 asks to specify.
    NOT_WORKFLOW_AGENTS = {"system", "baseline"}
    agents = [a for a in AgentId if a.value not in NOT_WORKFLOW_AGENTS]
    missing = [a.value for a in agents if f"## {a.value}" not in specs]
    check("agent spec covers all six agents", not missing and len(agents) == 6,
          f"{len(agents)} agents" + (f", missing {missing}" if missing else ""))
    # A specification that omits limits is a description, not a specification.
    for heading in ("Prohibited actions", "Failure behaviour", "Handoff", "Tools"):
        check(f"agent spec states '{heading}'", specs.count(heading) >= len(agents),
              f"{specs.count(heading)}/{len(agents)}")
except Exception as e:  # noqa: BLE001
    check("agent spec introspection", False, f"{type(e).__name__}: {e}")

# --- §31 adversarial tests -----------------------------------------------------
adv_src = read("tests/test_adversarial.py")
check("adversarial test module exists", bool(adv_src))
if adv_src:
    tree = ast.parse(adv_src)
    tests = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    check("at least 10 adversarial tests (§31)", len(tests) >= 10, f"{len(tests)} tests")

    # Every test must say what it defends, or the document generated from it is empty prose.
    undocumented = [n.name for n in tests
                    if "DEFENCE" not in (ast.get_docstring(n) or "")]
    check("every adversarial test documents its defence", not undocumented, str(undocumented))

    # A defence enforced only by prompt wording can be argued with. The suite must be
    # predominantly structural or the security claims are aspirational.
    structural = [n for n in tests
                  if re.search(r"DEFENCE\s*\(structural\)", ast.get_docstring(n) or "")]
    check("majority of defences are structural", len(structural) * 2 > len(tests),
          f"{len(structural)}/{len(tests)} structural")

    # §31 names the scenarios that must be covered.
    scenarios = {
        "conflicting sources": "a1", "missing information": "a2",
        "unsupported certainty": "a3", "invalid agent output": "a4",
        "critic rejects repeatedly": "a5", "tool failure": "a6",
        "prompt injection": "a7", "duplicate task": "a8",
        "runaway cost": "a9", "user changes objective": "a10",
        "agent impersonation": "a11", "fabricated citation": "a12",
    }
    names = " ".join(n.name for n in tests)
    for label, prefix in scenarios.items():
        check(f"§31 scenario covered: {label}", f"test_{prefix}_" in names or
              f"test_{prefix}b_" in names)

    # Run them. A suite that passed once is not evidence that it passes now.
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_adversarial.py", "-q",
                        "--no-header"], cwd=ROOT, capture_output=True, text=True)
    check("adversarial suite passes", r.returncode == 0,
          (r.stdout.strip().splitlines() or [""])[-1])

# --- §32 security review -------------------------------------------------------
sec = read("docs/A9_security_review.md")
risk_headings = re.findall(r"^## \d+\.\s+(.+)$", sec, re.M)
check("at least 8 security risks (§32)", len(risk_headings) >= 8, f"{len(risk_headings)} risks")
check("every risk states a mitigation", sec.count("**Mitigation.**") >= len(risk_headings),
      f"{sec.count('**Mitigation.**')}/{len(risk_headings)}")
check("every risk names a verifying test", sec.count("**Verified by.**") >= len(risk_headings),
      f"{sec.count('**Verified by.**')}/{len(risk_headings)}")
check("residual risks are stated, not hidden", "residual" in sec.lower())

# Named test references in the security review must actually exist. A trailing ".py" is a
# module reference, not a function, so the two are resolved differently.
modules = set(re.findall(r"\b(test_[a-z0-9_]+)\.py", sec))
referenced = set(re.findall(r"\btest_[a-z0-9_]+\b(?!\.py)", sec))
defined: set[str] = set()
for tf in (ROOT / "tests").glob("test_*.py"):
    defined |= set(re.findall(r"\bdef (test_[a-z0-9_]+)", tf.read_text(encoding="utf-8")))
dangling = sorted(referenced - defined)
check("security review cites only tests that exist", not dangling, str(dangling[:4]))
missing_mod = sorted(m for m in modules if not (ROOT / "tests" / f"{m}.py").is_file())
check("security review cites only test modules that exist", not missing_mod, str(missing_mod))

# --- §37 builder journal -------------------------------------------------------
journal = read("docs/BUILDER_JOURNAL.md")
words = len(re.findall(r"[A-Za-z0-9''-]+", journal))
# §37 caps the journal at two pages; ~650 words/page is the proxy used here.
check("builder journal within 2 pages", 0 < words <= 1300, f"{words} words")
check("journal records what went wrong, not just what shipped",
      any(k in journal.lower() for k in ("broke", "wrong", "mistake", "would do differently")))
check("journal reports unflattering measurements",
      "n=1" in journal or "noise" in journal.lower())

# --- §35 deployment ------------------------------------------------------------
dep = read("DEPLOYMENT.md")
check("deployment covers the IPv6 pooler gotcha", "pooler" in dep.lower())
check("deployment covers secret handling", "secret" in dep.lower() and ".env" in dep)
check("deployment states a verification step", "verify_phase11" in dep or "pytest" in dep)

# --- secrets hygiene across ALL documentation (§7.1 — Week 3 leaked a key) ------
SECRET_PATTERNS = [
    (r"gsk_[A-Za-z0-9]{20,}", "Groq key"),
    (r"sk-or-v1-[A-Za-z0-9]{20,}", "OpenRouter key"),
    (r"sk-proj-[A-Za-z0-9]{20,}", "OpenAI key"),
    (r"AIza[A-Za-z0-9_-]{30,}", "Google key"),
    (r"xai-[A-Za-z0-9]{20,}", "xAI key"),
    (r"postgresql://[^\s`<]*:[^\s`<@]*@", "Postgres URL with inline password"),
]
tracked_md = sorted(p for p in ROOT.rglob("*.md")
                    if ".venv" not in p.parts and "node_modules" not in p.parts)
leaks: list[str] = []
for p in tracked_md:
    text = p.read_text(encoding="utf-8", errors="replace")
    for pattern, what in SECRET_PATTERNS:
        for m in re.finditer(pattern, text):
            # Placeholders in examples are the documented, safe form.
            if any(tok in m.group(0) for tok in ("<", "…", "postgres.<", "your", "xxx")):
                continue
            leaks.append(f"{p.relative_to(ROOT)}: {what}")
check("no credential material in any markdown", not leaks, str(sorted(set(leaks))[:4]))

# --- README reflects the finished project --------------------------------------
readme = read("README.md")
for rel in ("DEPLOYMENT.md", "BUILDER_JOURNAL.md", "ROADMAP.md",
            "A8_adversarial_results.md", "A9_security_review.md"):
    check(f"README links {rel}", rel in readme)
check("README does not name the provider (§ requested anonymity)",
      not re.search(r"\bgroq\b", readme, re.I))

# A README that advertises a stale count is a metric nobody measured. Collect every test count
# it claims and compare against the suite it points at.
collected = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only"],
                           cwd=ROOT, capture_output=True, text=True)
m = re.search(r"(\d+) tests? collected", collected.stdout)
if m:
    actual = int(m.group(1))
    claimed = {int(c) for c in re.findall(r"tests-(\d+)_passing", readme)}
    claimed |= {int(c) for c in re.findall(r"·\s*(\d+) tests passing", readme)}
    check("README test counts match the suite", claimed == {actual} if claimed else False,
          f"README says {sorted(claimed) or 'nothing'}, suite collects {actual}")

adv_claimed = re.search(r"adversarial-(\d+)_tests_·_(\d+)_structural", readme)
if adv_claimed and adv_src:
    check("README adversarial badge matches the suite",
          int(adv_claimed.group(1)) == len(tests)
          and int(adv_claimed.group(2)) == len(structural),
          f"badge {adv_claimed.group(1)}/{adv_claimed.group(2)}, "
          f"actual {len(tests)}/{len(structural)}")

# --- earlier phases still pass --------------------------------------------------
check("screenshots directory exists", (ROOT / "screenshots").is_dir())
# Screenshots could not be captured in this environment. That is allowed; claiming otherwise is
# not. The guide must state the true status, so the gap stays visible instead of being forgotten.
shots = sorted((ROOT / "screenshots").glob("*.png"))
guide = read("screenshots/README.md")
check("screenshot capture guide exists", bool(guide))
if not shots:
    check("screenshot status is stated honestly as not captured",
          "not captured" in guide.lower(), f"{len(shots)} png files present")
else:
    check("captured screenshots match the guide's filenames",
          all(re.search(rf"`{re.escape(p.name)}`", guide) for p in shots),
          f"{len(shots)} captured")
check("all earlier phase verifiers present",
      all((ROOT / "scripts" / f"verify_phase{i}.py").is_file() for i in range(10)))

# --- README's headline total ------------------------------------------------------
# The README advertises a total across every phase verifier, this one included — so the figure
# moves whenever a check is added here. It is therefore measured, never trusted, and this check
# runs last so that len(checks) + 1 (itself) is the exact Phase 11 count.
# Run them concurrently: serially this check alone took ~8 minutes, because several of the
# earlier verifiers run a pytest pass of their own.
def _phase_total(ph: int) -> int:
    out = subprocess.run([sys.executable, f"scripts/verify_phase{ph}.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(rf"Phase {ph}: \d+/(\d+) checks passed", out)
    return int(m.group(1)) if m else 0


with ThreadPoolExecutor(max_workers=10) as pool:
    subtotal = sum(pool.map(_phase_total, range(10)))
expected_total = subtotal + len(checks) + 1
claimed_total = re.search(r"\*\*Current: (\d+)/(\d+) acceptance checks", readme)
check("README acceptance-check total is accurate",
      bool(claimed_total) and int(claimed_total.group(1)) == expected_total,
      f"README says {claimed_total.group(1) if claimed_total else '?'}, "
      f"measured {expected_total} ({subtotal} in phases 0-9 + {len(checks) + 1} here)")

# --- report ---------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 11: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
