"""
Phase 2 acceptance check: research corpus, retrieval, and evidence store.

    python scripts/verify_phase2.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


from app.storage.corpus import build_index, load_documents  # noqa: E402
from app.storage.evidence_store import TABLES, EvidenceStore, StorageError  # noqa: E402

# --- corpus ------------------------------------------------------------------
docs = load_documents()
check("§14 corpus has 24 documents", len(docs) == 24, f"{len(docs)} found")
check("three domains present",
      {d.domain for d in docs} == {"frameworks", "cloud", "coding_assistants"})
check("every document declares itself synthetic",
      all("synthetic: true" in d.path.read_text(encoding="utf-8") for d in docs))
check("every document has a parseable date and known reliability",
      all(d.published_date() and d.reliability in {"high", "medium", "low"} for d in docs))

types = {d.source_type for d in docs}
check("corpus mixes source types", len(types) >= 4, ", ".join(sorted(types)))

# --- planted defects ---------------------------------------------------------
manifest_path = ROOT / "corpus" / "planted_defects.json"
check("planted-defect manifest exists", manifest_path.is_file())
defects = []
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    defects = manifest["defects"]
    check("manifest marks the corpus synthetic", manifest.get("synthetic") is True)
    check("at least 8 planted defects", len(defects) >= 8, f"{len(defects)} declared")
    kinds = {d["type"] for d in defects}
    for required in ("contradiction", "unsupported_marketing_claim", "thin_document",
                     "prompt_injection"):
        check(f"defect type present: {required}", required in kinds)
    known = {d.doc_id for d in docs}
    bad = [d["id"] for d in defects if any(x not in known for x in d["documents"])]
    check("every defect references real documents", not bad, f"broken: {bad}" if bad else "")

# --- retrieval ---------------------------------------------------------------
index = build_index()
check("BM25 index built", len(index) > 50, f"{len(index)} chunks")
check("relevant query retrieves the right document",
      "fw-langgraph-docs" in {h.doc_id for h in index.search("LangGraph checkpointer resume", 5)})
check("§22 off-corpus query returns empty, not a bad match",
      index.search("quantum blockchain toaster recipes", 5) == [])
check("domain filter works",
      all(h.domain == "cloud" for h in index.search("cost pricing", 8, domain="cloud")))

pd1 = {h.doc_id for h in index.search("CrewAI human in the loop approval pause", 6)}
check("PD1 contradiction: both sides co-retrievable",
      {"fw-crewai-docs", "fw-practitioner-blog"} <= pd1)
pd2 = {h.doc_id for h in index.search("LangGraph CrewAI latency seconds benchmark", 8)}
check("PD2 contradiction: both sides co-retrievable",
      {"fw-benchmark-2026", "fw-vendor-comparison"} <= pd2)

thin = sum(c.word_count() for c in index.get_document("fw-llamaindex-workflows"))
check("PD5 thin document cannot support a conclusion", thin < 40, f"{thin} words")

injection = " ".join(c.text for c in index.get_document("ca-security-review")).lower()
check("PD6 prompt injection is present in retrievable text",
      "ignore all previous instructions" in injection)

check("marketing documents are marked low reliability",
      all(c.reliability == "low"
          for d in ("fw-vendor-comparison", "cl-vendor-whitepaper")
          for c in index.get_document(d)))

# --- evidence store ----------------------------------------------------------
store = EvidenceStore()
check("§15 evidence store configured", store.enabled,
      "" if store.enabled else "DATABASE_URL unset — persistence disabled")
if store.enabled:
    try:
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name = ANY(%s)",
                (list(TABLES),),
            ).fetchall()
        present = {r[0] for r in rows}
        missing = set(TABLES) - present
        check("all w4_ tables exist", not missing, f"missing: {missing}" if missing else "")
    except StorageError as e:
        check("all w4_ tables exist", False, str(e))

# a store with no URL must degrade, not explode
disabled = EvidenceStore(database_url=None)
disabled.database_url = None
check("workflow can run with no database at all",
      not disabled.enabled and disabled.save_evidence("r", []) == 0
      and disabled.load_evidence("r") == [])

# --- tests -------------------------------------------------------------------
res = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
    cwd=ROOT, capture_output=True, text=True,
)
tail = [ln for ln in res.stdout.strip().splitlines() if ln.strip()][-1:] or [""]
check("full test suite passes", res.returncode == 0, tail[0])

# --- report ------------------------------------------------------------------
failed = [c for c in checks if not c[1]]
width = max(len(c[0]) for c in checks) + 2
for name, ok, detail in checks:
    if not ok or detail:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}} {detail}")
print(f"\nPhase 2: {len(checks) - len(failed)}/{len(checks)} checks passed")
if failed:
    print("FAILED:", ", ".join(c[0] for c in failed))
raise SystemExit(1 if failed else 0)
