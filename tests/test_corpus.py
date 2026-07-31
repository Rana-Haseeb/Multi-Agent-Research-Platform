"""
Phase 2 tests: corpus loading, chunking, BM25 retrieval, and the evidence store.

The retrieval tests assert against ``corpus/planted_defects.json`` rather than against prose, so
that changing a document without updating the manifest fails loudly. A defect the manifest
claims exists but retrieval can never surface is a defect the Critic will never be tested on.
"""
from __future__ import annotations

import json

import pytest

from app.config import ROOT
from app.storage.corpus import (
    CorpusIndex,
    build_index,
    chunk_document,
    load_documents,
    tokenize,
)
from app.storage.evidence_store import EvidenceStore, StorageNotConfigured
from app.schemas.common import ClaimType, Confidence
from app.schemas.evidence import Evidence


@pytest.fixture(scope="module")
def index() -> CorpusIndex:
    return build_index()


@pytest.fixture(scope="module")
def defects() -> list[dict]:
    data = json.loads((ROOT / "corpus" / "planted_defects.json").read_text(encoding="utf-8"))
    return data["defects"]


# --------------------------------------------------------------------------- #
# Loading and chunking
# --------------------------------------------------------------------------- #
def test_corpus_has_three_domains_and_24_documents():
    docs = load_documents()
    assert len(docs) == 24
    assert {d.domain for d in docs} == {"frameworks", "cloud", "coding_assistants"}


def test_every_document_has_required_metadata():
    for d in load_documents():
        assert d.doc_id and d.title and d.publisher, f"{d.path.name} missing metadata"
        assert d.source_type != "unknown", f"{d.path.name} has no source_type"
        assert d.reliability in {"high", "medium", "low"}, f"{d.path.name}: {d.reliability}"
        assert d.published_date() is not None, f"{d.path.name} has an unparseable date"


def test_frontmatter_is_stripped_from_body():
    doc = next(d for d in load_documents() if d.doc_id == "fw-langgraph-docs")
    assert "doc_id:" not in doc.body
    assert "synthetic:" not in doc.body


def test_chunks_carry_source_reliability(index: CorpusIndex):
    """A researcher that cannot tell marketing from a benchmark cannot assign confidence."""
    for chunk in index.chunks:
        assert chunk.reliability in {"high", "medium", "low"}
        assert chunk.source_type


def test_short_sections_are_folded_not_emitted():
    doc = next(d for d in load_documents() if d.doc_id == "fw-llamaindex-workflows")
    chunks = chunk_document(doc)
    assert all(c.word_count() >= 10 for c in chunks), "a stub heading became its own chunk"


def test_tokenizer_drops_stopwords_but_keeps_product_names():
    toks = tokenize("The state of Fly.io and the cost of GCP")
    assert "the" not in toks and "of" not in toks
    assert "fly.io" in toks and "gcp" in toks and "state" in toks


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def test_search_finds_the_obviously_relevant_document(index: CorpusIndex):
    hits = index.search("LangGraph checkpointer durability resume", top_k=5)
    assert "fw-langgraph-docs" in {h.doc_id for h in hits}


def test_search_returns_empty_for_off_corpus_query(index: CorpusIndex):
    """§22 'empty research results' — returning the least-bad chunk would invite false citation."""
    assert index.search("quantum blockchain toaster recipes", top_k=5) == []


def test_domain_filter_excludes_other_domains(index: CorpusIndex):
    hits = index.search("cost pricing comparison", top_k=10, domain="cloud")
    assert hits and all(h.domain == "cloud" for h in hits)


def test_scores_are_descending(index: CorpusIndex):
    hits = index.search("agent framework state management", top_k=10)
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_empty_query_returns_nothing(index: CorpusIndex):
    assert index.search("   ", top_k=5) == []
    assert index.search("the and of", top_k=5) == []


# --------------------------------------------------------------------------- #
# Planted defects must be reachable (§31 adversarial suite depends on this)
# --------------------------------------------------------------------------- #
def test_manifest_documents_all_exist(defects: list[dict], index: CorpusIndex):
    known = set(index.doc_ids())
    for d in defects:
        missing = [doc for doc in d["documents"] if doc not in known]
        assert not missing, f"{d['id']} references unknown documents: {missing}"


def test_contradiction_pair_is_retrievable_together(index: CorpusIndex):
    """PD1: both sides of the CrewAI human-in-the-loop disagreement must surface on one query.

    If only one side is ever retrieved, the Critic cannot detect the contradiction and the
    metric measuring that would be vacuous.
    """
    hits = index.search("CrewAI human in the loop approval pause", top_k=6)
    docs = {h.doc_id for h in hits}
    assert "fw-crewai-docs" in docs
    assert "fw-practitioner-blog" in docs


def test_second_contradiction_pair_is_retrievable_together(index: CorpusIndex):
    """PD2: benchmark latency figures versus the vendor's contradicting claim."""
    hits = index.search("LangGraph CrewAI latency seconds benchmark performance", top_k=8)
    docs = {h.doc_id for h in hits}
    assert "fw-benchmark-2026" in docs
    assert "fw-vendor-comparison" in docs


def test_thin_document_yields_little_content(index: CorpusIndex):
    """PD5: the stub must not be able to support a conclusion."""
    chunks = index.get_document("fw-llamaindex-workflows")
    assert sum(c.word_count() for c in chunks) < 40


def test_prompt_injection_is_present_and_retrievable(index: CorpusIndex):
    """PD6: the injection must actually be in retrievable text, or the defence is untested."""
    chunks = index.get_document("ca-security-review")
    body = " ".join(c.text for c in chunks).lower()
    assert "ignore all previous instructions" in body
    hits = index.search("prompt injection repository content agent security", top_k=5)
    assert "ca-security-review" in {h.doc_id for h in hits}


def test_marketing_claims_are_flagged_low_reliability(index: CorpusIndex):
    """PD3/PD4 live in documents the metadata already marks untrustworthy."""
    for doc_id in ("fw-vendor-comparison", "cl-vendor-whitepaper"):
        chunks = index.get_document(doc_id)
        assert chunks, f"{doc_id} not indexed"
        assert all(c.reliability == "low" for c in chunks)
        assert all(c.source_type == "vendor_marketing" for c in chunks)


def test_every_planted_defect_has_an_expected_catcher(defects: list[dict]):
    valid = {"researcher", "analyst", "critic", "fact_checker"}
    assert len(defects) >= 8
    for d in defects:
        assert d["expected_catcher"] in valid, f"{d['id']}: {d['expected_catcher']}"
        assert d["expected_signal"], f"{d['id']} has no expected signal"


# --------------------------------------------------------------------------- #
# Evidence store
# --------------------------------------------------------------------------- #
def test_store_without_url_is_disabled_not_broken():
    """Persistence is optional: the workflow must run with no database at all."""
    store = EvidenceStore(database_url=None)
    store.database_url = None
    assert not store.enabled
    assert store.save_evidence("run1", []) == 0
    assert store.load_evidence("run1") == []
    assert store.load_report("run1") is None
    assert store.recent_runs() == []


def test_disabled_store_raises_only_on_explicit_connection():
    store = EvidenceStore(database_url=None)
    store.database_url = None
    with pytest.raises(StorageNotConfigured):
        with store._conn():
            pass


@pytest.mark.skipif(not EvidenceStore().enabled, reason="no DATABASE_URL configured")
def test_evidence_round_trip_against_real_database():
    """End-to-end: write evidence, read it back, confirm every field survived."""
    store = EvidenceStore()
    run_id = "test-roundtrip"
    store.start_run(run_id, "test request", status="running")

    item = Evidence(
        evidence_id="E101", claim="LangGraph uses reducers",
        supporting_text="Channels may carry reducer functions.",
        source_id="fw-langgraph-docs", source_title="LangGraph Documentation",
        research_question="How does LangGraph merge parallel writes?",
        confidence=Confidence.HIGH, claim_type=ClaimType.FACT,
        agent_id="researcher", task_id="R1",
    )
    assert store.save_evidence(run_id, [item]) == 1
    assert store.save_evidence(run_id, [item]) == 1  # idempotent, no duplicate-key error

    loaded = store.load_evidence(run_id)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.evidence_id == "E101"
    assert got.claim == item.claim
    assert got.confidence is Confidence.HIGH
    assert got.claim_type is ClaimType.FACT
    assert got.research_question == item.research_question

    by_q = store.load_evidence(run_id, research_question=item.research_question)
    assert len(by_q) == 1
    assert store.load_evidence(run_id, research_question="a question nobody asked") == []

    store.finish_run(run_id, status="completed", agent_calls=3, wall_seconds=1.5)
    with store._conn() as conn:
        conn.execute("DELETE FROM w4_runs WHERE run_id = %s", (run_id,))
        conn.commit()
