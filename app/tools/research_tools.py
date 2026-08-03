"""
Research tools (§14): search, extract, store evidence, retrieve evidence.

These four are the Researcher's whole world, and the permission split is the point. Only the
Researcher may search or store; everyone downstream may only *retrieve* what was stored. That is
what makes "the Writer cannot invent research" a structural fact rather than a prompt.

Two behaviours here are deliberate and worth defending:

1. **Empty search results are returned as empty.** No fallback to the least-bad chunk. §22 lists
   "empty research results" as a condition to handle, and the failure mode being avoided is a
   confident citation of an irrelevant source.

2. **``store_evidence`` refuses to store a claim whose supporting text is not in the cited
   source.** The model supplies the claim; the tool verifies the quote actually appears in the
   chunk it names. This is a deterministic guard against fabricated support — the single most
   damaging thing a research agent can do.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.common import AgentId, ClaimType, Confidence
from app.schemas.evidence import Evidence
from app.tools.registry import ToolContext, ToolError, register_tool

# Reliability of the source constrains how confident a finding may be. Vendor marketing cannot
# yield a high-confidence fact no matter how assertively it is written.
MAX_CONFIDENCE_BY_RELIABILITY = {
    "high": Confidence.HIGH,
    "medium": Confidence.MEDIUM,
    "low": Confidence.LOW,
    "unknown": Confidence.LOW,
}
_CONF_ORDER = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


# --------------------------------------------------------------------------- #
# 1. search_corpus
# --------------------------------------------------------------------------- #
# Characters of passage text returned per hit. Search results re-enter the researcher's message
# history on every subsequent call, so this cap bounds context growth across the tool loop.
SEARCH_SNIPPET_CHARS = 700


class SearchInput(BaseModel):
    query: str = Field(min_length=2, description="Search terms. Use distinctive product names.")
    top_k: int = Field(default=4, ge=1, le=8)
    domain: str | None = Field(
        default=None,
        description="Optional filter: frameworks, cloud, or coding_assistants",
    )


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    source_type: str
    publisher: str
    published: str
    reliability: str
    score: float


class SearchOutput(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    result_count: int = 0
    note: str = ""


@register_tool(
    name="search_corpus",
    description=(
        "Search the research corpus for passages relevant to a query. Returns ranked passages "
        "with their source, publisher and reliability rating. Returns zero results when nothing "
        "relevant exists — that is a valid answer, not a failure."
    ),
    input_model=SearchInput,
    output_model=SearchOutput,
    allowed_agents={AgentId.RESEARCHER},
    rationale=(
        "Only the Researcher gathers new information. Granting search to the Analyst or Writer "
        "would let them introduce facts that never passed through the evidence store, defeating "
        "traceability; granting it to the Critic would let the reviewer research its own "
        "objections instead of judging the work in front of it."
    ),
)
def search_corpus(args: SearchInput, ctx: ToolContext) -> SearchOutput:
    if ctx.index is None:
        raise ToolError("Corpus index is unavailable.")
    hits = ctx.index.search(args.query, top_k=args.top_k, domain=args.domain)
    return SearchOutput(
        query=args.query,
        hits=[
            SearchHit(
                chunk_id=h.chunk_id, doc_id=h.doc_id, doc_title=h.doc_title, heading=h.heading,
                text=(h.text if len(h.text) <= SEARCH_SNIPPET_CHARS
                      else h.text[:SEARCH_SNIPPET_CHARS].rstrip() + " …[truncated: use "
                           "extract_document for the full section]"),
                source_type=h.source_type, publisher=h.publisher,
                published=h.published, reliability=h.reliability, score=h.score,
            )
            for h in hits
        ],
        result_count=len(hits),
        note="" if hits else (
            "No passage matched. Record this as an evidence gap rather than inferring an answer."
        ),
    )


# --------------------------------------------------------------------------- #
# 2. extract_document
# --------------------------------------------------------------------------- #
class ExtractInput(BaseModel):
    doc_id: str = Field(min_length=1, description="Document id from a search hit")
    section: str | None = Field(default=None, description="Optional heading to narrow to")


class ExtractOutput(BaseModel):
    doc_id: str
    title: str
    source_type: str
    publisher: str
    published: str
    reliability: str
    sections: dict[str, str] = Field(default_factory=dict)
    word_count: int = 0


@register_tool(
    name="extract_document",
    description=(
        "Read the full text of a corpus document, or one named section of it. Use after "
        "search_corpus when a passage looks relevant but needs surrounding context."
    ),
    input_model=ExtractInput,
    output_model=ExtractOutput,
    allowed_agents={AgentId.RESEARCHER},
    rationale=(
        "Paired with search_corpus and restricted for the same reason: reading source documents "
        "is how new information enters the system, and that entry point belongs to exactly one "
        "role so every fact has a single, auditable provenance path."
    ),
)
def extract_document(args: ExtractInput, ctx: ToolContext) -> ExtractOutput:
    if ctx.index is None:
        raise ToolError("Corpus index is unavailable.")
    chunks = ctx.index.get_document(args.doc_id)
    if not chunks:
        raise ToolError(f"No document with id '{args.doc_id}'.")
    if args.section:
        wanted = _normalise(args.section)
        chunks = [c for c in chunks if wanted in _normalise(c.heading)] or chunks
    first = chunks[0]
    sections = {c.heading: c.text for c in chunks}
    return ExtractOutput(
        doc_id=first.doc_id, title=first.doc_title, source_type=first.source_type,
        publisher=first.publisher, published=first.published, reliability=first.reliability,
        sections=sections,
        word_count=sum(c.word_count() for c in chunks),
    )


# --------------------------------------------------------------------------- #
# 3. store_evidence
# --------------------------------------------------------------------------- #
class StoreEvidenceInput(BaseModel):
    claim: str = Field(min_length=3, description="One-line assertion supported by the source")
    supporting_text: str = Field(
        min_length=10,
        description="Verbatim passage from the source document that backs the claim",
    )
    source_doc_id: str = Field(min_length=1, description="doc_id the passage came from")
    claim_type: ClaimType = Field(
        description="fact: stated in the source. claim: asserted by an interested source. "
                    "assumption: your inference, not stated anywhere."
    )
    confidence: Confidence
    research_question: str = Field(default="", description="Defaults to this task's question")


class StoreEvidenceOutput(BaseModel):
    evidence_id: str
    stored: bool
    confidence: Confidence
    adjusted: bool = False
    note: str = ""


@register_tool(
    name="store_evidence",
    description=(
        "Record a finding as structured evidence. The supporting_text must be copied verbatim "
        "from the cited document — it is verified against the source and rejected if absent."
    ),
    input_model=StoreEvidenceInput,
    output_model=StoreEvidenceOutput,
    allowed_agents={AgentId.RESEARCHER},
    is_write=True,
    rationale=(
        "The only write path into the evidence store, held by the only role that reads sources. "
        "If any other agent could store evidence, a downstream claim could be laundered into "
        "the record as though a source had supported it."
    ),
)
def store_evidence(args: StoreEvidenceInput, ctx: ToolContext) -> StoreEvidenceOutput:
    if ctx.index is None:
        raise ToolError("Corpus index is unavailable.")
    chunks = ctx.index.get_document(args.source_doc_id)
    if not chunks:
        raise ToolError(
            f"Cannot store evidence: no document '{args.source_doc_id}' exists. "
            f"Cite a doc_id returned by search_corpus."
        )

    # Deterministic anti-fabrication check: the quote must actually be in the cited document.
    # Whitespace-normalised containment, so reformatting is tolerated but invention is not.
    haystack = _normalise(" ".join(c.text for c in chunks))
    needle = _normalise(args.supporting_text)
    if needle not in haystack:
        raise ToolError(
            f"supporting_text was not found in '{args.source_doc_id}'. Copy the passage "
            f"verbatim from the document rather than paraphrasing it."
        )

    # Source reliability caps confidence. An assumption is never better than low confidence.
    ceiling = MAX_CONFIDENCE_BY_RELIABILITY.get(chunks[0].reliability, Confidence.LOW)
    confidence = args.confidence
    adjusted = False
    if _CONF_ORDER[confidence] > _CONF_ORDER[ceiling]:
        confidence, adjusted = ceiling, True
    if args.claim_type is ClaimType.ASSUMPTION and confidence is not Confidence.LOW:
        confidence, adjusted = Confidence.LOW, True

    from app.graph.state import evidence_id_for

    task_id = ctx.task_id or "R0"
    evidence_id = evidence_id_for(task_id, len(ctx.collected) + 1)
    item = Evidence(
        evidence_id=evidence_id,
        claim=args.claim,
        supporting_text=args.supporting_text,
        source_id=args.source_doc_id,
        source_title=chunks[0].doc_title,
        retrieved_at=datetime.now(timezone.utc),
        research_question=args.research_question or ctx.research_question or "unspecified",
        confidence=confidence,
        claim_type=args.claim_type,
        agent_id=AgentId.RESEARCHER.value,
        task_id=task_id,
    )
    ctx.collected.append(item)

    note = ""
    if adjusted:
        note = (
            f"Confidence lowered to {confidence.value}: source reliability is "
            f"'{chunks[0].reliability}'."
        )
    return StoreEvidenceOutput(
        evidence_id=evidence_id, stored=True, confidence=confidence,
        adjusted=adjusted, note=note,
    )


# --------------------------------------------------------------------------- #
# 4. retrieve_evidence
# --------------------------------------------------------------------------- #
class RetrieveInput(BaseModel):
    research_question: str | None = Field(
        default=None, description="Filter to evidence answering this question"
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Fetch these specific ids"
    )
    index_only: bool = Field(
        default=False,
        description="Return one-line summaries without supporting text, to save context",
    )


class RetrieveOutput(BaseModel):
    items: list[str] = Field(default_factory=list)
    count: int = 0
    missing_ids: list[str] = Field(default_factory=list)


@register_tool(
    name="retrieve_evidence",
    description=(
        "Read evidence already gathered. Use index_only=true for a compact list of what exists; "
        "omit it to read supporting text. Never returns evidence that was not stored."
    ),
    input_model=RetrieveInput,
    output_model=RetrieveOutput,
    allowed_agents={AgentId.SUPERVISOR, AgentId.RESEARCHER, AgentId.ANALYST,
                    AgentId.FACT_CHECKER, AgentId.CRITIC, AgentId.WRITER},
    rationale=(
        "The one tool every agent needs, because the evidence store is the shared substrate the "
        "whole workflow reasons over. It is read-only, so wide access carries no risk of "
        "polluting the record. ``missing_ids`` is what turns a fabricated citation into a "
        "detectable event rather than a silent empty result."
    ),
)
def retrieve_evidence(args: RetrieveInput, ctx: ToolContext) -> RetrieveOutput:
    pool = list(ctx.evidence)
    missing: list[str] = []

    if args.evidence_ids:
        by_id = {e.evidence_id: e for e in pool}
        selected = [by_id[i] for i in args.evidence_ids if i in by_id]
        missing = [i for i in args.evidence_ids if i not in by_id]
        pool = selected
    if args.research_question:
        wanted = _normalise(args.research_question)
        pool = [e for e in pool if _normalise(e.research_question) == wanted]

    render = (lambda e: e.index_line()) if args.index_only else (
        lambda e: e.cite(settings.evidence_snippet_chars)
    )
    return RetrieveOutput(
        items=[render(e) for e in pool], count=len(pool), missing_ids=missing
    )
