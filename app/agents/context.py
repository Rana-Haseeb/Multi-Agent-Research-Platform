"""
Per-agent context assembly (§21).

§21's instruction is "do not send the complete workflow history to every agent". The reason is
not only cost. An agent given everything must decide what is relevant before it can start, and
that decision is where irrelevant material leaks into output — a Critic that can see raw corpus
text starts reviewing the corpus instead of the analysis.

So each agent has a builder here that returns exactly its slice, and the slices differ in kind,
not just in size:

    Supervisor    plan state + evidence COUNTS. Never evidence bodies — it routes, it doesn't read.
    Researcher    its own sub-question. Nothing from sibling researchers, so findings stay independent.
    Analyst       evidence grouped by question, supporting text truncated to a budget. No corpus.
    Fact-Checker  only the conclusions and only the evidence they cite. Full text, narrow set.
    Critic        the analysis in full + a one-line evidence INDEX. Knowing what evidence exists
                  is enough to judge coverage; reading all of it is not necessary and is the
                  single largest avoidable token cost in the workflow.
    Writer        approved analysis + cited evidence only. No corpus, no rejected drafts.

Experiment 4 compares these builders against a full-context control and measures the difference
in tokens and output relevance. :func:`full_context_control` is that control, and it lives here
rather than in the experiment so both arms are built from the same state.
"""
from __future__ import annotations

from app.config import settings
from app.schemas.evidence import Evidence, EvidenceIndex
from app.schemas.handoffs import AnalysisHandoff, FactCheckReport, ResearchHandoff
from app.schemas.request import RequestBrief
from app.schemas.tasks import TaskPlan


def _brief_block(brief: RequestBrief) -> str:
    lines = [f"OBJECTIVE: {brief.objective}"]
    if brief.sub_questions:
        lines.append("SUB-QUESTIONS:")
        lines += [f"  {i}. {q}" for i, q in enumerate(brief.sub_questions, 1)]
    if brief.evaluation_criteria:
        lines.append("EVALUATION CRITERIA: " + "; ".join(brief.evaluation_criteria))
    if brief.options_under_comparison:
        lines.append("OPTIONS: " + ", ".join(brief.options_under_comparison))
    if brief.constraints:
        lines.append("CONSTRAINTS: " + "; ".join(brief.constraints))
    if brief.deliverable:
        lines.append(f"DELIVERABLE: {brief.deliverable}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Supervisor — routing state only
# --------------------------------------------------------------------------- #
def supervisor_context(
    user_request: str,
    brief: RequestBrief | None,
    plan: TaskPlan | None,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
) -> str:
    """Counts, not content.

    The Supervisor decides what happens next. That decision needs to know how much evidence
    exists per question, not what it says — and feeding it evidence bodies would invite it to
    start analysing, which is another agent's job.
    """
    parts = [f"USER REQUEST: {user_request}"]
    if brief:
        parts += ["", _brief_block(brief)]
    if plan:
        done, total = plan.progress()
        parts += ["", f"PLAN: {done}/{total} tasks terminal", plan.render()]
    if brief and evidence:
        index = EvidenceIndex(items=evidence)
        coverage = index.coverage(brief.sub_questions)
        parts += ["", "EVIDENCE COVERAGE (confidence-weighted, need >= 1.0):"]
        parts += [f"  {score:>5.2f}  {q}" for q, score in coverage.items()]
    elif brief:
        parts += ["", "EVIDENCE: none gathered yet"]
    if handoffs:
        gaps = [g.research_question for h in handoffs for g in h.gaps]
        if gaps:
            parts += ["", "DECLARED GAPS: " + "; ".join(sorted(set(gaps)))]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Researcher — its own question, and nothing else
# --------------------------------------------------------------------------- #
def researcher_context(task_id: str, research_question: str, objective: str) -> str:
    """One question, plus the objective for disambiguation only.

    Sibling researchers' findings are deliberately withheld. Independent gathering is what makes
    two researchers finding the same fact meaningful rather than an echo, and it is what keeps
    the parallel branches genuinely parallel.
    """
    return (
        f"OVERALL OBJECTIVE (for context only — do not answer this):\n{objective}\n\n"
        f"YOUR ASSIGNED RESEARCH QUESTION (task {task_id}):\n{research_question}\n\n"
        f"Search the corpus, store evidence for what you find, and report your findings for this "
        f"question only."
    )


# --------------------------------------------------------------------------- #
# Analyst — evidence grouped by question, truncated
# --------------------------------------------------------------------------- #
def analyst_context(
    brief: RequestBrief,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
    revision_note: str = "",
    max_items: int | None = None,
) -> str:
    """Evidence grouped by the question it answers, supporting text capped.

    Grouping matters: ungrouped evidence makes the Analyst re-derive which finding answers which
    question, and it does that imperfectly. The per-item truncation is
    ``settings.evidence_snippet_chars``.
    """
    cap = max_items or settings.max_evidence_items_analyst
    parts = [_brief_block(brief), ""]

    index = EvidenceIndex(items=evidence)
    grouped = index.by_question()
    parts.append(f"EVIDENCE ({len(evidence)} items across {len(grouped)} questions):")
    shown = 0
    for question, items in grouped.items():
        parts.append(f"\n--- {question} ---")
        for item in items:
            if shown >= cap:
                parts.append(f"  [{len(evidence) - shown} further items omitted for length]")
                break
            parts.append(item.cite(settings.evidence_snippet_chars))
            shown += 1
        if shown >= cap:
            break

    gaps = [(g.research_question, g.reason) for h in handoffs for g in h.gaps]
    if gaps:
        parts += ["", "DECLARED EVIDENCE GAPS (do not fill these from your own knowledge):"]
        parts += [f"  - {q}: {reason}" for q, reason in gaps]

    unresolved = index.unresolved(brief.sub_questions)
    if unresolved:
        parts += ["", "QUESTIONS WITH THIN OR NO EVIDENCE:"]
        parts += [f"  - {q}" for q in unresolved]

    if revision_note:
        parts += ["", "REVISION REQUIRED — the Critic rejected your previous analysis:",
                  revision_note,
                  "Address every point above. Do not restate the previous analysis unchanged."]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Fact-Checker — conclusions and only the evidence they cite
# --------------------------------------------------------------------------- #
def fact_checker_context(analysis: AnalysisHandoff, evidence: list[Evidence]) -> str:
    """Narrow and deep: every conclusion, but only the evidence actually cited.

    Uncited evidence is irrelevant to this agent's question, and including it would invite it to
    judge completeness — the Critic's job, not this one's.
    """
    by_id = {e.evidence_id: e for e in evidence}
    parts = ["CONCLUSIONS TO CHECK:", ""]
    for c in analysis.conclusions:
        parts.append(
            f"[{c.conclusion_id}]{' (MAJOR)' if c.is_major else ''} {c.statement}"
        )
        if not c.evidence_ids:
            parts.append("    cites: nothing")
        for eid in c.evidence_ids:
            item = by_id.get(eid)
            if item is None:
                parts.append(f"    cites {eid}: DOES NOT EXIST (already flagged mechanically)")
            else:
                parts.append(
                    f"    cites {eid} ({item.claim_type.value}, {item.confidence.value}, "
                    f"src: {item.source_title}):\n"
                    f'        "{item.supporting_text[:settings.evidence_snippet_chars]}"'
                )
        parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Critic — full analysis, one-line evidence index
# --------------------------------------------------------------------------- #
def critic_context(
    brief: RequestBrief,
    analysis: AnalysisHandoff,
    evidence: list[Evidence],
    fact_check: FactCheckReport | None,
    handoffs: list[ResearchHandoff],
    cycle: int = 0,
) -> str:
    """The largest single context saving in the system.

    The Critic judges coverage, consistency and completeness. For coverage it needs to know what
    evidence *exists*; it does not need every supporting passage. Sending one line per item
    instead of full bodies is what Experiment 4 measures.
    """
    parts = [_brief_block(brief), ""]

    if cycle:
        parts += [f"NOTE: this is revision cycle {cycle}. Judge the current version on its "
                  f"merits, not on whether it improved.", ""]

    parts += ["ANALYSIS UNDER REVIEW", "", f"Summary: {analysis.summary}", "", "Conclusions:"]
    for c in analysis.conclusions:
        cites = ", ".join(c.evidence_ids) or "NO CITATIONS"
        parts.append(
            f"  [{c.conclusion_id}]{' MAJOR' if c.is_major else ''} "
            f"({c.confidence.value}) {c.statement}\n      cites: {cites}"
        )
    if analysis.comparison:
        parts += ["", "Comparison:"]
        for row in analysis.comparison:
            scores = "; ".join(f"{k}: {v}" for k, v in row.scores.items())
            parts.append(f"  {row.option}: {scores}")
    if analysis.trade_offs:
        parts += ["", "Trade-offs:"] + [f"  - {t}" for t in analysis.trade_offs]
    if analysis.assumptions:
        parts += ["", "Stated assumptions:"] + [f"  - {a}" for a in analysis.assumptions]

    # The index, not the bodies.
    index = EvidenceIndex(items=evidence)
    parts += ["", f"EVIDENCE INDEX ({len(evidence)} items — summaries only):"]
    parts += [f"  {e.index_line()}" for e in evidence[: settings.max_evidence_items_critic]]
    if len(evidence) > settings.max_evidence_items_critic:
        parts.append(f"  [{len(evidence) - settings.max_evidence_items_critic} more]")

    coverage = index.coverage(brief.sub_questions)
    parts += ["", "COVERAGE PER SUB-QUESTION (confidence-weighted):"]
    parts += [f"  {score:>5.2f}  {q}" for q, score in coverage.items()]

    gaps = [(g.research_question, g.reason) for h in handoffs for g in h.gaps]
    if gaps:
        parts += ["", "DECLARED GAPS:"] + [f"  - {q}: {r}" for q, r in gaps]

    if fact_check and fact_check.checks:
        parts += ["", "FACT-CHECK REPORT:"]
        if fact_check.fabricated:
            parts.append(f"  FABRICATED CITATIONS: {', '.join(fact_check.fabricated)}")
        if fact_check.unsupported_ids:
            parts.append(f"  NOT SUPPORTED BY CITED EVIDENCE: "
                         f"{', '.join(fact_check.unsupported_ids)}")
        if fact_check.is_clean():
            parts.append("  All citations exist and are supported.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Writer — approved analysis and cited evidence
# --------------------------------------------------------------------------- #
def writer_context(
    brief: RequestBrief,
    analysis: AnalysisHandoff,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
    critic_note: str = "",
) -> str:
    """Only what survived review.

    Rejected drafts and superseded critic feedback are withheld: the Writer's job is to present
    the approved analysis, and showing it earlier rejected versions invites it to re-litigate
    decisions the Critic already made.
    """
    cited = analysis.cited_ids()
    parts = [_brief_block(brief), "",
             "APPROVED ANALYSIS", "", f"Summary: {analysis.summary}", "", "Conclusions:"]
    for c in analysis.conclusions:
        parts.append(f"  [{c.conclusion_id}] ({c.confidence.value}) {c.statement}"
                     f"  [{', '.join(c.evidence_ids) or 'uncited'}]")
    if analysis.comparison:
        parts += ["", "Comparison:"]
        for row in analysis.comparison:
            scores = "; ".join(f"{k}: {v}" for k, v in row.scores.items())
            parts.append(f"  {row.option}: {scores}  [{', '.join(row.evidence_ids)}]")
    if analysis.trade_offs:
        parts += ["", "Trade-offs:"] + [f"  - {t}" for t in analysis.trade_offs]
    if analysis.assumptions:
        parts += ["", "Assumptions the evidence does not establish (must appear in Limitations):"]
        parts += [f"  - {a}" for a in analysis.assumptions]

    parts += ["", f"CITED EVIDENCE ({len(cited)} items):"]
    parts += [f"  {e.index_line()}" for e in evidence if e.evidence_id in cited]

    gaps = [(g.research_question, g.reason) for h in handoffs for g in h.gaps]
    if gaps:
        parts += ["", "EVIDENCE GAPS (these MUST appear in Risks and Limitations):"]
        parts += [f"  - {q}: {r}" for q, r in gaps]

    if critic_note:
        parts += ["", "REVIEWER NOTES TO REFLECT:", critic_note]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Experiment 4 control
# --------------------------------------------------------------------------- #
def full_context_control(
    user_request: str,
    brief: RequestBrief | None,
    plan: TaskPlan | None,
    evidence: list[Evidence],
    handoffs: list[ResearchHandoff],
    analysis: AnalysisHandoff | None,
    fact_check: FactCheckReport | None,
) -> str:
    """Everything, to every agent — the naive approach Experiment 4 measures against.

    Deliberately unfiltered: full evidence bodies, all handoffs, the whole plan. This is what
    the role-specific builders above are being compared with, and it is what most first-draft
    multi-agent systems do by default.
    """
    parts = [f"USER REQUEST: {user_request}"]
    if brief:
        parts += ["", _brief_block(brief)]
        if brief.clarifying_questions:
            parts += ["CLARIFYING QUESTIONS ASKED:"]
            parts += [f"  - {q.question} ({q.why_it_matters})"
                      for q in brief.clarifying_questions]
    if plan:
        parts += ["", "FULL PLAN:", plan.render()]
    if handoffs:
        parts += ["", "ALL RESEARCH HANDOFFS:"]
        for h in handoffs:
            parts.append(f"  [{h.task_id}] {h.research_question}\n    {h.findings}\n"
                         f"    evidence: {', '.join(h.evidence_ids)} "
                         f"confidence: {h.confidence.value}")
            parts += [f"    gap: {g.research_question} — {g.reason}" for g in h.gaps]
    if evidence:
        parts += ["", f"ALL EVIDENCE ({len(evidence)} items, full text):"]
        parts += [e.cite(10_000) for e in evidence]
    if analysis:
        parts += ["", "ANALYSIS:", analysis.model_dump_json(indent=2)]
    if fact_check:
        parts += ["", "FACT CHECK:", fact_check.model_dump_json(indent=2)]
    return "\n".join(parts)
