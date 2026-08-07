"""
Multi-agent workflow console (§24).

Deliberately **not** a chat app. Weeks 1-3 of this fellowship were chat interfaces; §24 asks for
observability — "a simple visualisation is sufficient, the goal is observability, not animation"
— which is a different product. The layout is a pipeline view plus tabs onto the workflow state,
so a reader can see which agent is running, what evidence exists, and why the Critic objected,
without reading a log.

Two Week 3 lessons are designed in rather than rediscovered:

1. **The session object lives in ``st.session_state``.** Streamlit re-runs this module top to
   bottom on every interaction. A ``WorkflowSession`` rebuilt per rerun would lose its
   checkpointer, and the ``interrupt()`` pause would become unresumable.

2. **Errors are persisted, never rendered immediately before a rerun.** In Week 3 an
   ``st.error(...)`` followed by ``st.rerun()`` wiped the message before it painted, so every
   failure — including rate limits — looked like "the agent thinks, then nothing happens". Here
   messages go into ``st.session_state.notice`` and are rendered at the top of the *next* run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# `streamlit run app/main.py` puts app/ on sys.path, not the project root, so `import app.*`
# fails with ModuleNotFoundError. Prepend the root before any first-party import.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Streamlit Cloud provides secrets rather than a .env file; bridge them before config loads.
try:  # pragma: no cover - only exercised on Streamlit Cloud
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:  # noqa: BLE001
    pass

from app import theme  # noqa: E402
from app.config import settings  # noqa: E402
from app.graph.workflow import WorkflowSession  # noqa: E402
from app.schemas.common import AgentId, TaskStatus, WorkflowStatus  # noqa: E402
from app.services.llm_service import model_backend_status  # noqa: E402
from app.storage.corpus import get_index  # noqa: E402
from app.storage.evidence_store import EvidenceStore  # noqa: E402

st.set_page_config(page_title="Multi-Agent Research Platform", page_icon=theme.page_icon(),
                   layout="wide", initial_sidebar_state="expanded")

SAMPLES = [
    "Compare LangGraph and CrewAI for a 4-person Python team building a production "
    "multi-agent support system. Recommend one.",
    "Compare three cloud platforms for deploying an AI SaaS application at moderate scale.",
    "Which AI coding assistant should a 300-engineer organisation standardise on?",
    "Find the best AI framework.",          # deliberately vague — triggers clarification
]

# The pipeline, in execution order. Each entry maps a display name to the trace nodes that
# indicate it ran, so status is derived from the trace rather than tracked separately.
PIPELINE: list[tuple[str, str, tuple[str, ...]]] = [
    ("🧭 Supervisor · analyse", "supervisor_analyse", ("supervisor_analyse",)),
    ("🧭 Supervisor · plan", "supervisor_plan", ("supervisor_plan",)),
    ("⏸ Plan approval", "plan_approval", ("plan_approval",)),
    ("🔍 Researchers", "research", ("research", "research_dispatch", "researcher",
                                    "researcher_summary")),
    ("⚖️ Evidence gate", "evidence_gate", ("evidence_gate",)),
    ("📊 Analyst", "analyst", ("analyst",)),
    ("🧾 Fact-Checker", "fact_checker", ("fact_checker",)),
    ("⚔️ Critic", "critic", ("critic",)),
    ("✍️ Writer", "writer", ("writer",)),
    ("⏸ Final review", "final_review", ("final_review",)),
]


# --------------------------------------------------------------------------- #
# Session helpers
# --------------------------------------------------------------------------- #
def notice(kind: str, message: str) -> None:
    """Queue a message for the next run.

    Never call ``st.error`` immediately before ``st.rerun`` — the rerun discards the widget
    before it paints and the user sees silence where an error should be.
    """
    st.session_state.notice = (kind, message)


def render_notice() -> None:
    pending = st.session_state.pop("notice", None)
    if not pending:
        return
    kind, message = pending
    {"error": st.error, "warning": st.warning, "success": st.success}.get(kind, st.info)(message)


def current_state() -> dict:
    session = st.session_state.get("session")
    if session is None:
        return {}
    return st.session_state.get("last_state") or session.snapshot() or {}


def pipeline_statuses(state: dict) -> list[str]:
    """Status for every stage, in order.

    Computed as a whole rather than per-stage because "running" is only meaningful relative to
    the others: the first stage the trace has not reached is the one running, and everything
    after it is queued. Deciding per stage in isolation marked *every* unreached stage as
    running, so the whole pipeline showed spinners at once and the view said nothing.
    """
    seen = {e.node for e in state.get("trace", [])}
    pending = st.session_state.get("pending")
    active = bool(st.session_state.get("running"))

    statuses: list[str] = []
    found_current = False
    for _label, _key, nodes in PIPELINE:
        if pending and pending.get("gate") in nodes:
            statuses.append("waiting_human")
            found_current = True
        elif any(n in seen for n in nodes):
            statuses.append("done")
        elif active and not found_current:
            statuses.append("running")
            found_current = True
        else:
            statuses.append("pending")
    return statuses


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    with st.sidebar:
        st.markdown("### Request")
        # Passing BOTH `value=` and `key=` to a Streamlit widget is a trap: after the first
        # render session_state wins and `value` is silently ignored, so the box stayed empty and
        # the Run button stayed disabled with no explanation. The preset is instead pushed into
        # session_state by an on_change callback, which also preserves the user's own edits
        # across reruns.
        if "request_text" not in st.session_state:
            st.session_state.request_text = SAMPLES[0]

        def _apply_preset() -> None:
            choice = st.session_state.get("sample_choice")
            if choice and choice != "(write my own)":
                st.session_state.request_text = choice

        st.selectbox("Sample requests", ["(write my own)"] + SAMPLES, index=1,
                     key="sample_choice", on_change=_apply_preset)
        request = st.text_area("What should the agents research?", height=130,
                               key="request_text")

        st.markdown("### Run options")
        col_a, col_b = st.columns(2)
        with col_a:
            hitl = st.toggle("Human checkpoints", value=True,
                             help="Pause for plan approval and final review (§20)")
            parallel = st.toggle("Parallel research", value=True,
                                 help="Fan out researchers concurrently (§19)")
        with col_b:
            critic_on = st.toggle("Critic enabled", value=True,
                                  help="Disable to reproduce Experiment 2's control arm")
            persist = st.toggle("Persist run", value=EvidenceStore().enabled,
                                disabled=not EvidenceStore().enabled,
                                help="Write evidence, trace and report to Postgres")

        disabled = st.session_state.get("running", False) or not request.strip()
        if st.button("▶  Run workflow", type="primary", use_container_width=True,
                     disabled=disabled):
            start_run(request, hitl, parallel, critic_on, persist)

        if st.session_state.get("session") and st.button("↺  Reset", use_container_width=True):
            for key in ("session", "last_state", "pending", "running", "result"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.markdown("### AI model")
        # One anonymous row. Vendor identity is an implementation detail and would otherwise
        # appear in every screenshot and demo recording — see model_backend_status().
        live, backends = model_backend_status()
        if live:
            st.markdown(f"🟢 Ready · {backends} backend{'s' if backends != 1 else ''}")
            st.caption("Extra backends widen the daily allowance and provide failover."
                       if backends > 1 else
                       "Single backend: a rate limit ends the run instead of failing over.")
        else:
            st.markdown("⚪ No model backend configured")
            st.caption("Set an API key in your environment or deployment secrets.")

        st.divider()
        st.markdown("### Budget")
        session = st.session_state.get("session")
        usage = session.deps.usage if session else None
        calls = usage.billable_calls if usage else 0
        st.progress(min(1.0, calls / max(1, settings.max_agent_calls_per_run)),
                    text=f"{calls}/{settings.max_agent_calls_per_run} model calls")
        if usage:
            st.caption(f"{usage.total_input_tokens:,} in / {usage.total_output_tokens:,} out "
                       f"· {usage.refused_calls} refused · {usage.elapsed_seconds:.0f}s")
        st.caption(f"Caps: {settings.max_revision_cycles} revisions · "
                   f"{settings.max_research_rounds} research rounds · "
                   f"{settings.max_run_seconds}s")


def start_run(request: str, hitl: bool, parallel: bool, critic_on: bool, persist: bool) -> None:
    session = WorkflowSession(
        request,
        index=get_index(),
        store=EvidenceStore() if persist else None,
        human_in_the_loop=hitl,
        parallel_research=parallel,
        critic_enabled=critic_on,
    )
    st.session_state.session = session
    st.session_state.running = True
    st.session_state.pending = None
    st.session_state.last_state = None
    st.session_state.result = None
    st.rerun()


def advance(decision=None) -> None:
    """Run or resume the workflow, capturing failures as a queued notice."""
    session = st.session_state.session
    try:
        result = session.resume(decision) if decision is not None else session.start()
    except Exception as e:  # noqa: BLE001
        st.session_state.running = False
        notice("error", f"The workflow could not continue: {type(e).__name__}: {e}")
        st.rerun()
        return

    st.session_state.result = result
    st.session_state.last_state = result.state
    st.session_state.pending = session.pending_interrupt()
    st.session_state.running = bool(st.session_state.pending)

    if not st.session_state.pending:
        status = result.status
        if status is WorkflowStatus.COMPLETED:
            notice("success", f"Workflow completed in {result.wall_seconds:.0f}s.")
        elif status is WorkflowStatus.AWAITING_CLARIFICATION:
            notice("warning", "The request needs clarification before research can start.")
        elif status is WorkflowStatus.ABORTED:
            notice("warning", result.state.get("abort_reason", "Run aborted."))
        else:
            notice("error", result.state.get("abort_reason", "The workflow failed."))
    st.rerun()


# --------------------------------------------------------------------------- #
# Pipeline view
# --------------------------------------------------------------------------- #
def pipeline_view(state: dict) -> None:
    st.markdown("#### Pipeline")
    icons = {"done": "✅", "running": "⏳", "waiting_human": "⏸", "pending": "○"}
    trace = state.get("trace", [])
    durations: dict[str, float] = {}
    for event in trace:
        durations[event.node] = durations.get(event.node, 0.0) + event.duration_seconds

    for (label, _key, nodes), status in zip(PIPELINE, pipeline_statuses(state)):
        seconds = sum(durations.get(n, 0.0) for n in nodes)
        left, right = st.columns([5, 2])
        left.markdown(f"{icons[status]}  {label}")
        right.markdown(f"<div style='text-align:right;opacity:.65'>"
                       f"{seconds:.1f}s</div>" if seconds else
                       "<div style='text-align:right;opacity:.35'>—</div>",
                       unsafe_allow_html=True)

    plan = state.get("plan")
    if plan:
        done, total = plan.progress()
        st.progress(done / max(1, total), text=f"{done}/{total} tasks complete")


def metrics_row(state: dict) -> None:
    result = st.session_state.get("result")
    usage = st.session_state.session.deps.usage if st.session_state.get("session") else None
    cols = st.columns(5)
    cols[0].metric("Evidence", len(state.get("evidence", [])))
    cols[1].metric("Revisions", f"{state.get('revision_count', 0)}/"
                                f"{settings.max_revision_cycles}")
    cols[2].metric("Model calls", usage.billable_calls if usage else 0)
    cols[3].metric("Errors", len(state.get("errors", [])))
    cols[4].metric("Elapsed", f"{result.wall_seconds:.0f}s" if result else "—")


# --------------------------------------------------------------------------- #
# Human checkpoints
# --------------------------------------------------------------------------- #
def checkpoint_panel(pending: dict) -> None:
    gate = pending.get("gate")
    if gate == "plan_approval":
        st.warning("⏸ **Human checkpoint 1 — research plan approval**", icon="⏸")
        st.markdown(f"**Objective:** {pending.get('objective', '')}")
        if pending.get("sub_questions"):
            st.markdown("**Sub-questions**")
            for i, q in enumerate(pending["sub_questions"], 1):
                st.markdown(f"{i}. {q}")
        if pending.get("evaluation_criteria"):
            st.markdown("**Criteria:** " + ", ".join(pending["evaluation_criteria"]))
        st.code(pending.get("plan", ""), language="text")
        st.caption(f"{pending.get('research_task_count', 0)} research tasks will run. "
                   f"Nothing has been researched yet.")
    else:
        st.warning("⏸ **Human checkpoint 2 — final recommendation review**", icon="⏸")
        st.markdown(f"### {pending.get('title', '')}")
        if pending.get("recommendation"):
            st.success(f"**Recommendation:** {pending['recommendation']}  \n"
                       f"Confidence: {pending.get('confidence', 'unknown')}")
        if pending.get("critic_approved") is False:
            st.error("The Critic did not approve this analysis and the revision limit was "
                     "reached. Unresolved objections:")
            for issue in pending.get("unresolved_objections", []):
                st.markdown(f"- {issue}")
        for finding in pending.get("key_findings", [])[:5]:
            st.markdown(f"- {finding}")
        with st.expander(f"Full report · {pending.get('evidence_count', 0)} cited sources"):
            st.markdown(pending.get("markdown", ""))

    note = st.text_input("Note (optional)", key=f"note_{gate}")
    cols = st.columns(3 if gate == "plan_approval" else 2)
    if cols[0].button("✅ Approve", type="primary", use_container_width=True):
        advance({"decision": "approve", "note": note})
    if gate == "plan_approval":
        if cols[1].button("✏️ Edit", use_container_width=True,
                          help="Not editable from the UI yet — approve or reject"):
            notice("info", "Plan editing is available through the API "
                           "(resume with decision='edit'); the UI exposes approve and reject.")
            st.rerun()
        reject_col = cols[2]
    else:
        reject_col = cols[1]
    if reject_col.button("⛔ Reject", use_container_width=True):
        advance({"decision": "reject", "note": note})


# --------------------------------------------------------------------------- #
# Detail tabs
# --------------------------------------------------------------------------- #
def detail_tabs(state: dict) -> None:
    tabs = st.tabs(["Plan", "Evidence", "Analysis", "Critic", "Report", "Trace", "Cost"])

    with tabs[0]:
        plan = state.get("plan")
        brief = state.get("brief")
        if brief:
            st.markdown(f"**Objective:** {brief.objective}")
            if brief.clarifying_questions:
                st.warning("The Supervisor asked for clarification:")
                for q in brief.clarifying_questions:
                    st.markdown(f"- **{q.question}** — {q.why_it_matters}")
        if plan:
            st.code(plan.render(), language="text")
            statuses = state.get("task_status", {})
            if statuses:
                st.markdown("**Task status**")
                for task_id, status in sorted(statuses.items()):
                    mark = {TaskStatus.COMPLETED: "✅", TaskStatus.FAILED: "❌",
                            TaskStatus.SKIPPED: "⏭"}.get(status, "○")
                    st.markdown(f"{mark} `{task_id}` {getattr(status, 'value', status)}")
        elif not brief:
            st.caption("No plan yet.")

    with tabs[1]:
        evidence = state.get("evidence", [])
        st.caption(f"{len(evidence)} records. Every claim in the report traces to one of these.")
        for item in evidence:
            with st.expander(f"`{item.evidence_id}` · {item.claim_type.value} · "
                             f"{item.confidence.value} · {item.claim[:70]}"):
                st.markdown(f"> {item.supporting_text}")
                st.caption(f"{item.source_title} (`{item.source_id}`) · "
                           f"question: {item.research_question}")
        for handoff in state.get("research_handoffs", []):
            for gap in handoff.gaps:
                st.warning(f"Gap — {gap.research_question}: {gap.reason}")

    with tabs[2]:
        analysis = state.get("analysis")
        if not analysis:
            st.caption("No analysis yet.")
        else:
            st.markdown(f"**Summary.** {analysis.summary}")
            for c in analysis.conclusions:
                badge = "**MAJOR**" if c.is_major else "minor"
                st.markdown(f"- [{c.conclusion_id}] {badge} ({c.confidence.value}) "
                            f"{c.statement}  \n  `{', '.join(c.evidence_ids) or 'uncited'}`")
            if analysis.assumptions:
                st.markdown("**Assumptions**")
                for a in analysis.assumptions:
                    st.markdown(f"- {a}")

    with tabs[3]:
        fact_check = state.get("fact_check")
        if fact_check and fact_check.fabricated:
            st.error(f"Fabricated citations detected: {', '.join(fact_check.fabricated)}")
        for verdict in state.get("critic_verdicts", []):
            head = "✅ approved" if verdict.approved else "⛔ rejected"
            st.markdown(f"**Cycle {verdict.cycle} — {head}**")
            for problem in verdict.problems:
                st.markdown(f"- `{problem.criterion.value}` / {problem.severity.value} — "
                            f"{problem.issue}")
            if verdict.required_revisions:
                st.caption("Required: " + "; ".join(verdict.required_revisions))
            if verdict.scores:
                st.caption(" · ".join(f"{k.value}: {v}/5" for k, v in verdict.scores.items()))
        if not state.get("critic_verdicts"):
            st.caption("No review yet.")

    with tabs[4]:
        report = state.get("report")
        if not report:
            st.caption("No report yet.")
        else:
            st.download_button("⬇ Download Markdown", report.to_markdown(),
                               file_name=f"report_{state.get('run_id', 'run')}.md",
                               mime="text/markdown")
            st.markdown(report.to_markdown())

    with tabs[5]:
        trace = state.get("trace", [])
        st.caption(f"{len(trace)} events. Operational only — no chain-of-thought is stored (§23).")
        if trace:
            st.code("\n".join(e.line() for e in trace), language="text")
        for error in state.get("errors", []):
            st.error(f"`{error.kind}` {error.agent_id.value} — {error.message}"
                     + (f"  (recovered: {error.action_taken})" if error.recovered else ""))
        decisions = state.get("human_decisions", [])
        if decisions:
            st.markdown("**Human decisions**")
            for d in decisions:
                st.markdown(f"- `{d['gate']}` → **{d['decision']}** {d.get('note', '')}")

    with tabs[6]:
        session = st.session_state.get("session")
        if not session:
            st.caption("No run yet.")
        else:
            summary = session.deps.usage.summary()
            st.markdown(f"**{summary['billable_calls']} billable calls** "
                        f"({summary['total_calls']} attempted, "
                        f"{summary['refused_calls']} refused) · "
                        f"{summary['input_tokens']:,} in / {summary['output_tokens']:,} out")
            rows = summary["by_agent"]
            if rows:
                st.dataframe(
                    [{"agent": a, "calls": r["calls"], "in": r["input_tokens"],
                      "out": r["output_tokens"], "seconds": round(r["seconds"], 1),
                      "failures": r["failures"]}
                     for a, r in sorted(rows.items())],
                    use_container_width=True, hide_index=True)
            st.caption("Cost reads $0.00 on free provider tiers. Token counts are measured.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # The ported Week 3 design system. `appearance_toggle` owns the shared dark/light key, so it
    # must run before `inject_css` reads it.
    dark = theme.appearance_toggle()
    theme.inject_css(dark if dark is not None else True)
    theme.render_hero(
        "Multi-Agent Research Platform",
        "Six specialised agents research, argue, and cite their sources.",
        pill="Week 4",
    )
    render_notice()
    sidebar()

    session = st.session_state.get("session")
    if session is None:
        st.info("Choose a sample request in the sidebar and press **Run workflow**.")
        st.markdown(
            "The Supervisor decomposes your request, researchers gather evidence in parallel, "
            "an Analyst compares the options, a Fact-Checker verifies every citation and a "
            "Critic can send the work back. Two checkpoints pause for your approval."
        )
        return

    state = current_state()
    left, right = st.columns([2, 5], gap="large")
    with left:
        pipeline_view(state)
    with right:
        metrics_row(state)
        pending = st.session_state.get("pending")
        if pending:
            checkpoint_panel(pending)
        elif st.session_state.get("running"):
            with st.status("Running the workflow…", expanded=False):
                advance()
        detail_tabs(state)


main()
