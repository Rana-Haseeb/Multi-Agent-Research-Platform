<div align="center">

# 🧠 Multi-Agent Research & Decision Intelligence Platform

**Six specialised AI agents that research a hard question, argue about the answer,
and hand you a report where every claim traces back to a source.**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b-F55036?style=for-the-badge)](https://groq.com)
[![Postgres](https://img.shields.io/badge/Supabase-Postgres_17-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![Tests](https://img.shields.io/badge/tests-100_passing-success?style=for-the-badge)](tests/)

<samp>Visibility Bots Innovation Lab · AI Summer Fellowship 2026 · Track 2: NLP & AI Agents · **Week 4**</samp>

</div>

---

> ### The question this repo exists to answer
>
> **"Why does this workflow need multiple agents, and what measurable value does each one add?"**
>
> Multi-agent systems are not automatically better. They add latency, cost, failure modes and
> debugging pain. This project is an attempt to earn the complexity — and to measure honestly
> where it *wasn't* earned.

---

## Contents

- [What it does](#-what-it-does)
- [The 60-second version](#-the-60-second-version)
- [Architecture](#-architecture)
- [The six agents](#-the-six-agents)
- [What makes it reliable](#-what-makes-it-reliable)
- [Measured decisions](#-measured-decisions-no-vibes)
- [The corpus fights back](#-the-corpus-fights-back)
- [Quick start](#-quick-start)
- [Project structure](#-project-structure)
- [Build status](#-build-status)
- [Known limitations](#-known-limitations)

---

## 🎯 What it does

You give it a genuinely hard question:

```
Compare LangGraph, CrewAI and the OpenAI Agents SDK for a 4-person Python team
building a production multi-agent support system. Recommend one, with reasoning.
```

It does **not** forward that to one model and hope. It decomposes the question, researches each
part in parallel against a controlled corpus, records structured evidence, builds a comparison,
attacks its own conclusions, and produces a report in which **every major claim carries an
evidence ID that a reviewer can follow back to a source.**

Two points in the workflow stop and wait for a human.

---

## ⚡ The 60-second version

Think of a small consulting firm:

| A real firm | This system | Refuses to |
|---|---|---|
| 📋 **Project manager** — scopes the brief | **Supervisor** | do the research itself |
| 🔍 **Junior researchers** — dig through sources | **Researcher** ×N *(parallel)* | draw conclusions |
| 📊 **Senior analyst** — turns findings into a comparison | **Analyst** | search for new facts |
| 🧾 **Compliance** — checks every claim has a source | **Fact-Checker** | judge the argument |
| ⚔️ **Partner** — tears the draft apart | **Critic** | rewrite the answer |
| ✍️ **Writer** — produces the deliverable | **Writer** | invent research |

Nobody at a real firm does all six jobs. **That's the whole argument** — and the "refuses to"
column is enforced in code, not requested in a prompt.

---

## 🏗 Architecture

```mermaid
flowchart TD
    START([User request]) --> INTAKE[["intake · deterministic<br/>validate · run_id · budget"]]
    INTAKE --> ANALYSE["🧭 SUPERVISOR<br/>analyse request"]

    ANALYSE -->|ambiguous| CLARIFY{{"⏸ interrupt<br/>ask the user"}}
    CLARIFY --> ANALYSE
    ANALYSE -->|clear| PLAN["🧭 SUPERVISOR<br/>build dynamic task plan"]

    PLAN --> GATE1{{"⏸ HUMAN CHECKPOINT 1<br/>approve · edit · reject plan"}}
    GATE1 --> FANOUT(["Send() fan-out"])

    FANOUT --> R1["🔍 RESEARCHER<br/>sub-question A"]
    FANOUT --> R2["🔍 RESEARCHER<br/>sub-question B"]
    FANOUT --> R3["🔍 RESEARCHER<br/>sub-question C"]

    R1 --> EGATE
    R2 --> EGATE
    R3 --> EGATE

    EGATE[["evidence gate · deterministic<br/>confidence-weighted coverage"]]
    EGATE -->|thin, round 1| PLAN
    EGATE -->|sufficient| ANALYST["📊 ANALYST<br/>compare · trade-offs"]

    ANALYST --> FC["🧾 FACT-CHECKER<br/>code: does the ID exist?<br/>model: does it support?"]
    FC --> CRITIC["⚔️ CRITIC<br/>6 criteria · can reject"]

    CRITIC -->|revise · max 2| ANALYST
    CRITIC -->|needs research| PLAN
    CRITIC -->|approved| WRITER["✍️ WRITER<br/>no search tool"]

    WRITER --> GATE2{{"⏸ HUMAN CHECKPOINT 2<br/>review recommendation"}}
    GATE2 --> OUT([📄 Markdown report<br/>+ execution trace])

    style GATE1 fill:#7c5cff,stroke:#5b3fd6,color:#fff
    style GATE2 fill:#7c5cff,stroke:#5b3fd6,color:#fff
    style CLARIFY fill:#7c5cff,stroke:#5b3fd6,color:#fff
    style INTAKE fill:#1f6feb,stroke:#1158c7,color:#fff
    style EGATE fill:#1f6feb,stroke:#1158c7,color:#fff
    style CRITIC fill:#d1242f,stroke:#a40e26,color:#fff
    style OUT fill:#1a7f37,stroke:#116329,color:#fff
```

<div align="center">
<sub>🟦 deterministic code · ⬜ model call · 🟪 human gate · 🟥 quality control</sub>
</div>

**The blue nodes are the interesting part.** Routing decisions that don't need judgement don't
get a model call. Coverage scoring, budget enforcement, citation existence and duplicate
detection are all plain Python — a model is spent only where judgement is actually required.

---

## 🤖 The six agents

<table>
<tr><th>Agent</th><th>Owns</th><th>Tools</th><th>Cannot</th></tr>
<tr>
<td>🧭 <b>Supervisor</b></td>
<td>Request analysis, dynamic plan, routing, completion</td>
<td><code>retrieve_evidence</code></td>
<td>Search. It plans research; it doesn't do it.</td>
</tr>
<tr>
<td>🔍 <b>Researcher</b><br/><sub>N in parallel</sub></td>
<td>One sub-question each. Gathers and stores evidence.</td>
<td><code>search_corpus</code> <code>extract_document</code> <code>store_evidence</code> <code>retrieve_evidence</code></td>
<td>Draw conclusions or compute figures a source never stated.</td>
</tr>
<tr>
<td>📊 <b>Analyst</b></td>
<td>Comparison, trade-offs, conclusions bound to evidence</td>
<td><code>calculate</code> <code>retrieve_evidence</code></td>
<td>Search — so it cannot introduce untraceable facts.</td>
</tr>
<tr>
<td>🧾 <b>Fact-Checker</b></td>
<td>Citation integrity</td>
<td><code>validate_citations</code> <code>retrieve_evidence</code></td>
<td>Judge the argument. Only whether citations hold.</td>
</tr>
<tr>
<td>⚔️ <b>Critic</b></td>
<td>Six review criteria. Approve or reject with fixes.</td>
<td><code>validate_citations</code> <code>retrieve_evidence</code></td>
<td>Rewrite the answer. It evaluates; it doesn't author.</td>
</tr>
<tr>
<td>✍️ <b>Writer</b></td>
<td>The final report</td>
<td><code>export_report</code> <code>retrieve_evidence</code></td>
<td><b>Search.</b> It physically cannot invent research.</td>
</tr>
</table>

> **The sixth agent earns its place.** The Fact-Checker was chosen over other optional specialists
> because it directly produces a required metric — *unsupported major claims below 10%* — and
> because it splits cleanly into a deterministic half and a judgement half.

---

## 🛡 What makes it reliable

### Permission is a property of the call, not a line in a prompt

```python
run_tool("search_corpus", args, ctx, agent_id=AgentId.WRITER)
# ToolPermissionError: Agent 'writer' is not permitted to call 'search_corpus'.
```

The check happens **before** the tool function is reached, and before argument validation, so a
forbidden call fails *as forbidden* even when its arguments are also malformed. Every refused
call is logged with `outcome: permission_denied`.

**29 forbidden agent/tool pairings are enumerated and asserted in the test suite.** A matrix in a
README is a claim; [`tests/test_tools.py`](tests/test_tools.py) is the proof.

### Contracts that refuse bad input

Every handoff is a validated Pydantic object. These are model validators with tests proving each
one rejects:

| Rule | The failure it prevents |
|---|---|
| Empty research **must declare a gap** | A confident paragraph with no evidence reads downstream as substance |
| No evidence → **cannot claim high confidence** | The core hallucination pattern |
| A **major** conclusion **must cite** evidence | Makes "unsupported major claims" structurally impossible |
| A rejection **must name a problem and a fix** | "Not good enough" burns a revision cycle repeating the same mistake |
| An ambiguity flag **must carry questions** | Otherwise the human checkpoint is a dead end |
| Task plans are validated as a **DAG** | A dependency cycle would deadlock the graph *silently* |

### The quote must actually be in the source

`store_evidence` verifies, deterministically, that the supporting text appears in the document
being cited — whitespace-normalised, so reformatting is fine and invention is not.

```python
store_evidence(claim="LangGraph guarantees exactly-once delivery", ...)
# ToolError: supporting_text was not found in 'fw-langgraph-docs'.
```

Source reliability also **caps** confidence: vendor marketing cannot yield a high-confidence
fact however assertively it's written.

### Hard caps, already trip-tested

| Limit | Default | Stops |
|---|---|---|
| Revision cycles | `2` | Critic ↔ Analyst ping-pong |
| Research rounds | `2` | Endless re-planning |
| Model calls per run | `40` | Runaway loops |
| Wall clock | `600s` | Hung runs |
| Estimated spend | `$0.50` | Surprise bills |

### Cross-provider failover

Five OpenAI-compatible providers in one registry. Fallback spans **providers**, not just models —
a rate-limited Groq key continues on OpenRouter *mid-workflow*. Pull a key during a run and the
workflow completes, with the switch recorded in the trace.

---

## 📐 Measured decisions (no vibes)

Every engineering choice here was made from data in
[`scripts/probe_results.json`](scripts/probe_results.json), not from a docs page.

### Which model? Two probes, two different answers

The naive probe — *"can this model emit structured output?"* — passed **all five** candidates.
That test was too easy. The Supervisor needs a *nested* plan with dependencies, so:

<div align="center">

| Model | req/day | tok/min | Planning | Critic defects found |
|---|---:|---:|:---:|:---:|
| **`openai/gpt-oss-120b`** | 1,000 | 8,000 | **10/10** | **6 / 6** ✅ |
| **`llama-3.3-70b-versatile`** | 1,000 | 12,000 | 9/10 | 3 / 6 |
| `openai/gpt-oss-20b` | 1,000 | 8,000 | 5/10 | ❌ HTTP 400 |
| `qwen/qwen3.6-27b` | 1,000 | 8,000 | 5/10 | — |
| `llama-3.1-8b-instant` | 14,400 | 6,000 | 4/10 | ❌ HTTP 400 |

</div>

**Three models hard-fail on nested schemas.** The simple probe said five were usable; the
realistic one says two.

### The tiering, and why it's defensible

The critic probe plants **six defects** in an analysis and scores detection. What llama *missed*
decided the architecture:

| | Unsupported claim | Fabricated cite | Wrong cite | Contradiction | Overgeneralised | Missing criterion |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `gpt-oss-120b` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `llama-3.3-70b` | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |

llama caught **every citation defect** and **no reasoning defect**. Sound where the work is
mechanical, unsound where it's judgement. So:

- 🧠 **Judgement tier** → `gpt-oss-120b` — Supervisor, Analyst, **Critic**
- ⚡ **Throughput tier** → `llama-3.3-70b` — Researcher (×N parallel), Fact-Checker, Writer

<details>
<summary><b>A metric that nearly picked the wrong model</b></summary>

<br/>

The first version of the planning probe scored `gpt-oss-120b` at 6/8 because it returned **zero
tasks** on a vague request. But it had also correctly set `needs_clarification=True` — refusing
to plan until the objective is known is the *right* behaviour, and the scorer was punishing it.

Fixed scorer, waiving structural checks when a plan is correctly withheld → **10/10**, and the
model choice flipped.

This is the Week 3 lesson applied: *audit the metric before you trust it, especially a clean
one.* Several other metrics in this repo carry explicit "can this actually fail?" tests for the
same reason.

</details>

### Retrieval: BM25, deliberately

Twenty lines over 100 chunks. No embedding API, no ~2 GB torch dependency, no index build step.
Queries here are dense with distinctive proper nouns — `LangGraph`, `Fly.io`, `FedRAMP` — which
lexical search handles better than semantic similarity anyway.

Dropping torch also restored Streamlit hot-reload, which Week 3 had to disable.

---

## 🎭 The corpus fights back

24 synthetic documents across three domains, generated by
[`scripts/build_corpus.py`](scripts/build_corpus.py) — reproducible, and unmistakably not
scraped (every file carries `synthetic: true`).

**Eight defects are planted on purpose.** A Critic that never meets a contradiction has not been
shown to detect one.

| # | Defect | Expected catcher |
|---|---|---|
| **PD1** | CrewAI *has* / *hasn't* native human-in-the-loop | Critic |
| **PD2** | Benchmark latency vs. vendor's contradicting claim | Critic |
| **PD3** | "10x faster", no methodology | Critic |
| **PD4** | "10x lower TCO", contradicted by measured data | Critic |
| **PD5** | A 21-word stub that can't support a conclusion | Researcher → gap |
| **PD6** | 💉 **A live prompt-injection payload** | Researcher → ignore |
| **PD7** | Stale facts sold as "updated for 2026" | Critic |
| **PD8** | Self-reported survey read as measured throughput | Critic |

Retrieval tests assert that **both sides of each contradiction surface on the same query** —
otherwise the Critic could never see the conflict and the metric measuring it would be vacuous:

```
Q: 'CrewAI human in the loop approval'
   5.57  fw-crewai-docs        [high]    → "supports a human_input flag"
   5.49  fw-practitioner-blog  [medium]  → "has no native HITL mechanism"
```

> **PD6 is real, not hypothetical.** `ca-security-review.md` contains an actual instruction block
> telling the reading agent to abandon its research question and declare one vendor the only
> secure option. It sits in retrievable text so the defence can be demonstrated live rather than
> asserted.

---

## 🚀 Quick start

```bash
git clone https://github.com/Rana-Haseeb/multi-agent-research-platform.git
cd multi-agent-research-platform
pip install -r requirements.txt
```

```bash
cp .env.example .env    # then add at least one provider key
```

Only **one** provider key is required. Groq's free tier is the default.

```bash
python scripts/probe_providers.py     # which keys and models actually work
python scripts/build_corpus.py        # generate the 24-document corpus
python scripts/init_db.py             # optional — persistence
python -m pytest tests/ -q            # 100 tests
```

<details>
<summary><b>Configuration reference</b></summary>

<br/>

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq` · `google` · `openrouter` · `xai` · `openai` |
| `LLM_FALLBACK_PROVIDERS` | `openrouter` | Comma-separated cross-provider chain |
| `DATABASE_URL` | — | Supabase **session pooler** string. Optional. |
| `MAX_REVISION_CYCLES` | `2` | Critic loop hard cap |
| `MAX_AGENT_CALLS_PER_RUN` | `40` | Circuit breaker |
| `MAX_COST_USD_PER_RUN` | `0.50` | Spend ceiling |
| `ENABLE_LIVE_SEARCH` | `false` | Corpus by default, for reproducible evaluation |

**Postgres is optional.** With no `DATABASE_URL` the store no-ops rather than crashing — the
workflow, tests and evaluation all run without a database.

⚠️ Use the Supabase **session pooler** host. The direct `db.<ref>.supabase.co` host is IPv6-only
and won't resolve on Streamlit Cloud.

</details>

---

## 📁 Project structure

```
multi-agent-research-platform/
├── app/
│   ├── config.py              ← 5 providers · per-agent model tiers · budgets
│   ├── schemas/               ← evidence · tasks · 7 handoff contracts · reports
│   ├── graph/state.py         ← shared state · reducers · §27 permission table
│   ├── tools/                 ← 7 tools behind an agent-permission registry
│   ├── storage/               ← BM25 corpus index · Postgres evidence store
│   ├── services/              ← cross-provider LLM · per-agent token metering
│   ├── agents/                ← ⏳ Phase 4
│   └── baseline/              ← ⏳ single-agent control for Experiment 1
├── corpus/                    ← 24 documents + planted_defects.json
├── docs/                      ← generated from live code, never hand-written
├── scripts/
│   ├── probe_*.py             ← the measurements behind every model decision
│   ├── build_corpus.py        ← reproducible corpus generation
│   ├── gen_specs.py           ← docs ← code
│   └── verify_phase*.py       ← per-phase acceptance checks
└── tests/                     ← 100 tests, no network required
```

> **Docs are generated, not written.** `A2_state_specification.md`,
> `A3_handoff_contracts.md` and `A4_tool_specification.md` are built from
> `FIELD_PERMISSIONS` and the live Pydantic models. A spec that contradicts the code is worse
> than no spec — it misleads whoever trusts it.

---

## 📊 Build status

Every phase ends with an acceptance script that checks claims against the code, because *"done"
reported from memory is not done*.

| Phase | Scope | Checks | Status |
|---|---|:---:|:---:|
| **0** | Scaffold · 5 providers · metering · model selection | 42/42 | ✅ |
| **1** | Schemas · shared state · 7 handoff contracts | 22/22 | ✅ |
| **2** | 24-doc corpus · BM25 · Postgres evidence store | 26/26 | ✅ |
| **3** | 7 tools · permission boundaries · audit log | 23/23 | ✅ |
| 4 | Six agents + single-agent baseline | — | ⏳ |
| 5 | Orchestration graph (sequential) | — | ⏳ |
| 6 | Critic revision loop | — | ⏳ |
| 7 | Parallel fan-out + measured speedup | — | ⏳ |
| 8 | Human checkpoints · dashboard · export | — | ⏳ |
| 9 | Automated tests | — | ⏳ |
| 10 | 25 eval cases · 5 experiments · 10 adversarial | — | ⏳ |
| 11 | Docs · security review · deploy | — | ⏳ |

```bash
python scripts/verify_phase0.py && python scripts/verify_phase1.py \
  && python scripts/verify_phase2.py && python scripts/verify_phase3.py
```

**Current: 113/113 acceptance checks · 100 tests passing.**

---

## ⚠️ Known limitations

Stated plainly, because a limitation you disclose is engineering and one you hide is a defect.

- **The corpus is synthetic.** 24 documents written for this project. Retrieval quality on real
  web sources is unmeasured. Live search exists behind `ENABLE_LIVE_SEARCH` but isn't the
  evaluated path.
- **No end-to-end workflow numbers yet.** Latency, cost and quality figures arrive in Phase 10
  from actual runs. This README will not carry an estimated metric.
- **BM25, not semantic search.** Adequate for 24 keyword-dense documents; a paraphrased query
  with no shared vocabulary will retrieve poorly.
- **Free-tier rate limits.** 1,000 requests/day per model. Sufficient for ~480 planned calls,
  but not for sustained load.
- **Single-language corpus.** English only.
- **Cost tracking reads $0.00 on Groq's free tier.** Token counts are real and measured; USD
  figures only become meaningful on a paid provider.

---

<div align="center">
<br/>

**Built by [Rana Muhammad Haseeb Khan](https://github.com/Rana-Haseeb)**
<br/>
<sub>FAST NUCES · Software Engineering · Visibility Bots AI Fellowship 2026</sub>

<br/>

<sub><i>More agents is not better. Every agent must earn its latency and its cost.</i></sub>

</div>
