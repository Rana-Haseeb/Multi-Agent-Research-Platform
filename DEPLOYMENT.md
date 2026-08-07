# Deployment (§35)

Two supported targets: **local** (full capability, including experiments and evaluation) and
**Streamlit Community Cloud** (the interactive app). The app runs without a database — evidence then
lives in memory for the session only — so a missing `DATABASE_URL` degrades a feature rather than
breaking startup.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | 3.11 or 3.12 recommended |
| At least one LLM provider key | The app fails fast with a readable message if none is set |
| Postgres (optional) | Supabase free tier is sufficient; without it, evidence is not persisted |

No GPU, no `torch`, no embedding service is required. Retrieval defaults to BM25, which is pure
Python and needs no key.

---

## 2. Local setup

```bash
python -m venv .venv && source .venv/bin/activate
```

On Windows PowerShell the activation line is `.venv\Scripts\Activate.ps1`.

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Fill in `.env` — at minimum one provider key. **Never commit it**; `.gitignore` blocks `.env` and
allows only `.env.example`.

Confirm which keys and model ids actually work. This prints capability, never key material:

```bash
python scripts/probe_providers.py
```

Build the retrieval index from the corpus:

```bash
python scripts/build_corpus.py
```

Create the database tables (skip if running without persistence):

```bash
python scripts/init_db.py
```

Run the app:

```bash
streamlit run app/main.py
```

---

## 3. Configuration that matters

**Provider chain.** `LLM_PROVIDER` is tried first, then each entry of
`LLM_FALLBACK_PROVIDERS` in order. Set the fallback list to `""` to fail fast instead.

**Quota is per organisation, not per key.** `GROQ_API_KEY_2` and `_3` are only useful if they belong
to *different* accounts. Three keys on one account share one daily allowance.

**The binding limit is tokens per day, not per minute.** It does not appear in response headers —
only in the body of a 429. Budget for it before running the evaluation or experiments; a full
workflow costs roughly 100k tokens.

**Workflow caps** (`MAX_REVISION_CYCLES`, `MAX_AGENT_CALLS_PER_RUN`, `MAX_RUN_SECONDS`, …) are
enforced in code and are the cost circuit-breaker. Raising them raises your bill and your exposure to
runaway loops; `MAX_REVISION_CYCLES` above 2 also puts the workflow outside the §18 requirement.

---

## 4. Streamlit Community Cloud

1. Push the repository to GitHub. Verify `.env` is **not** in the tree:

```bash
git ls-files | grep -i env
```

Only `.env.example` may appear. If `.env` is listed, stop and remove it from history before pushing.

2. On [share.streamlit.io](https://share.streamlit.io), create an app pointing at
   `app/main.py` on the `main` branch.

3. Add secrets under **App settings → Secrets**, in TOML form — not as a committed file. Add
   every key you have; each one widens the daily allowance and adds a failover hop:

```toml
GROQ_API_KEY = "paste-key-1"
GROQ_API_KEY_2 = "paste-key-2"
GROQ_API_KEY_3 = "paste-key-3"
GOOGLE_API_KEY = "paste-key"
OPENROUTER_API_KEY = "paste-key"
```

That is the whole configuration. Nothing else is required: `LLM_PROVIDER` already defaults to the
primary backend, and `LLM_FALLBACK_PROVIDERS` defaults to every other registered backend, with
unconfigured ones filtered out by `provider_chain()`. **Adding a key is sufficient to use it.**

`app/main.py` copies Streamlit secrets into the process environment *before* `app/config.py`
loads, so the same `os.getenv` path serves local and cloud alike — no code changes between them.
Every variable in `.env.example` can be set here using the same names.

### Do not deploy with a single key

A one-key deployment was tried and failed mid-run: **216 calls attempted, 164 refused**, the
budget tripped at 52/50 and the Analyst never produced a result. The free tier's ceiling is tokens
*per day* and a full workflow costs roughly 100k tokens, so one account cannot finish a run
reliably once the day's allowance is partly spent.

Quota is per **organisation**, not per key — extra keys only help if they are on different
accounts. To deliberately disable failover and fail fast instead, set
`LLM_FALLBACK_PROVIDERS = ""` explicitly.

4. Deploy. First boot builds the BM25 index from `corpus/`, which is committed, so no build step is
   required.

### Cloud-specific gotchas

**Use the Session Pooler connection string.** The direct `db.<ref>.supabase.co` host is IPv6-only and
will not resolve on Streamlit Cloud — the symptom is `getaddrinfo failed`. The pooler host
(`aws-1-<region>.pooler.supabase.com`) is IPv4-compatible.

**URL-encode the password.** An `@` in the password must be written `%40`, or the connection string
parses with the wrong host.

**Checkpointer state is per-process.** `MemorySaver` holds interrupted runs in memory, so a cloud
restart drops any run paused at a human checkpoint. This is acceptable for a demo and is the first
thing to replace with a persistent checkpointer for real use — see [ROADMAP.md](docs/ROADMAP.md).

---

## 5. Verifying a deployment

Run the phase verifiers — they check claims against the actual repo rather than against a checklist:

```bash
python scripts/verify_phase11.py
```

Run the test suite:

```bash
python -m pytest tests/ -q
```

Neither needs an API key: the suite has an autouse guard that fails any test attempting an unmocked
model call, so a green run is genuinely offline.

In the app itself, a successful deployment shows: the six agents in the sidebar status panel, a run
that reaches a research-plan checkpoint, and a final report with an evidence section separate from
the recommendation.

---

## 6. Credential hygiene

- Keys live in `.env` locally and in Streamlit Secrets in the cloud. Nowhere else — in particular,
  never in a markdown file, a notebook output, or a commit message.
- `scripts/probe_providers.py` reports which providers work without printing key material; use it
  instead of echoing variables when debugging.
- **Rotate any key that has ever been pasted into a chat, an issue, or a screenshot.** Deleting the
  message does not un-expose it.
