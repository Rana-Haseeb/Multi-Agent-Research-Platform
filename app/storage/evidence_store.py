"""
Postgres persistence for evidence, runs and execution traces (§15, §23).

Patterns carried over from the Week 3 data layer because each was earned:

- **A connection per call, never a long-lived one.** Streamlit reruns the script top to bottom on
  every interaction; a module-level connection goes stale and fails in ways that look like data
  loss. Opening per call is slower and correct.
- **Bound parameters everywhere.** No f-string SQL, including for values that "obviously" come
  from our own code — corpus text reaches these tables and corpus text is untrusted input.
- **Typed errors.** Callers distinguish "not configured" from "query failed" so the app can run
  in memory-only mode without the database rather than crashing.

Tables are prefixed ``w4_`` so this schema coexists with the Week 3 tables in the same project.

Persistence is **optional by design**. Every method degrades to a no-op when ``DATABASE_URL`` is
unset, so the full workflow, the test suite and the evaluation all run without a database. The
store is durability, not a dependency.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import settings
from app.schemas.common import AgentId, ClaimType, Confidence
from app.schemas.evidence import Evidence
from app.schemas.reports import ErrorRecord, TraceEvent


class StorageError(RuntimeError):
    """A database operation failed for a user-safe, reportable reason."""


class StorageNotConfigured(StorageError):
    """No DATABASE_URL. Callers may treat this as "run without persistence"."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS w4_runs (
    run_id          TEXT PRIMARY KEY,
    user_request    TEXT        NOT NULL,
    status          TEXT        NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    agent_calls     INTEGER     NOT NULL DEFAULT 0,
    input_tokens    INTEGER     NOT NULL DEFAULT 0,
    output_tokens   INTEGER     NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12, 6) NOT NULL DEFAULT 0,
    revision_count  INTEGER     NOT NULL DEFAULT 0,
    wall_seconds    NUMERIC(10, 2) NOT NULL DEFAULT 0,
    abort_reason    TEXT        NOT NULL DEFAULT ''
);

-- evidence_id (E101) is unique only within a run, so the key is composite.
CREATE TABLE IF NOT EXISTS w4_evidence (
    run_id            TEXT NOT NULL REFERENCES w4_runs(run_id) ON DELETE CASCADE,
    evidence_id       TEXT NOT NULL,
    claim             TEXT NOT NULL,
    supporting_text   TEXT NOT NULL,
    source_id         TEXT NOT NULL,
    source_title      TEXT NOT NULL,
    retrieved_at      TIMESTAMPTZ NOT NULL,
    research_question TEXT NOT NULL,
    confidence        TEXT NOT NULL,
    claim_type        TEXT NOT NULL,
    agent_id          TEXT NOT NULL,
    task_id           TEXT NOT NULL,
    PRIMARY KEY (run_id, evidence_id)
);
CREATE INDEX IF NOT EXISTS w4_evidence_question_idx
    ON w4_evidence (run_id, research_question);

CREATE TABLE IF NOT EXISTS w4_trace (
    id               BIGSERIAL PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES w4_runs(run_id) ON DELETE CASCADE,
    at               TIMESTAMPTZ NOT NULL,
    agent_id         TEXT NOT NULL,
    event            TEXT NOT NULL,
    node             TEXT NOT NULL DEFAULT '',
    task_id          TEXT NOT NULL DEFAULT '',
    detail           TEXT NOT NULL DEFAULT '',
    duration_seconds NUMERIC(10, 3) NOT NULL DEFAULT 0,
    provider         TEXT NOT NULL DEFAULT '',
    model            TEXT NOT NULL DEFAULT '',
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS w4_trace_run_idx ON w4_trace (run_id, at);

CREATE TABLE IF NOT EXISTS w4_errors (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES w4_runs(run_id) ON DELETE CASCADE,
    at           TIMESTAMPTZ NOT NULL,
    agent_id     TEXT NOT NULL,
    node         TEXT NOT NULL DEFAULT '',
    task_id      TEXT NOT NULL DEFAULT '',
    kind         TEXT NOT NULL,
    message      TEXT NOT NULL,
    recovered    BOOLEAN NOT NULL DEFAULT FALSE,
    action_taken TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS w4_reports (
    run_id       TEXT PRIMARY KEY REFERENCES w4_runs(run_id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    markdown     TEXT NOT NULL,
    payload      JSONB,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

TABLES = ("w4_runs", "w4_evidence", "w4_trace", "w4_errors", "w4_reports")


class EvidenceStore:
    """Durable store for one workflow's evidence and trace.

    ``enabled`` is False when no DATABASE_URL is set; every method then no-ops so callers need no
    conditional logic. ``EvidenceStore().enabled`` is the only check the app makes.
    """

    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or (
            settings.database_url if settings.db_configured() else None
        )

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    # ------------------------------------------------------------ connection
    @contextmanager
    def _conn(self) -> Iterator[Any]:
        if not self.enabled:
            raise StorageNotConfigured("DATABASE_URL is not configured.")
        try:
            import psycopg
        except ImportError as e:  # pragma: no cover
            raise StorageError("psycopg is not installed.") from e
        try:
            with psycopg.connect(self.database_url, connect_timeout=20) as conn:
                yield conn
        except StorageError:
            raise
        except Exception as e:  # noqa: BLE001
            raise StorageError(f"Database operation failed: {type(e).__name__}") from e

    # --------------------------------------------------------------- schema
    def init_schema(self) -> list[str]:
        """Create the tables. Idempotent. Returns the table names that now exist."""
        with self._conn() as conn:
            conn.execute(SCHEMA)
            conn.commit()
            rows = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = ANY(%s) ORDER BY table_name",
                (list(TABLES),),
            ).fetchall()
        return [r[0] for r in rows]

    # ----------------------------------------------------------------- runs
    def start_run(self, run_id: str, user_request: str, status: str = "pending") -> None:
        if not self.enabled:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO w4_runs (run_id, user_request, status) VALUES (%s, %s, %s) "
                "ON CONFLICT (run_id) DO UPDATE SET user_request = EXCLUDED.user_request",
                (run_id, user_request, status),
            )
            conn.commit()

    def finish_run(
        self, run_id: str, *, status: str, agent_calls: int = 0, input_tokens: int = 0,
        output_tokens: int = 0, cost_usd: float = 0.0, revision_count: int = 0,
        wall_seconds: float = 0.0, abort_reason: str = "",
    ) -> None:
        if not self.enabled:
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE w4_runs SET status = %s, finished_at = now(), agent_calls = %s, "
                "input_tokens = %s, output_tokens = %s, cost_usd = %s, revision_count = %s, "
                "wall_seconds = %s, abort_reason = %s WHERE run_id = %s",
                (status, agent_calls, input_tokens, output_tokens, cost_usd,
                 revision_count, wall_seconds, abort_reason, run_id),
            )
            conn.commit()

    # ------------------------------------------------------------- evidence
    def save_evidence(self, run_id: str, items: list[Evidence]) -> int:
        """Persist evidence. Re-saving the same id is a no-op, so retries are safe."""
        if not self.enabled or not items:
            return 0
        rows = [
            (run_id, e.evidence_id, e.claim, e.supporting_text, e.source_id, e.source_title,
             e.retrieved_at, e.research_question, e.confidence.value, e.claim_type.value,
             e.agent_id, e.task_id)
            for e in items
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO w4_evidence (run_id, evidence_id, claim, supporting_text, "
                    "source_id, source_title, retrieved_at, research_question, confidence, "
                    "claim_type, agent_id, task_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (run_id, evidence_id) DO NOTHING",
                    rows,
                )
            conn.commit()
        return len(rows)

    def load_evidence(self, run_id: str, research_question: str | None = None) -> list[Evidence]:
        if not self.enabled:
            return []
        sql = (
            "SELECT evidence_id, claim, supporting_text, source_id, source_title, retrieved_at, "
            "research_question, confidence, claim_type, agent_id, task_id "
            "FROM w4_evidence WHERE run_id = %s"
        )
        params: list[Any] = [run_id]
        if research_question:
            sql += " AND research_question = %s"
            params.append(research_question)
        sql += " ORDER BY evidence_id"
        with self._conn() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [
            Evidence(
                evidence_id=r[0], claim=r[1], supporting_text=r[2], source_id=r[3],
                source_title=r[4], retrieved_at=r[5], research_question=r[6],
                confidence=Confidence(r[7]), claim_type=ClaimType(r[8]),
                agent_id=r[9], task_id=r[10],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- trace
    def save_trace(self, run_id: str, events: list[TraceEvent]) -> int:
        if not self.enabled or not events:
            return 0
        rows = [
            (run_id, e.at, e.agent_id.value if isinstance(e.agent_id, AgentId) else str(e.agent_id),
             e.event, e.node, e.task_id, e.detail, e.duration_seconds, e.provider, e.model,
             e.input_tokens, e.output_tokens)
            for e in events
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO w4_trace (run_id, at, agent_id, event, node, task_id, detail, "
                    "duration_seconds, provider, model, input_tokens, output_tokens) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    rows,
                )
            conn.commit()
        return len(rows)

    def save_errors(self, run_id: str, errors: list[ErrorRecord]) -> int:
        if not self.enabled or not errors:
            return 0
        rows = [
            (run_id, e.at, e.agent_id.value if isinstance(e.agent_id, AgentId) else str(e.agent_id),
             e.node, e.task_id, e.kind, e.message, e.recovered, e.action_taken)
            for e in errors
        ]
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO w4_errors (run_id, at, agent_id, node, task_id, kind, message, "
                    "recovered, action_taken) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    rows,
                )
            conn.commit()
        return len(rows)

    # -------------------------------------------------------------- reports
    def save_report(self, run_id: str, title: str, markdown: str, payload: dict | None = None) -> None:
        if not self.enabled:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO w4_reports (run_id, title, markdown, payload) VALUES (%s,%s,%s,%s) "
                "ON CONFLICT (run_id) DO UPDATE SET title = EXCLUDED.title, "
                "markdown = EXCLUDED.markdown, payload = EXCLUDED.payload, generated_at = now()",
                (run_id, title, markdown, json.dumps(payload or {})),
            )
            conn.commit()

    def load_report(self, run_id: str) -> str | None:
        if not self.enabled:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT markdown FROM w4_reports WHERE run_id = %s", (run_id,)
            ).fetchone()
        return row[0] if row else None

    def recent_runs(self, limit: int = 20) -> list[dict]:
        if not self.enabled:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id, user_request, status, started_at, agent_calls, cost_usd, "
                "wall_seconds FROM w4_runs ORDER BY started_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [
            {"run_id": r[0], "user_request": r[1], "status": r[2], "started_at": r[3],
             "agent_calls": r[4], "cost_usd": float(r[5]), "wall_seconds": float(r[6])}
            for r in rows
        ]
