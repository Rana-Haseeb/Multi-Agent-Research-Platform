"""
Create the Week 4 database schema. Idempotent — safe to re-run.

    python scripts/init_db.py
    python scripts/init_db.py --check     # report only, create nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.storage.evidence_store import TABLES, EvidenceStore, StorageError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report existing tables only")
    args = ap.parse_args()

    store = EvidenceStore()
    if not store.enabled:
        print("DATABASE_URL is not configured — nothing to do.")
        print("The workflow still runs without it; persistence is simply disabled.")
        return 1

    # Never print the URL: it carries the password.
    host = settings.database_url.split("@")[-1].split("/")[0] if settings.database_url else "?"
    print(f"host: {host}")

    try:
        if args.check:
            with store._conn() as conn:
                rows = conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = ANY(%s) ORDER BY table_name",
                    (list(TABLES),),
                ).fetchall()
            present = [r[0] for r in rows]
        else:
            present = store.init_schema()
    except StorageError as e:
        print(f"FAILED: {e}")
        return 1

    for table in TABLES:
        print(f"  {'ok  ' if table in present else 'MISSING'} {table}")

    missing = [t for t in TABLES if t not in present]
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        return 1
    print(f"\nAll {len(TABLES)} tables present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
