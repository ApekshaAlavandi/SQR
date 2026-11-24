"""Local schema vs the real upstream schema (``schema_new.sql``).

Checks that every column the real DB has on the tables scout-ai / the B screens touch also
exists in the local DDL (``schema.py`` + ``schema_ext.py``). Column NAMES only — types and
constraints are copied by hand; this catches the usual drift: upstream adds a column and the
local schema silently lacks it.

Provide the real dump with  SCOUT_REAL_SCHEMA_SQL=/path/to/schema_new.sql
(or drop it at tests/fixtures/schema_new.sql). Without it the parity test is skipped.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

import pytest

from scout_ai.transport.local import bootstrap
from scout_ai.transport.local.schema import SCHEMA_STATEMENTS
from scout_ai.transport.local.schema_ext import EXTENSION_STATEMENTS

# Tables the flows write, or the B screens read. Others (users, clubs, ...) are upstream-only.
CHECKED_TABLES = (
    "clips",
    "processing_jobs",
    "video_assets",
    "clip_reviews",
    "clip_segments",
    "clip_analyses",
    "clip_presence_spans",
    "clip_observations",
    "clip_tallies",
)

_COL = re.compile(r"^\s{4}([a-z_][a-z0-9_]*)\s+\S", re.M)


def _real_schema_path() -> Path | None:
    candidates = [os.environ.get("SCOUT_REAL_SCHEMA_SQL"), "tests/fixtures/schema_new.sql"]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def _real_columns(sql: str, table: str) -> set[str]:
    m = re.search(rf"CREATE TABLE public\.{table} \((.*?)\n\);", sql, re.S)
    assert m, f"{table} not found in the real schema"
    cols = set()
    for line in m.group(1).splitlines():
        if line.startswith("    CONSTRAINT") or not line.startswith("    "):
            continue
        cols.add(line.split()[0])
    return cols


def _local_columns(table: str) -> set[str]:
    cols: set[str] = set()
    for stmt in (*SCHEMA_STATEMENTS, *EXTENSION_STATEMENTS):
        create = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*)\)\s*$", stmt, re.S)
        if create:
            for line in create.group(1).splitlines():
                tok = line.strip().split(" ")[0]
                if line.startswith("        ") and not line.startswith("         ") and re.fullmatch(
                    r"[a-z_][a-z0-9_]*", tok
                ) and tok not in {"CONSTRAINT", "PRIMARY", "UNIQUE", "FOREIGN", "CHECK"}:
                    cols.add(tok)
        if re.match(rf"\s*ALTER TABLE {table}\b", stmt):
            cols.update(re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", stmt))
    return cols


@pytest.mark.skipif(_real_schema_path() is None, reason="set SCOUT_REAL_SCHEMA_SQL to schema_new.sql")
@pytest.mark.parametrize("table", CHECKED_TABLES)
def test_local_schema_has_every_real_column(table):
    real = _real_columns(_real_schema_path().read_text(), table)
    local = _local_columns(table)
    assert real - local == set(), f"{table}: local schema lacks columns {sorted(real - local)}"


def test_seed_uses_a_real_clip_status():
    """The real enum has no 'ready' (it is 'completed'); a regression here breaks seeding."""
    assert "'ready'" not in inspect.getsource(bootstrap)
    ddl = " ".join(EXTENSION_STATEMENTS)
    assert "'completed'" in ddl and "CREATE TYPE clip_status" in ddl


def test_extension_is_idempotent_by_construction():
    """Every extension statement must be safe to re-run (init-db runs on every command)."""
    for stmt in EXTENSION_STATEMENTS:
        s = " ".join(stmt.split())
        if s.startswith("CREATE TABLE"):
            assert "IF NOT EXISTS" in s, s[:80]
        elif s.startswith("CREATE INDEX"):
            assert "IF NOT EXISTS" in s, s[:80]
        elif s.startswith("ALTER TABLE"):
            assert "ADD COLUMN IF NOT EXISTS" in s, s[:80]
        # DO $$ ... $$ blocks and the backfill UPDATE are guarded in their own text


# --- end to end against a real local Postgres (opt-in, same switch as test_local_mode) ----
@pytest.mark.skipif(
    not os.environ.get("SCOUT_TEST_DATABASE_URL"),
    reason="set SCOUT_TEST_DATABASE_URL to a LOCAL Postgres to run the end-to-end checks",
)
def test_demo_fills_extended_tables(monkeypatch):
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    from scout_ai.config import get_settings
    from scout_ai.transport.local import cli

    url = os.environ["SCOUT_TEST_DATABASE_URL"]
    monkeypatch.setenv("SCOUT_DATABASE_URL", url)
    monkeypatch.setenv("SCOUT_AI_BACKEND", "stub")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["demo"])
        assert exc.value.code == 0
    finally:
        get_settings.cache_clear()

    async def scalar(sql: str) -> int:
        engine = create_async_engine(url)
        try:
            async with engine.connect() as conn:
                return (await conn.execute(text(sql))).scalar_one()
        finally:
            await engine.dispose()

    # every draft ran under a job that is linked to its analysis and carries both versions
    assert asyncio.run(scalar(
        "SELECT count(*) FROM processing_jobs j JOIN clip_analyses a ON a.job_id = j.id "
        "WHERE j.status = 'succeeded' AND j.job_type = 'ai_analysis' "
        "AND j.model_version IS NOT NULL AND j.taxonomy_version IS NOT NULL")) >= 2
    # approve/seed keep review_source consistent with review_status (real CHECK constraint)
    assert asyncio.run(scalar(
        "SELECT count(*) FROM clip_reviews WHERE review_status <> 'pending' "
        "AND review_source IS NULL")) == 0
    # every local clip carries the new upstream columns
    assert asyncio.run(scalar(
        "SELECT count(*) FROM clips WHERE created_by IS NULL OR position_played IS NULL")) == 0
