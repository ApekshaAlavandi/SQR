"""Local-only schema extension: brings the local Postgres in line with ``schema_new.sql``.

``schema.py`` is the original minimal approximation. Upstream has since extended ``clips``
and ``processing_jobs`` (and others); this module adds those on top WITHOUT editing
``schema.py``, so the original local schema stays as it was and everything here is
additive and idempotent (safe to run on every ``init-db`` / ``seed`` / ``demo``).

What it adds (all copied from ``schema_new.sql``; the CI-friendly parity test in
``tests/test_local_schema_parity.py`` checks the column names against that file):

* enums ``clip_status``, ``processing_job_status``, ``processing_job_type``, ``review_source``
* ``clips``: timeline_order, upload_expires_at, created_by, created_at, updated_at,
  language, position_played, half, footage_class, competition, venue, their CHECKs and
  indexes; ``upload_status`` becomes the real ``clip_status`` enum (old local ``ready``
  rows are mapped to ``completed``)
* ``processing_jobs`` (new table, incl. chunk_plan / chunks_done / model_version /
  taxonomy_version) and the ``clip_analyses.job_id`` foreign key to it
* ``video_assets``: the extra columns (club_id, original_filename, content_type, ...)
* ``clip_reviews.review_source`` + the real ``chk_clip_reviews_source_status`` CHECK
* ``clip_segments`` (table only; scout-ai does not write it, see PRODUCTION_CHANGES.md)

Deliberate local differences (upstream-owned tables we do not recreate): no FKs to
``users`` / ``clubs`` / ``players`` / ``club_sports``, so ``created_by`` is a plain
nullable uuid, and the columns upstream marks NOT NULL on ``video_assets`` are nullable
here so rows from older local runs keep working (they are backfilled below).
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

# Stand-in for the upstream user who "uploaded" local clips (clips.created_by).
LOCAL_USER_ID = UUID("00000000-0000-4000-8000-0000000000d1")


def _guarded(sql: str) -> str:
    """Make a CREATE TYPE / ADD CONSTRAINT idempotent (Postgres has no IF NOT EXISTS for them)."""
    return (
        f"DO $$ BEGIN {sql}; "
        "EXCEPTION WHEN duplicate_object OR duplicate_table THEN NULL; END $$"
    )


EXTENSION_STATEMENTS: tuple[str, ...] = (
    # ---- enum types (labels copied from schema_new.sql) -------------------------------
    _guarded(
        "CREATE TYPE clip_status AS ENUM ('pending_upload', 'video_processing', "
        "'ai_analyzing', 'completed', 'failed', 'player_tracking')"
    ),
    _guarded(
        "CREATE TYPE processing_job_status AS ENUM "
        "('queued', 'running', 'succeeded', 'failed', 'cancelled')"
    ),
    _guarded(
        "CREATE TYPE processing_job_type AS ENUM ('video_transcode', 'ai_draft', "
        "'ai_regenerate', 'kpi_compute', 'player_tracking', 'ai_analysis')"
    ),
    _guarded("CREATE TYPE review_source AS ENUM ('ai', 'human')"),
    # ---- clips ------------------------------------------------------------------------
    """
    ALTER TABLE clips
        ADD COLUMN IF NOT EXISTS timeline_order    integer     NOT NULL DEFAULT 0,
        ADD COLUMN IF NOT EXISTS upload_expires_at timestamptz,
        ADD COLUMN IF NOT EXISTS created_by        uuid,
        ADD COLUMN IF NOT EXISTS created_at        timestamptz NOT NULL DEFAULT now(),
        ADD COLUMN IF NOT EXISTS updated_at        timestamptz NOT NULL DEFAULT now(),
        ADD COLUMN IF NOT EXISTS language          text        NOT NULL DEFAULT 'en',
        ADD COLUMN IF NOT EXISTS position_played   text,
        ADD COLUMN IF NOT EXISTS half              smallint,
        ADD COLUMN IF NOT EXISTS footage_class     text        NOT NULL DEFAULT 'short',
        ADD COLUMN IF NOT EXISTS competition       text,
        ADD COLUMN IF NOT EXISTS venue             text
    """,
    _guarded(
        "ALTER TABLE clips ADD CONSTRAINT chk_clips_footage_class "
        "CHECK (footage_class = ANY (ARRAY['short', 'half', 'full']))"
    ),
    _guarded(
        "ALTER TABLE clips ADD CONSTRAINT chk_clips_half "
        "CHECK (half IS NULL OR half = ANY (ARRAY[1, 2]))"
    ),
    _guarded(
        "ALTER TABLE clips ADD CONSTRAINT chk_clips_language "
        "CHECK (language = ANY (ARRAY['ar', 'en']))"
    ),
    _guarded(
        "ALTER TABLE clips ADD CONSTRAINT chk_clips_minute "
        "CHECK (minute IS NULL OR (minute >= 0 AND minute <= 200))"
    ),
    # upload_status: text ('ready' ...) -> the real clip_status enum. Only runs while the
    # column is still text, so it is a no-op on a fresh/converted database.
    """
    DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM information_schema.columns
                    WHERE table_schema = current_schema() AND table_name = 'clips'
                      AND column_name = 'upload_status' AND data_type = 'text') THEN
            ALTER TABLE clips ALTER COLUMN upload_status DROP DEFAULT;
            ALTER TABLE clips ALTER COLUMN upload_status TYPE clip_status
                USING (CASE upload_status
                           WHEN 'ready'     THEN 'completed'
                           WHEN 'uploading' THEN 'pending_upload'
                           WHEN 'processing' THEN 'video_processing'
                           ELSE upload_status
                       END)::clip_status;
            ALTER TABLE clips ALTER COLUMN upload_status SET DEFAULT 'pending_upload';
        END IF;
    END $$
    """,
    "CREATE INDEX IF NOT EXISTS idx_clips_player_timeline "
    "ON clips (club_id, player_id, match_date DESC, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_clips_upload_status ON clips (club_id, upload_status) "
    "WHERE upload_status = ANY (ARRAY['pending_upload'::clip_status, "
    "'video_processing'::clip_status, 'ai_analyzing'::clip_status])",
    # ---- video_assets (upstream NOT NULLs relaxed locally; see module docstring) ------
    """
    ALTER TABLE video_assets
        ADD COLUMN IF NOT EXISTS club_id              uuid,
        ADD COLUMN IF NOT EXISTS original_filename    text,
        ADD COLUMN IF NOT EXISTS content_type         text,
        ADD COLUMN IF NOT EXISTS raw_object_key       text,
        ADD COLUMN IF NOT EXISTS raw_size_bytes       bigint,
        ADD COLUMN IF NOT EXISTS raw_uploaded_at      timestamptz,
        ADD COLUMN IF NOT EXISTS processed_size_bytes bigint,
        ADD COLUMN IF NOT EXISTS duration_seconds     numeric(8,2),
        ADD COLUMN IF NOT EXISTS width                integer,
        ADD COLUMN IF NOT EXISTS height               integer,
        ADD COLUMN IF NOT EXISTS processing_error     text,
        ADD COLUMN IF NOT EXISTS processed_at         timestamptz,
        ADD COLUMN IF NOT EXISTS created_at           timestamptz NOT NULL DEFAULT now(),
        ADD COLUMN IF NOT EXISTS updated_at           timestamptz NOT NULL DEFAULT now(),
        ADD COLUMN IF NOT EXISTS tracking_seed        jsonb
    """,
    # ---- clip_reviews: review_source + the real source/status CHECK -------------------
    "ALTER TABLE clip_reviews ADD COLUMN IF NOT EXISTS review_source review_source",
    # Rows written before this column existed must satisfy the CHECK below.
    """
    UPDATE clip_reviews
       SET review_source = CASE review_status
               WHEN 'accepted_ai_draft' THEN 'ai'::review_source
               WHEN 'edited_by_scout'   THEN 'human'::review_source
           END
     WHERE review_source IS NULL AND review_status <> 'pending'
    """,
    _guarded(
        "ALTER TABLE clip_reviews ADD CONSTRAINT chk_clip_reviews_source_status CHECK ("
        "(review_status = 'pending' AND review_source IS NULL) OR "
        "(review_status = 'accepted_ai_draft' AND review_source = 'ai') OR "
        "(review_status = 'edited_by_scout' AND review_source = 'human'))"
    ),
    # ---- processing_jobs (upstream-owned; recreated minus the clubs FK) ---------------
    """
    CREATE TABLE IF NOT EXISTS processing_jobs (
        id               uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
        club_id          uuid NOT NULL,
        clip_id          uuid,
        job_type         processing_job_type   NOT NULL,
        status           processing_job_status NOT NULL DEFAULT 'queued',
        idempotency_key  text NOT NULL,
        payload          jsonb NOT NULL DEFAULT '{}',
        attempt_count    smallint NOT NULL DEFAULT 0,
        max_attempts     smallint NOT NULL DEFAULT 5,
        last_error       text,
        queued_at        timestamptz NOT NULL DEFAULT now(),
        started_at       timestamptz,
        finished_at      timestamptz,
        created_at       timestamptz NOT NULL DEFAULT now(),
        chunk_plan       jsonb,
        chunks_done      smallint NOT NULL DEFAULT 0,
        model_version    text,
        taxonomy_version text,
        CONSTRAINT processing_jobs_idempotency_key_key UNIQUE (idempotency_key),
        CONSTRAINT uq_processing_jobs_id_club UNIQUE (id, club_id),
        CONSTRAINT fk_processing_jobs_clip FOREIGN KEY (clip_id, club_id)
            REFERENCES clips (id, club_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_processing_jobs_status ON processing_jobs (status, queued_at) "
    "WHERE status = ANY (ARRAY['queued'::processing_job_status, 'running'::processing_job_status])",
    # clip_analyses.job_id -> processing_jobs (schema.py left this FK out on purpose)
    _guarded(
        "ALTER TABLE clip_analyses ADD CONSTRAINT fk_clip_analyses_job "
        "FOREIGN KEY (job_id, club_id) REFERENCES processing_jobs (id, club_id)"
    ),
    "CREATE INDEX IF NOT EXISTS idx_clip_analyses_job ON clip_analyses (job_id)",
    # ---- clip_segments (table only; not written by scout-ai) --------------------------
    """
    CREATE TABLE IF NOT EXISTS clip_segments (
        id             uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
        club_id        uuid NOT NULL,
        analysis_id    uuid NOT NULL,
        observation_id uuid NOT NULL,
        object_key     text NOT NULL,
        t0             numeric(8,2) NOT NULL,
        t1             numeric(8,2) NOT NULL,
        caption_en     text,
        caption_ar     text,
        hidden         boolean DEFAULT false NOT NULL,
        created_at     timestamptz DEFAULT now() NOT NULL,
        CONSTRAINT chk_segments_time CHECK (t0 >= 0 AND t0 < t1),
        CONSTRAINT uq_clip_segments_observation UNIQUE (observation_id),
        CONSTRAINT fk_segments_analysis FOREIGN KEY (analysis_id, club_id)
            REFERENCES clip_analyses (id, club_id) ON DELETE CASCADE,
        CONSTRAINT fk_segments_observation FOREIGN KEY (observation_id, analysis_id)
            REFERENCES clip_observations (id, analysis_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clip_segments_analysis ON clip_segments (analysis_id)",
)


async def backfill_extended(conn: AsyncConnection) -> None:
    """Fill the new NOT-NULL-upstream columns on rows that predate them (idempotent).

    Called at the end of ``init_db``, ``seed`` and ``create_clip`` so every local clip
    looks like a real upstream row. Only NULLs are touched (COALESCE), so values that
    were set explicitly are never overwritten.
    """
    await conn.execute(
        text("UPDATE clips SET created_by = :u WHERE created_by IS NULL"), {"u": LOCAL_USER_ID}
    )
    await conn.execute(
        text(
            "UPDATE clips c SET position_played = p.position_key "
            "FROM players p WHERE p.id = c.player_id AND c.position_played IS NULL"
        )
    )
    await conn.execute(
        text(
            "UPDATE video_assets v SET "
            "  club_id = c.club_id, "
            "  original_filename = COALESCE(v.original_filename, regexp_replace("
            "      COALESCE(v.processed_object_key, v.tracking_object_key, 'clip.mp4'), "
            "      '^.*/', '')), "
            "  content_type = COALESCE(v.content_type, 'video/mp4'), "
            "  processed_at = COALESCE(v.processed_at, now()) "
            "FROM clips c WHERE c.id = v.clip_id AND v.club_id IS NULL"
        )
    )
