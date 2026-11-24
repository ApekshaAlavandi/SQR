"""Minimal local Postgres schema.

In production these tables are owned by the upstream (scout) service and scout-ai only reads
and writes the columns below. This is a *local approximation* of exactly those columns —
enough to run every flow — NOT the authoritative schema. Never apply it to a shared database
(the CLI refuses non-local hosts unless ``--allow-remote-db`` is given).

Kept as a tuple of single statements because asyncpg cannot run several in one call.
"""

from __future__ import annotations

SCHEMA_STATEMENTS: tuple[str, ...] = (
    # ---- enum types (the ORM maps them with create_type=False) ------------------------
    """
    DO $$ BEGIN
        CREATE TYPE clip_verdict AS ENUM ('sign', 'monitor', 'pass');
    EXCEPTION WHEN duplicate_object THEN NULL; END $$
    """,
    """
    DO $$ BEGIN
        CREATE TYPE ai_generation_status AS ENUM
            ('pending', 'processing', 'completed', 'failed');
    EXCEPTION WHEN duplicate_object THEN NULL; END $$
    """,
    """
    DO $$ BEGIN
        CREATE TYPE review_status AS ENUM
            ('pending', 'edited_by_scout', 'accepted_ai_draft');
    EXCEPTION WHEN duplicate_object THEN NULL; END $$
    """,
    # ---- upstream-owned tables --------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS clips (
        id             uuid PRIMARY KEY,
        opponent_name  text     NOT NULL,
        match_date     date     NOT NULL,
        action_tag_key text     NOT NULL,
        minute         smallint,
        ident_text     text,
        upload_status  text     NOT NULL DEFAULT 'ready'
    )
    """,
    # Columns the B-screen tables / view need (real schema: NOT NULL; nullable here so
    # clips created by older local runs keep working - they are backfilled below).
    "ALTER TABLE clips ADD COLUMN IF NOT EXISTS club_id   uuid",
    "ALTER TABLE clips ADD COLUMN IF NOT EXISTS sport_id  uuid",
    "ALTER TABLE clips ADD COLUMN IF NOT EXISTS player_id uuid",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_clips_id_club ON clips (id, club_id)",
    """
    CREATE TABLE IF NOT EXISTS video_assets (
        id                   uuid PRIMARY KEY,
        clip_id              uuid NOT NULL UNIQUE,
        processed_object_key text,
        thumbnail_object_key text,
        tracking_object_key  text
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS players (
        id           uuid PRIMARY KEY,
        club_id      uuid NOT NULL,
        name         text,
        position_key text NOT NULL,
        UNIQUE (id, club_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS club_sports (
        id        uuid PRIMARY KEY,
        club_id   uuid NOT NULL,
        sport_key text NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sport_catalog (
        sport_key      text PRIMARY KEY,
        default_config jsonb NOT NULL
    )
    """,
    # ---- tables scout-ai writes to ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS clip_reviews (
        id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
        clip_id              uuid NOT NULL UNIQUE,
        club_id              uuid NOT NULL,
        player_id            uuid,
        match_date           date,
        ai_review            text,
        ai_score             numeric(4, 2),
        ai_verdict           clip_verdict,
        human_review         text,
        human_score          numeric(4, 2),
        human_verdict        clip_verdict,
        ai_generation_status ai_generation_status NOT NULL DEFAULT 'pending',
        review_status        review_status NOT NULL DEFAULT 'pending',
        ai_model             text,
        ai_total_tokens      integer,
        reviewed_by          uuid,
        reviewed_at          timestamptz,
        created_at           timestamptz NOT NULL DEFAULT now(),
        updated_at           timestamptz NOT NULL DEFAULT now()
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_clip_reviews_player_id ON clip_reviews (player_id)",
    """
    CREATE TABLE IF NOT EXISTS player_kpis (
        club_id                   uuid NOT NULL,
        player_id                 uuid NOT NULL,
        kpi_key                   text NOT NULL,
        category_key              text,
        ai_score                  numeric(4, 2),
        ai_rationale              text,
        evidence_version          text,
        ai_model                  text,
        computed_at               timestamptz,
        review_status             text NOT NULL DEFAULT 'pending',
        scout_score               numeric(4, 2),
        reviewed_at               timestamptz,
        reviewed_evidence_version text,
        created_at                timestamptz NOT NULL DEFAULT now(),
        updated_at                timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (player_id, kpi_key)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_stats (
        player_id  uuid PRIMARY KEY,
        ai_summary text,
        updated_at timestamptz NOT NULL DEFAULT now()
    )
    """,
    # ---- evidence-first analysis tables (B screens) -----------------------------------
    # Copied from the real schema (schema_new.sql). Differences: no processing_jobs FK on
    # clip_analyses.job_id (that table is upstream-owned), and no clip_segments table
    # (scout-ai does not write it yet).
    """
    CREATE OR REPLACE FUNCTION clip_int_array_sum(arr integer[]) RETURNS bigint
        LANGUAGE sql IMMUTABLE
        AS $$ SELECT COALESCE(SUM(value), 0) FROM unnest(arr) AS value; $$
    """,
    """
    CREATE OR REPLACE FUNCTION clip_jsonb_int_sum(data jsonb) RETURNS bigint
        LANGUAGE sql IMMUTABLE
        AS $$ SELECT COALESCE(SUM(value::BIGINT), 0)
              FROM jsonb_each_text(data) AS entry(key, value); $$
    """,
    """
    CREATE TABLE IF NOT EXISTS clip_analyses (
        id                uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
        club_id           uuid NOT NULL,
        clip_id           uuid NOT NULL,
        job_id            uuid,
        lane              text DEFAULT 'club' NOT NULL,
        version           integer DEFAULT 1 NOT NULL,
        parent_version_id uuid,
        status            text DEFAULT 'published' NOT NULL,
        rating_value      numeric(3,1),
        duration_s        numeric(8,2) NOT NULL,
        minutes_in_frame  numeric(6,2) NOT NULL,
        coverage          numeric(4,3) GENERATED ALWAYS AS
                          (round(((minutes_in_frame * 60) / duration_s), 3)) STORED,
        identification    jsonb NOT NULL,
        confidence_by_area jsonb,
        not_observed      text[] DEFAULT '{}' NOT NULL,
        prose_en          text,
        prose_ar          text,
        model_version     text NOT NULL,
        taxonomy_version  text NOT NULL,
        published_at      timestamptz,
        created_at        timestamptz DEFAULT now() NOT NULL,
        updated_at        timestamptz DEFAULT now() NOT NULL,
        CONSTRAINT chk_clip_analyses_duration CHECK (duration_s > 0),
        CONSTRAINT chk_clip_analyses_lane CHECK (lane = ANY (ARRAY['club','self'])),
        CONSTRAINT chk_clip_analyses_minutes CHECK (minutes_in_frame >= 0
            AND minutes_in_frame * 60 <= duration_s + 3),
        CONSTRAINT chk_clip_analyses_parent CHECK (parent_version_id IS NULL
            OR parent_version_id <> id),
        CONSTRAINT chk_clip_analyses_rating CHECK (rating_value IS NULL
            OR (rating_value >= 0 AND rating_value <= 10)),
        CONSTRAINT chk_clip_analyses_status CHECK (status = ANY (ARRAY['published','superseded'])),
        CONSTRAINT chk_clip_analyses_version CHECK (version >= 1),
        CONSTRAINT uq_clip_analyses_id_clip UNIQUE (id, clip_id),
        CONSTRAINT uq_clip_analyses_id_club UNIQUE (id, club_id),
        CONSTRAINT uq_clip_analyses_version UNIQUE (clip_id, version),
        CONSTRAINT fk_clip_analyses_clip FOREIGN KEY (clip_id, club_id)
            REFERENCES clips (id, club_id) ON DELETE CASCADE,
        CONSTRAINT fk_clip_analyses_parent FOREIGN KEY (parent_version_id, clip_id)
            REFERENCES clip_analyses (id, clip_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clip_analyses_club_clip ON clip_analyses (club_id, clip_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_clip_analyses_one_published "
    "ON clip_analyses (clip_id) WHERE status = 'published'",
    """
    CREATE TABLE IF NOT EXISTS clip_presence_spans (
        analysis_id uuid NOT NULL,
        club_id     uuid NOT NULL,
        t0          numeric(8,2) NOT NULL,
        t1          numeric(8,2) NOT NULL,
        state       text NOT NULL,
        PRIMARY KEY (analysis_id, t0),
        CONSTRAINT chk_presence_state CHECK (state = ANY (ARRAY['in','out','off'])),
        CONSTRAINT chk_presence_time CHECK (t0 >= 0 AND t1 > t0),
        CONSTRAINT fk_presence_analysis FOREIGN KEY (analysis_id, club_id)
            REFERENCES clip_analyses (id, club_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS clip_observations (
        id             uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
        club_id        uuid NOT NULL,
        analysis_id    uuid NOT NULL,
        seq            integer NOT NULL,
        t0             numeric(8,2) NOT NULL,
        t1             numeric(8,2) NOT NULL,
        phase          text NOT NULL,
        zone           smallint NOT NULL,
        action         text NOT NULL,
        quality        text NOT NULL,
        significance   smallint NOT NULL,
        attributes     text[] DEFAULT '{}' NOT NULL,
        outcome        text NOT NULL,
        confidence     numeric(3,2) NOT NULL,
        low_confidence boolean DEFAULT false NOT NULL,
        text_en        text NOT NULL,
        text_ar        text NOT NULL,
        cited_in_prose boolean DEFAULT false NOT NULL,
        created_at     timestamptz DEFAULT now() NOT NULL,
        CONSTRAINT chk_obs_confidence CHECK (confidence >= 0 AND confidence <= 1),
        CONSTRAINT chk_obs_outcome CHECK (outcome = ANY (ARRAY['success','fail','neutral'])),
        CONSTRAINT chk_obs_phase CHECK (phase = ANY (ARRAY['oa','ot','od','dt','sp'])),
        CONSTRAINT chk_obs_quality CHECK (quality = ANY (ARRAY['good','work','neutral'])),
        CONSTRAINT chk_obs_significance CHECK (significance >= 1 AND significance <= 3),
        CONSTRAINT chk_obs_time CHECK (t0 >= 0 AND t0 < t1),
        CONSTRAINT chk_obs_zone CHECK (zone >= 1 AND zone <= 18),
        CONSTRAINT uq_clip_obs_id_analysis UNIQUE (id, analysis_id),
        CONSTRAINT uq_clip_obs_seq UNIQUE (analysis_id, seq),
        CONSTRAINT fk_clip_obs_analysis FOREIGN KEY (analysis_id, club_id)
            REFERENCES clip_analyses (id, club_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_clip_obs_analysis_phase "
    "ON clip_observations (analysis_id, phase)",
    "CREATE INDEX IF NOT EXISTS idx_clip_obs_attributes ON clip_observations USING gin (attributes)",
    """
    CREATE TABLE IF NOT EXISTS clip_tallies (
        analysis_id uuid NOT NULL PRIMARY KEY,
        club_id     uuid NOT NULL,
        touches     integer NOT NULL,
        by_minute   integer[] NOT NULL,
        by_zone     jsonb NOT NULL,
        actions     jsonb NOT NULL,
        shots       jsonb DEFAULT '[]' NOT NULL,
        confidence  jsonb NOT NULL,
        created_at  timestamptz DEFAULT now() NOT NULL,
        CONSTRAINT chk_tallies_by_minute CHECK (touches = clip_int_array_sum(by_minute)),
        CONSTRAINT chk_tallies_by_zone CHECK (touches = clip_jsonb_int_sum(by_zone)),
        CONSTRAINT chk_tallies_shots CHECK (jsonb_typeof(shots) = 'array'),
        CONSTRAINT chk_tallies_touches CHECK (touches >= 0),
        CONSTRAINT fk_tallies_analysis FOREIGN KEY (analysis_id, club_id)
            REFERENCES clip_analyses (id, club_id) ON DELETE CASCADE
    )
    """,
    """
    CREATE OR REPLACE VIEW v_clip_rating_trend AS
     SELECT c.club_id, c.player_id, c.id AS clip_id, a.id AS analysis_id,
            c.match_date AS at, a.rating_value AS rating,
            COALESCE((r.review_status = 'pending'::review_status), true) AS rating_provisional
       FROM clip_analyses a
       JOIN clips c ON c.id = a.clip_id AND c.club_id = a.club_id
       LEFT JOIN clip_reviews r ON r.clip_id = c.id
      WHERE a.status = 'published'
    """,
)
