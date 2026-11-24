"""Create the local schema and load a small, deterministic demo dataset."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from scout_ai.transport.local.schema import SCHEMA_STATEMENTS
from scout_ai.transport.local.schema_ext import (
    EXTENSION_STATEMENTS,
    LOCAL_USER_ID,
    backfill_extended,
)

# Fixed IDs so documented commands work verbatim after ``seed``.
CLUB_ID = UUID("00000000-0000-4000-8000-0000000000c1")
SPORT_ID = UUID("00000000-0000-4000-8000-0000000000a1")  # club_sports.id
PLAYER_ID = UUID("00000000-0000-4000-8000-0000000000b1")
CLIP_UNTRACKED = UUID("00000000-0000-4000-8000-000000000101")  # needs a draft
CLIP_TRACKED = UUID("00000000-0000-4000-8000-000000000102")  # needs a draft (tracked)
CLIP_REVIEWED_A = UUID("00000000-0000-4000-8000-000000000103")  # already scout-reviewed
CLIP_REVIEWED_B = UUID("00000000-0000-4000-8000-000000000104")  # already scout-reviewed

_CATALOG = {
    "buckets": [
        {"key": "GK", "positions": ["GK"]},
        {"key": "DEF", "positions": ["CB", "LB", "RB"]},
        {"key": "MID", "positions": ["CDM", "CM", "CAM"]},
        {"key": "FWD", "positions": ["LW", "RW", "ST"]},
    ],
    "kpi_categories": [
        {"key": "passing", "buckets": ["DEF", "MID", "FWD"]},
        {"key": "defending", "buckets": ["DEF", "MID"]},
        {"key": "shooting", "buckets": ["MID", "FWD"]},
        {"key": "goalkeeping", "buckets": ["GK"]},
        {"key": "physical", "buckets": ["GK", "DEF", "MID", "FWD"]},
    ],
    "kpis": [
        {"key": "short_passing", "category": "passing", "label_en": "Short passing",
         "measures": "Accuracy and weight of short passes under pressure"},
        {"key": "progressive_passing", "category": "passing", "label_en": "Progressive passing",
         "measures": "Passes that break lines or advance play"},
        {"key": "tackling", "category": "defending", "label_en": "Tackling",
         "measures": "Timing and success of tackles"},
        {"key": "interceptions", "category": "defending", "label_en": "Interceptions",
         "measures": "Reading of play and cutting passing lanes"},
        {"key": "finishing", "category": "shooting", "label_en": "Finishing",
         "measures": "Shot quality and conversion"},
        {"key": "shot_stopping", "category": "goalkeeping", "label_en": "Shot stopping",
         "measures": "Saves and positioning against shots"},
        {"key": "work_rate", "category": "physical", "label_en": "Work rate",
         "measures": "Off-ball running and recovery effort"},
    ],
}


async def init_db(engine: AsyncEngine) -> None:
    """Create enums and tables (idempotent)."""
    async with engine.begin() as conn:
        for stmt in (*SCHEMA_STATEMENTS, *EXTENSION_STATEMENTS):
            await conn.execute(text(stmt))
        # Clips created by older local runs have no club_id/player_id; the B-screen tables
        # reference clips(id, club_id), so backfill from what we already know.
        await conn.execute(
            text(
                "UPDATE clips c SET club_id = r.club_id, player_id = COALESCE(c.player_id, r.player_id) "
                "FROM clip_reviews r WHERE r.clip_id = c.id AND c.club_id IS NULL"
            )
        )
        await conn.execute(
            text(
                "UPDATE clips SET club_id = :club, player_id = COALESCE(player_id, :player), "
                "sport_id = COALESCE(sport_id, :sport) WHERE id = ANY(:ids) AND club_id IS NULL"
            ),
            {
                "club": CLUB_ID, "player": PLAYER_ID, "sport": SPORT_ID,
                "ids": [CLIP_UNTRACKED, CLIP_TRACKED, CLIP_REVIEWED_A, CLIP_REVIEWED_B],
            },
        )
        await backfill_extended(conn)


async def seed(engine: AsyncEngine) -> None:
    """Insert demo data (idempotent; never overwrites rows that already exist)."""
    now = datetime.now(UTC)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO sport_catalog (sport_key, default_config) "
                "VALUES ('football', CAST(:cfg AS jsonb)) ON CONFLICT DO NOTHING"
            ),
            {"cfg": json.dumps(_CATALOG)},
        )
        await conn.execute(
            text(
                "INSERT INTO club_sports (id, club_id, sport_key) "
                "VALUES (:id, :club, 'football') ON CONFLICT DO NOTHING"
            ),
            {"id": SPORT_ID, "club": CLUB_ID},
        )
        await conn.execute(
            text(
                "INSERT INTO players (id, club_id, name, position_key) "
                "VALUES (:id, :club, 'Demo Midfielder', 'CM') ON CONFLICT DO NOTHING"
            ),
            {"id": PLAYER_ID, "club": CLUB_ID},
        )

        clips = [
            # (id, action, date, ident_text, processed_key, tracking_key)
            (CLIP_UNTRACKED, "pass", "2026-10-01", "Midfielder in the white shirt, number 8",
             "demo/clip1.mp4", None),
            (CLIP_TRACKED, "tackle", "2026-10-08", None,
             "demo/clip2.mp4", "demo/clip2_tracked.mp4"),
            (CLIP_REVIEWED_A, "shot", "2026-09-20", "Number 8, white shirt",
             "demo/clip3.mp4", None),
            (CLIP_REVIEWED_B, "dribble", "2026-09-27", "Number 8, white shirt",
             "demo/clip4.mp4", None),
        ]
        for i, (cid, action, mdate, ident, processed, tracking) in enumerate(clips):
            await conn.execute(
                text(
                    "INSERT INTO clips (id, opponent_name, match_date, action_tag_key, "
                    "minute, ident_text, upload_status, club_id, sport_id, player_id) "
                    "VALUES (:id, :opp, CAST(:d AS date), :action, :minute, :ident, 'completed', "
                    ":club, :sport, :player) ON CONFLICT DO NOTHING"
                ),
                {"id": cid, "opp": f"Demo FC {i + 1}", "d": date.fromisoformat(mdate),
                 "action": action, "minute": 10 + i * 5, "ident": ident,
                 "club": CLUB_ID, "sport": SPORT_ID, "player": PLAYER_ID},
            )
            await conn.execute(
                text(
                    "INSERT INTO video_assets (id, clip_id, processed_object_key, "
                    "tracking_object_key) VALUES (gen_random_uuid(), :clip, :p, :t) "
                    "ON CONFLICT (clip_id) DO NOTHING"
                ),
                {"clip": cid, "p": processed, "t": tracking},
            )

        # Two clips that already carry scout-touched reviews, so the summary and KPI
        # flows have evidence immediately (the other two get theirs via draft + approve).
        reviewed = [
            (CLIP_REVIEWED_A, "2026-09-20", "edited_by_scout", 6.5, "monitor", 7.0, "monitor",
             "Scout: good range of shooting, decision-making inconsistent."),
            (CLIP_REVIEWED_B, "2026-09-27", "accepted_ai_draft", 7.5, "sign", None, None, None),
        ]
        for cid, mdate, rstatus, ai_score, ai_verdict, h_score, h_verdict, h_review in reviewed:
            await conn.execute(
                text(
                    "INSERT INTO clip_reviews (clip_id, club_id, player_id, match_date, "
                    "ai_review, ai_score, ai_verdict, human_review, human_score, "
                    "human_verdict, ai_generation_status, review_status, review_source, "
                    "ai_model, reviewed_at) VALUES (:clip, :club, :player, CAST(:d AS date), "
                    "'Seeded AI review.', :ai_score, CAST(:ai_verdict AS clip_verdict), "
                    ":h_review, :h_score, CAST(:h_verdict AS clip_verdict), 'completed', "
                    "CAST(:rstatus AS review_status), CAST(:rsource AS review_source), "
                    "'seed', :now) "
                    "ON CONFLICT (clip_id) DO NOTHING"
                ),
                {"clip": cid, "club": CLUB_ID, "player": PLAYER_ID, "d": date.fromisoformat(mdate),
                 "ai_score": Decimal(str(ai_score)), "ai_verdict": ai_verdict,
                 "h_review": h_review,
                 "h_score": None if h_score is None else Decimal(str(h_score)),
                 "h_verdict": h_verdict, "rstatus": rstatus, "now": now,
                 "rsource": "human" if rstatus == "edited_by_scout" else "ai"},
            )

        await backfill_extended(conn)


async def create_clip(
    engine: AsyncEngine,
    *,
    club_id: UUID,
    player_id: UUID,
    player_name: str | None,
    position_key: str,
    clip_id: UUID,
    opponent_name: str,
    match_date: date,
    action_tag_key: str,
    minute: int | None,
    ident_text: str | None,
    processed_object_key: str | None,
    tracking_object_key: str | None,
    # --- columns added upstream to ``clips`` (see schema_ext.py) ---
    language: str = "en",
    position_played: str | None = None,
    half: int | None = None,
    footage_class: str = "short",
    competition: str | None = None,
    venue: str | None = None,
) -> UUID:
    """Register one real clip (plus what the flows need around it) in a single transaction.

    Stands in for the upstream service: the clip, its video record, the player, and the
    football sport catalog / club-sport link that the KPI flow reads. Existing players and
    catalog rows are reused, never overwritten. Returns the ``club_sports.id`` to use with
    the ``kpis`` command.
    """
    async with engine.begin() as conn:
        exists = await conn.execute(text("SELECT 1 FROM clips WHERE id = :id"), {"id": clip_id})
        if exists.first():
            raise ValueError(f"clip {clip_id} already exists")

        owner_row = await conn.execute(
            text("SELECT club_id FROM players WHERE id = :id"), {"id": player_id}
        )
        owner = owner_row.first()
        if owner is not None and owner[0] != club_id:
            raise ValueError(
                f"player {player_id} already belongs to club {owner[0]}, not {club_id}"
            )
        if owner is None:
            await conn.execute(
                text(
                    "INSERT INTO players (id, club_id, name, position_key) "
                    "VALUES (:id, :club, :name, :pos)"
                ),
                {"id": player_id, "club": club_id, "name": player_name, "pos": position_key},
            )

        await conn.execute(
            text(
                "INSERT INTO sport_catalog (sport_key, default_config) "
                "VALUES ('football', CAST(:cfg AS jsonb)) ON CONFLICT DO NOTHING"
            ),
            {"cfg": json.dumps(_CATALOG)},
        )
        row = (
            await conn.execute(
                text(
                    "SELECT id FROM club_sports WHERE club_id = :club AND sport_key = 'football' "
                    "LIMIT 1"
                ),
                {"club": club_id},
            )
        ).first()
        if row is not None:
            sport_id: UUID = row[0]
        else:
            sport_id = SPORT_ID if club_id == CLUB_ID else uuid4()
            await conn.execute(
                text(
                    "INSERT INTO club_sports (id, club_id, sport_key) "
                    "VALUES (:id, :club, 'football') ON CONFLICT DO NOTHING"
                ),
                {"id": sport_id, "club": club_id},
            )

        await conn.execute(
            text(
                "INSERT INTO clips (id, opponent_name, match_date, action_tag_key, minute, "
                "ident_text, upload_status, club_id, sport_id, player_id, created_by, "
                "language, position_played, half, footage_class, competition, venue) "
                "VALUES (:id, :opp, :d, :action, :minute, :ident, 'completed', :club, :sport, "
                ":player, :user, :lang, :pos, :half, :fclass, :comp, :venue)"
            ),
            {"id": clip_id, "opp": opponent_name, "d": match_date, "action": action_tag_key,
             "minute": minute, "ident": ident_text, "club": club_id, "sport": sport_id,
             "player": player_id, "user": LOCAL_USER_ID,
             "lang": language if language in ("en", "ar") else "en",
             "pos": position_played or position_key, "half": half,
             "fclass": footage_class, "comp": competition, "venue": venue},
        )
        await conn.execute(
            text(
                "INSERT INTO video_assets (id, clip_id, processed_object_key, "
                "tracking_object_key) VALUES (gen_random_uuid(), :clip, :p, :t)"
            ),
            {"clip": clip_id, "p": processed_object_key, "t": tracking_object_key},
        )
        await backfill_extended(conn)
    return sport_id
