from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.domain.kpi import KpiScore
from scout_ai.domain.ports import PlayerKpisRepository

logger = logging.getLogger(__name__)

# Insert one row per KPI, or refresh the AI layer if it already exists. The DO UPDATE
# SET touches ONLY the AI columns (+ updated_at) — never review_status / scout_score /
# reviewed_at / reviewed_evidence_version — so chk_player_kpis_review stays satisfied and
# a scout's edits ride along untouched. A brand-new row defaults review_status='pending'.
_UPSERT_AI_SCORE = text(
    """
    INSERT INTO player_kpis (club_id, player_id, kpi_key, category_key,
        ai_score, ai_rationale, evidence_version, ai_model, computed_at)
    VALUES (:club_id, :player_id, :kpi_key, :category_key,
        :ai_score, :ai_rationale, :evidence_version, :ai_model, :computed_at)
    ON CONFLICT (player_id, kpi_key) DO UPDATE
       SET ai_score         = EXCLUDED.ai_score,
           ai_rationale     = EXCLUDED.ai_rationale,
           evidence_version = EXCLUDED.evidence_version,
           ai_model         = EXCLUDED.ai_model,
           computed_at      = EXCLUDED.computed_at,
           updated_at       = now()
    """
)


class SqlAlchemyPlayerKpisRepository(PlayerKpisRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_ai_scores(
        self,
        *,
        club_id: UUID,
        player_id: UUID,
        scores: list[KpiScore],
        evidence_version: str,
        model: str,
        computed_at: datetime,
    ) -> None:
        if not scores:
            logger.info("No KPI scores to upsert: player=%s", player_id)
            return

        params = [
            {
                "club_id": club_id,
                "player_id": player_id,
                "kpi_key": s.kpi_key,
                "category_key": s.category_key,
                # NULL ai_score = "not observed"; NUMERIC(4,2) wants a Decimal.
                "ai_score": None if s.score is None else Decimal(str(s.score)),
                "ai_rationale": s.rationale,
                "evidence_version": evidence_version,
                "ai_model": model,
                "computed_at": computed_at,
            }
            for s in scores
        ]
        await self._session.execute(_UPSERT_AI_SCORE, params)
        await self._session.commit()
        logger.info(
            "Upserted %d player_kpis rows: player=%s version=%s",
            len(params),
            player_id,
            evidence_version,
        )
