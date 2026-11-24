from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.adapters.db.models import ClipReviewModel
from scout_ai.domain.entities import ClipReviewEvidence, ReviewStatus
from scout_ai.domain.ports import ClipReviewEvidenceRepository

logger = logging.getLogger(__name__)

# Only reviews a scout has touched are trustworthy KPI evidence (same rule the
# player-summary flow uses).
_SCOUT_TOUCHED = (ReviewStatus.edited_by_scout, ReviewStatus.accepted_ai_draft)


class SqlAlchemyClipReviewEvidenceRepository(ClipReviewEvidenceRepository):
    """Reads a player's recent scout-touched clip reviews as KPI evidence.

    Distinct from ``SqlAlchemyPlayerReviewsRepository`` (player-summary flow): this one
    carries ``clip_id`` and ``updated_at`` so the KPI flow can fingerprint the set.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def latest_scout_touched(
        self, player_id: UUID, club_id: UUID, limit: int
    ) -> list[ClipReviewEvidence]:
        stmt = (
            select(ClipReviewModel)
            .where(
                ClipReviewModel.player_id == player_id,
                ClipReviewModel.club_id == club_id,
                ClipReviewModel.review_status.in_(_SCOUT_TOUCHED),
            )
            .order_by(ClipReviewModel.match_date.desc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        logger.debug(
            "scout-touched evidence: player=%s club=%s limit=%d -> %d rows",
            player_id,
            club_id,
            limit,
            len(rows),
        )
        return [self._to_evidence(row) for row in rows]

    @staticmethod
    def _to_evidence(model: ClipReviewModel) -> ClipReviewEvidence:
        return ClipReviewEvidence(
            clip_id=model.clip_id,
            match_date=model.match_date,
            review_status=model.review_status,
            ai_review=model.ai_review,
            ai_score=model.ai_score,
            ai_verdict=model.ai_verdict,
            human_review=model.human_review,
            human_score=model.human_score,
            human_verdict=model.human_verdict,
            updated_at=model.updated_at,
        )
