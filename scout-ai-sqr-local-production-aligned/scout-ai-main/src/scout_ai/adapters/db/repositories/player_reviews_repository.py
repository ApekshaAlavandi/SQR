from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.adapters.db.models import ClipReviewModel
from scout_ai.domain.entities import PlayerReviewItem, ReviewStatus
from scout_ai.domain.ports import PlayerReviewsRepository

# Only reviews a scout has touched feed the player summary.
_SCOUT_TOUCHED = (ReviewStatus.edited_by_scout, ReviewStatus.accepted_ai_draft)


class SqlAlchemyPlayerReviewsRepository(PlayerReviewsRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def latest_reviewed(
        self, player_id: UUID, club_id: UUID, limit: int
    ) -> list[PlayerReviewItem]:
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
        return [self._to_item(row) for row in rows]

    @staticmethod
    def _to_item(model: ClipReviewModel) -> PlayerReviewItem:
        return PlayerReviewItem(
            match_date=model.match_date,
            review_status=model.review_status,
            ai_review=model.ai_review,
            ai_score=model.ai_score,
            ai_verdict=model.ai_verdict,
            human_review=model.human_review,
            human_score=model.human_score,
            human_verdict=model.human_verdict,
        )
