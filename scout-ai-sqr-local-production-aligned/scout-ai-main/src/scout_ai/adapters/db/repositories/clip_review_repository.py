from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.adapters.db.models import ClipReviewModel
from scout_ai.domain.entities import AIGenerationStatus, ClipReview
from scout_ai.domain.exceptions import ClipReviewNotFoundError
from scout_ai.domain.ports import ClipReviewRepository

# Create the row in 'processing' if the clip has none yet; otherwise just flip it
# back to 'processing'. Only touches the columns we own — never the human fields.
_START_AI_GENERATION = text(
    """
    INSERT INTO clip_reviews (id, clip_id, club_id, player_id, match_date,
                              ai_generation_status)
    VALUES (gen_random_uuid(), :clip_id, :club_id, :player_id, :match_date,
            'processing')
    ON CONFLICT (clip_id) DO UPDATE
       SET ai_generation_status = 'processing'
    """
)


class SqlAlchemyClipReviewRepository(ClipReviewRepository):

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def start_ai_generation(
        self,
        clip_id: UUID,
        club_id: UUID,
        player_id: UUID | None = None,
        match_date: date | None = None,
    ) -> None:
        await self._session.execute(
            _START_AI_GENERATION,
            {
                "clip_id": clip_id,
                "club_id": club_id,
                "player_id": player_id,
                "match_date": match_date,
            },
        )
        await self._session.commit()

    async def get(self, clip_id: UUID) -> ClipReview:
        result = await self._session.execute(
            select(ClipReviewModel).where(ClipReviewModel.clip_id == clip_id)
        )
        model = result.scalar_one_or_none()
        if model is None:
            raise ClipReviewNotFoundError(str(clip_id))

        return self._to_entity(model)


    async def get_status(self, clip_id: UUID) -> AIGenerationStatus | None:
        stmt = select(ClipReviewModel.ai_generation_status).where(
            ClipReviewModel.clip_id == clip_id
        )
        value = (await self._session.execute(stmt)).scalar_one_or_none()
        return AIGenerationStatus(value) if value is not None else None

    async def mark_ai_generation_status(
        self,
        clip_id: UUID,
        status: AIGenerationStatus,
    ) -> None:
        review = await self.get(clip_id)
        review.ai_generation_status = status
        await self.update(review)

    async def save_ai_review(
        self,
        clip_id: UUID,
        review: str,
        score: Decimal,
        verdict: str,
        model: str,
        player_id: UUID | None = None,
        match_date: date | None = None,
        total_tokens: int | None = None,
    ) -> None:
        clip_review = await self.get(clip_id)

        clip_review.ai_review = review
        clip_review.ai_score = score
        clip_review.ai_verdict = verdict
        clip_review.ai_model = model
        clip_review.total_tokens = total_tokens
        clip_review.ai_generation_status = AIGenerationStatus.completed

        # Denormalize so the player-summary flow can read clip_reviews alone.
        if player_id is not None:
            clip_review.player_id = player_id
        if match_date is not None:
            clip_review.match_date = match_date

        await self.update(clip_review)

    async def update(self, clip_review: ClipReview) -> None:
        model = await self._session.get(
            ClipReviewModel,
            clip_review.id,
        )

        if model is None:
            raise ClipReviewNotFoundError(str(clip_review.id))

        self._apply(clip_review, model)
        await self._session.commit()

    # ----------------------------------------------------------------- #
    # Mapping helpers
    # ----------------------------------------------------------------- #

    def _apply(
        self,
        review: ClipReview,
        model: ClipReviewModel,
    ) -> None:
        model.clip_id = review.clip_id
        model.club_id = review.club_id
        model.player_id = review.player_id
        model.match_date = review.match_date

        model.ai_review = review.ai_review
        model.ai_score = review.ai_score
        model.ai_verdict = review.ai_verdict

        model.human_review = review.human_review
        model.human_score = review.human_score
        model.human_verdict = review.human_verdict

        model.ai_generation_status = review.ai_generation_status
        model.review_status = review.review_status

        model.ai_model = review.ai_model
        model.ai_total_tokens = review.total_tokens

        model.reviewed_by = review.reviewed_by
        model.reviewed_at = review.reviewed_at

        model.created_at = review.created_at
        model.updated_at = review.updated_at

    def _to_entity(self, model: ClipReviewModel) -> ClipReview:
        return ClipReview(
            id=model.id,
            clip_id=model.clip_id,
            club_id=model.club_id,
            player_id=model.player_id,
            match_date=model.match_date,
            ai_review=model.ai_review,
            ai_score=model.ai_score,
            ai_verdict=model.ai_verdict,
            human_review=model.human_review,
            human_score=model.human_score,
            human_verdict=model.human_verdict,
            ai_generation_status=model.ai_generation_status,
            review_status=model.review_status,
            ai_model=model.ai_model,
            reviewed_by=model.reviewed_by,
            reviewed_at=model.reviewed_at,
            created_at=model.created_at,
            updated_at=model.updated_at,
            total_tokens=model.ai_total_tokens,
        )
