from __future__ import annotations

import logging
from uuid import UUID

from scout_ai.domain.entities import Language, PlayerSummary
from scout_ai.domain.ports import (
    PlayerReviewsRepository,
    PlayerStatsRepository,
    PlayerSummarizer,
)

logger = logging.getLogger(__name__)


class GeneratePlayerSummaryUseCase:
    def __init__(
        self,
        *,
        reviews: PlayerReviewsRepository,
        summarizer: PlayerSummarizer,
        player_stats: PlayerStatsRepository,
        review_limit: int,
    ) -> None:
        self._reviews = reviews
        self._summarizer = summarizer
        self._player_stats = player_stats
        self._review_limit = review_limit

    async def execute(
        self,
        club_id: UUID,
        player_id: UUID,
        language: Language = Language.english,
    ) -> PlayerSummary | None:
        items = await self._reviews.latest_reviewed(player_id, club_id, self._review_limit)
        if not items:
            return None

        logger.info(
            "Regenerating summary: player=%s club=%s reviews=%d language=%s",
            player_id,
            club_id,
            len(items),
            language.value,
        )
        summary = await self._summarizer.summarize(items, language)

        # Upserts: inserts the player_stats row if it doesn't exist yet.
        await self._player_stats.set_latest_ai_summary(player_id, summary.summary)
        logger.info("Updated player_stats summary: player=%s", player_id)
        return summary
