from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.domain.ports import PlayerStatsRepository

# Insert the row if the player has none yet; otherwise update in place.
_UPSERT_SUMMARY = text(
    """
    INSERT INTO player_stats (player_id, ai_summary)
    VALUES (:player_id, :summary)
    ON CONFLICT (player_id) DO UPDATE
       SET ai_summary = EXCLUDED.ai_summary,
           updated_at = now()
    """
)


class SqlAlchemyPlayerStatsRepository(PlayerStatsRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def set_latest_ai_summary(self, player_id: UUID, summary: str) -> None:
        await self._session.execute(
            _UPSERT_SUMMARY, {"player_id": player_id, "summary": summary}
        )
        await self._session.commit()
