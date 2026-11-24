from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.domain.exceptions import PlayerNotFoundError
from scout_ai.domain.ports import PlayerRepository

logger = logging.getLogger(__name__)

# players is upstream-owned (no ORM model here). The unique (id, club_id) keeps this
# tenant-scoped read to a single row.
_GET_POSITION = text(
    """
    SELECT position_key
    FROM players
    WHERE id = :player_id AND club_id = :club_id
    """
)


class SqlAlchemyPlayerRepository(PlayerRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_position(self, player_id: UUID, club_id: UUID) -> str:
        logger.debug("players.position lookup: player=%s club=%s", player_id, club_id)
        result = await self._session.execute(
            _GET_POSITION, {"player_id": player_id, "club_id": club_id}
        )
        position_key = result.scalar_one_or_none()
        if position_key is None:
            logger.warning("No players row for player=%s club=%s", player_id, club_id)
            raise PlayerNotFoundError(str(player_id))
        return position_key
