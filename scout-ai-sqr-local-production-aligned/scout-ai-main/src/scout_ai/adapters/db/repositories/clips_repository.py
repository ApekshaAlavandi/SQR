from __future__ import annotations

from uuid import UUID
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.adapters.db.models import Clips
from scout_ai.domain.entities import (
    ClipDetails,
)
from scout_ai.domain.exceptions import ClipNotFoundError
from scout_ai.domain.ports import ClipRepository


logger = logging.getLogger(__name__)


class SqlAlchemyClipRepository(ClipRepository):

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_details(self, clip_id: UUID) -> ClipDetails:
        logging.info(f"Getting clip details for clip_id: {clip_id}")
        model = await self._session.get(Clips, clip_id)
        logging.info(f"Retrieved model from database: {model}")
        if model is None:
            raise ClipNotFoundError(str(clip_id))
        return self._to_entity(model)

    def _to_entity(self, model: Clips) -> ClipDetails:
        return ClipDetails(
            id=model.id,
            opponent_name=model.opponent_name,
            match_date=model.match_date,
            action_tag_key=model.action_tag_key,
            minute=model.minute,
            ident_text=model.ident_text,
            upload_status=model.upload_status,
        )
