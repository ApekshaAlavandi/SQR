from __future__ import annotations

from uuid import UUID

from sqlalchemy import select

from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.adapters.db.models import VideoAssets
from scout_ai.domain.entities import (
    VideoAssetsDetails,
)
from scout_ai.domain.exceptions import ClipNotFoundError
from scout_ai.domain.ports import VideoAssetsRepository


class SqlAlchemyVideoAssetsRepository(VideoAssetsRepository):

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_details(self, clip_id: UUID) -> VideoAssetsDetails:
        result = await self._session.execute(
                    select(VideoAssets).where(VideoAssets.clip_id == clip_id)
                )
        model = result.scalar_one_or_none()
        if model is None:
            raise ClipNotFoundError(str(clip_id))
        return self._to_entity(model)

    def _to_entity(self, model: VideoAssets) -> VideoAssetsDetails:
        return VideoAssetsDetails(
            id=model.id,
            clip_id=model.clip_id,
            processed_object_key=model.processed_object_key,
            tracking_object_key=model.tracking_object_key,
        )
