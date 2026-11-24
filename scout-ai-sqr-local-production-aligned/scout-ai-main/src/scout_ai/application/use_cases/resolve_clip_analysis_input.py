from __future__ import annotations

import logging
from uuid import UUID

from scout_ai.domain.entities import Language
from scout_ai.domain.exceptions import VideoNotReadyError
from scout_ai.domain.ports import ClipRepository, VideoAssetsRepository
from scout_ai.domain.value_objects import ClipAnalysisInput, GcsUri

logger = logging.getLogger(__name__)


class ResolveClipAnalysisInputUseCase:
    def __init__(
        self,
        *,
        clips: ClipRepository,
        video_assets: VideoAssetsRepository,
        processed_bucket: str,
        tracking_bucket: str,
    ) -> None:
        self._clips = clips
        self._video_assets = video_assets
        self._processed_bucket = processed_bucket
        self._tracking_bucket = tracking_bucket

    async def execute(
        self,
        clip_id: UUID,
        club_id: UUID,
        player_id: UUID | None = None,
        language: Language = Language.english,
    ) -> ClipAnalysisInput:
        # `get_details` raises if the clip is unknown. Then branch on tracking:
        #   - tracking_object_key set  -> the clip was tracked; analyze the annotated
        #     (box-marked) video in the tracking bucket. Its presence is itself the
        #     "tracking finished" signal, so no separate readiness check is needed.
        #   - otherwise                -> analyze the downsized processed video; a missing
        #     processed_object_key means downsizing has not finished yet (retryable).
        asset = await self._video_assets.get_details(clip_id)

        if asset.tracking_object_key:
            video_uri = GcsUri(
                bucket=self._tracking_bucket,
                object_name=asset.tracking_object_key,
            )
            is_tracked = True
        else:
            if not asset.processed_object_key:
                raise VideoNotReadyError(str(clip_id))
            video_uri = GcsUri(
                bucket=self._processed_bucket,
                object_name=asset.processed_object_key,
            )
            is_tracked = False

        # Clip context used to steer the prompt and denormalized onto the review.
        clip = await self._clips.get_details(clip_id)

        logger.debug(
            "Resolved clip=%s video=%s tracked=%s action=%s",
            clip_id,
            video_uri.uri,
            is_tracked,
            clip.action_tag_key,
        )
        return ClipAnalysisInput(
            clip_id=clip_id,
            club_id=club_id,
            video_uri=video_uri,
            action_tag_key=clip.action_tag_key,
            ident_text=clip.ident_text,
            player_id=player_id,
            match_date=clip.match_date,
            language=language,
            is_tracked=is_tracked,
        )
