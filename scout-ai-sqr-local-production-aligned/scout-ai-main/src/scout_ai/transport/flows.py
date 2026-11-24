"""Transport-agnostic flow processors.

Each flow turns a raw message payload (``bytes`` of JSON) into one use-case run and
reports a :class:`ProcessOutcome`. They know nothing about *how* the message arrived:

* ``scout_ai.transport.pubsub.worker`` feeds them messages pulled from Google Pub/Sub;
* ``scout_ai.transport.local`` (the ``scout-ai-local`` CLI) feeds them payloads typed on a
  command line.

Keeping them here (and free of any ``google.cloud`` import) is what lets local mode run
the *exact* production flow logic — idempotency check, error classification, failure
marking — without Pub/Sub installed or reachable.

Contract with the caller: returning a ``ProcessOutcome`` means "ack"; raising means
"retryable failure" (the Pub/Sub consumer nacks so the message is redelivered).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from enum import Enum, auto

from pydantic import ValidationError

from scout_ai.domain.entities import AIGenerationStatus
from scout_ai.domain.exceptions import (
    ClipNotFoundError,
    PlayerNotFoundError,
    SportCatalogNotFoundError,
)
from scout_ai.transport.container import AppContainer
from scout_ai.transport.pubsub.schemas import (
    DraftRequestedEvent,
    KpiRecomputeEvent,
    RegenerateSummaryEvent,
)

logger = logging.getLogger(__name__)


class ProcessOutcome(Enum):
    PROCESSED = auto()
    SKIPPED = auto()
    FAILED = auto()


Processor = Callable[[bytes], Awaitable[ProcessOutcome]]


# --------------------------------------------------------------------------- #
# Flow processors
# --------------------------------------------------------------------------- #
def _build_draft_processor(container: AppContainer):

    async def process(data: bytes) -> ProcessOutcome:
        try:
            event = DraftRequestedEvent.model_validate_json(data)
        except ValidationError:
            logger.warning("Dropping unparseable draft message")
            return ProcessOutcome.FAILED

        dto = event.to_dto()

        async with container.session_factory() as session:
            status = await container.clip_review_repository(session).get_status(dto.clip_id)
        if status in (AIGenerationStatus.completed, AIGenerationStatus.processing):
            logging.info("Clip %s is already being processed", dto.clip_id)
            return ProcessOutcome.SKIPPED

        # resolve gs:// URI + clip context (VideoNotReadyError -> nack).
        try:
            async with container.session_factory() as session:
                analysis_input = await container.resolve_input_use_case(session).execute(
                    dto.clip_id, dto.club_id, dto.player_id, dto.language
                )
        except ClipNotFoundError:
            logger.warning("clip %s has no clip/asset row; dropping", dto.clip_id)
            return ProcessOutcome.FAILED

        # Gemini analyses the video; persist the review.
        logging.info(
            "Analysis: clip_id=%s, video_uri=%s, tracked=%s, action_tag_key=%s, ident_text=%s",
            analysis_input.clip_id,
            analysis_input.video_uri.uri,
            analysis_input.is_tracked,
            analysis_input.action_tag_key,
            analysis_input.ident_text,
        )

        try:
            async with container.session_factory() as session:
                await container.generate_review_use_case(session).execute(analysis_input)
        except Exception:
            logger.exception("Draft processing failed: clip=%s", dto.clip_id)
            try:
                async with container.session_factory() as session:
                    await container.clip_review_repository(session).mark_ai_generation_status(
                        dto.clip_id, AIGenerationStatus.failed
                    )
            except Exception:
                logger.exception(
                    "Failed to mark clip %s as failed", dto.clip_id
                )
            raise
        return ProcessOutcome.PROCESSED

    return process


def _build_regenerate_processor(container: AppContainer):
    """ai-regenerate → per-player summary from recent scout-touched reviews."""

    async def process(data: bytes) -> ProcessOutcome:
        try:
            event = RegenerateSummaryEvent.model_validate_json(data)
        except ValidationError:
            logger.warning("Dropping unparseable regenerate message")
            return ProcessOutcome.FAILED

        dto = event.to_dto()
        async with container.session_factory() as session:
            summary = await container.generate_player_summary_use_case(session).execute(
                dto.club_id, dto.player_id, dto.language
            )

        if summary is None:
            logger.info("no reviewed clips for player %s; skipping", dto.player_id)
            return ProcessOutcome.SKIPPED
        return ProcessOutcome.PROCESSED

    return process


def _build_kpi_compute_processor(container: AppContainer):
    """ai-kpi-recompute → recompute a player's AI KPI scores from recent reviews."""

    async def process(data: bytes) -> ProcessOutcome:
        try:
            event = KpiRecomputeEvent.model_validate_json(data)
        except ValidationError:
            logger.warning("Dropping unparseable kpi-recompute message: %r", data[:500])
            return ProcessOutcome.FAILED

        dto = event.to_dto()
        logger.info(
            "kpi-recompute received: club=%s sport=%s player=%s",
            dto.club_id,
            dto.sport_id,
            dto.player_id,
        )
        try:
            async with container.session_factory() as session:
                written = await container.recompute_player_kpis_use_case(session).execute(
                    dto.club_id, dto.sport_id, dto.player_id, dto.language
                )
        except (PlayerNotFoundError, SportCatalogNotFoundError):
            # Non-retryable: the player or catalog is genuinely absent. Drop it.
            logger.warning(
                "kpi-recompute: missing player/catalog for player=%s sport=%s; dropping",
                dto.player_id,
                dto.sport_id,
            )
            return ProcessOutcome.FAILED
        except Exception:
            # Retryable: log with context, then re-raise so the consumer nacks it.
            logger.exception(
                "kpi-recompute failed (will retry): club=%s sport=%s player=%s",
                dto.club_id,
                dto.sport_id,
                dto.player_id,
            )
            raise

        if written is None:
            logger.info("kpi-recompute skipped: no KPIs/evidence for player=%s", dto.player_id)
            return ProcessOutcome.SKIPPED
        logger.info(
            "kpi-recompute processed: player=%s kpis_written=%d", dto.player_id, written
        )
        return ProcessOutcome.PROCESSED

    return process


# flow name -> processor builder. Names match the worker's ``--flow`` choices.
FLOW_BUILDERS: dict[str, Callable[[AppContainer], Processor]] = {
    "draft": _build_draft_processor,
    "regenerate": _build_regenerate_processor,
    "kpi_compute": _build_kpi_compute_processor,
}
