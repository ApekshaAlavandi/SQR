from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import UUID

from scout_ai.domain.entities import Language
from scout_ai.domain.kpi import evidence_version
from scout_ai.domain.ports import (
    ClipReviewEvidenceRepository,
    KpiScorer,
    PlayerKpisRepository,
    PlayerRepository,
    SportCatalogRepository,
)

logger = logging.getLogger(__name__)


class RecomputePlayerKpisUseCase:
    """Recompute the AI layer of a player's ``player_kpis`` from recent reviews.

    Picks the KPI set for the player's position bucket, scores it from the last-N
    scout-touched clip reviews, and upserts the AI columns — leaving the scout review
    layer untouched. Returns the number of KPI rows written, or ``None`` when there's
    nothing to do (no bucket/KPIs, or no evidence yet).
    """

    def __init__(
        self,
        *,
        players: PlayerRepository,
        catalog: SportCatalogRepository,
        evidence: ClipReviewEvidenceRepository,
        scorer: KpiScorer,
        kpis: PlayerKpisRepository,
        model_name: str,
        evidence_limit: int,
    ) -> None:
        self._players = players
        self._catalog = catalog
        self._evidence = evidence
        self._scorer = scorer
        self._kpis = kpis
        self._model_name = model_name
        self._evidence_limit = evidence_limit

    async def execute(
        self,
        club_id: UUID,
        sport_id: UUID,
        player_id: UUID,
        language: Language = Language.english,
    ) -> int | None:
        logger.info(
            "KPI recompute start: club=%s sport=%s player=%s (evidence_limit=%d language=%s)",
            club_id,
            sport_id,
            player_id,
            self._evidence_limit,
            language.value,
        )

        # PlayerNotFoundError / SportCatalogNotFoundError propagate: the caller drops
        # them (non-retryable), since retrying won't make the missing row appear.
        position_key = await self._players.get_position(player_id, club_id)
        logger.info("Resolved player=%s position_key=%s", player_id, position_key)

        catalog = await self._catalog.get(sport_id, club_id)
        logger.info(
            "Catalog loaded: sport=%s buckets=%d categories=%d kpis=%d",
            sport_id,
            len(catalog.buckets),
            len(catalog.categories),
            len(catalog.kpis),
        )

        bucket_key, kpi_defs = catalog.select_for_position(position_key)
        if not kpi_defs:
            logger.info(
                "No KPIs for player=%s position=%s bucket=%s; skipping",
                player_id,
                position_key,
                bucket_key,
            )
            return None
        logger.info(
            "Selected bucket=%s kpis=%d: %s",
            bucket_key,
            len(kpi_defs),
            ", ".join(k.key for k in kpi_defs),
        )

        evidence = await self._evidence.latest_scout_touched(
            player_id, club_id, self._evidence_limit
        )
        if not evidence:
            logger.info("No scout-touched reviews for player=%s; skipping", player_id)
            return None

        version = evidence_version(evidence)
        logger.info(
            "Evidence: player=%s reviews=%d version=%s clips=[%s]",
            player_id,
            len(evidence),
            version,
            ", ".join(str(e.clip_id) for e in evidence),
        )

        logger.info(
            "Scoring %d KPIs from %d reviews: player=%s model=%s",
            len(kpi_defs),
            len(evidence),
            player_id,
            self._model_name,
        )
        scores = await self._scorer.score(kpi_defs, evidence, language)
        observed = sum(1 for s in scores if s.score is not None)
        logger.info(
            "Scored player=%s: observed=%d not_observed=%d",
            player_id,
            observed,
            len(scores) - observed,
        )

        await self._kpis.upsert_ai_scores(
            club_id=club_id,
            player_id=player_id,
            scores=scores,
            evidence_version=version,
            model=self._model_name,
            computed_at=datetime.now(UTC),
        )
        logger.info(
            "KPI recompute done: player=%s rows=%d observed=%d version=%s",
            player_id,
            len(scores),
            observed,
            version,
        )
        return len(scores)
