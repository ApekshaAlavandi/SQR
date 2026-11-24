from __future__ import annotations

import logging
from decimal import Decimal

from scout_ai.domain.entities import VideoAnalysis
from scout_ai.domain.clip_taxonomy import TAXONOMY_VERSION
from scout_ai.domain.ports import ClipAnalysisRepository, ClipReviewRepository, VideoAnalyzer
from scout_ai.domain.value_objects import ClipAnalysisInput

logger = logging.getLogger(__name__)


class GenerateClipReviewUseCase:
    def __init__(
        self,
        *,
        analyzer: VideoAnalyzer,
        reviews: ClipReviewRepository,
        model_name: str,
        analyses: ClipAnalysisRepository | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._reviews = reviews
        self._analyses = analyses
        self._model_name = model_name

    async def execute(self, data: ClipAnalysisInput) -> VideoAnalysis:
        logger.info(
            "Generating clip review: clip=%s club=%s player=%s",
            data.clip_id,
            data.club_id,
            data.player_id,
        )

        # Create the row if absent and mark it processing, before the Gemini call.
        # The caller is responsible for flipping it to 'failed' if anything below
        # raises (it owns a clean session for that); see _build_draft_processor.
        await self._reviews.start_ai_generation(
            data.clip_id, data.club_id, data.player_id, data.match_date
        )
        analysis = await self._analyzer.analyze(data)

        await self._reviews.save_ai_review(
            clip_id=data.clip_id,
            review=analysis.review,
            score=Decimal(str(analysis.score)),
            verdict=analysis.verdict,
            model=self._model_name,
            player_id=data.player_id,
            match_date=data.match_date,
            total_tokens=analysis.tokens.total_tokens if analysis.tokens else None,
        )
        logger.info(
            "Saved AI review: clip=%s verdict=%s score=%s",
            data.clip_id,
            analysis.verdict,
            analysis.score,
        )

        await self._save_evidence(data, analysis)
        return analysis

    async def _save_evidence(self, data: ClipAnalysisInput, analysis: VideoAnalysis) -> None:
        """Persist the structured evidence for the B screens.

        Deliberately NON-FATAL: the review above is already saved and is what the existing
        screens read. A failure here must not flip the review to 'failed' nor trigger a
        redelivery (= a second, paid Gemini call), so it is logged loudly and swallowed.
        """
        if self._analyses is None or analysis.evidence is None:
            return
        for w in analysis.evidence.warnings:
            logger.warning("Evidence sanitizer: clip=%s %s", data.clip_id, w)
        try:
            analysis_id = await self._analyses.save(
                clip_id=data.clip_id,
                club_id=data.club_id,
                evidence=analysis.evidence,
                rating=Decimal(str(analysis.score)),
                model_version=self._model_name,
                taxonomy_version=TAXONOMY_VERSION,
            )
            logger.info(
                "Saved clip analysis: clip=%s analysis=%s obs=%d touches=%d",
                data.clip_id,
                analysis_id,
                len(analysis.evidence.observations),
                len(analysis.evidence.touches),
            )
        except Exception:
            logger.exception("Saving clip analysis evidence FAILED (review kept): clip=%s", data.clip_id)
