from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from scout_ai.adapters.ai.evidence_analyzer import (
    GeminiEvidenceVideoAnalyzer,
    LocalGeminiEvidenceVideoAnalyzer,
)
from scout_ai.adapters.ai.gemini_client import build_gemini_client, build_local_gemini_client
from scout_ai.adapters.ai.kpi_scorer import GeminiKpiScorer
from scout_ai.adapters.ai.local_video_analyzer import LocalGeminiVideoAnalyzer
from scout_ai.adapters.ai.player_summarizer import GeminiPlayerSummarizer
from scout_ai.adapters.ai.stub import (
    STUB_MODEL_NAME,
    StubKpiScorer,
    StubPlayerSummarizer,
    StubVideoAnalyzer,
)
from scout_ai.adapters.ai.video_analyzer import GeminiVideoAnalyzer
from scout_ai.adapters.db.base import create_engine, create_session_factory
from scout_ai.adapters.db.repositories.clip_analysis_repository import (
    SqlAlchemyClipAnalysisRepository,
)
from scout_ai.adapters.db.repositories.clip_review_evidence_repository import (
    SqlAlchemyClipReviewEvidenceRepository,
)
from scout_ai.adapters.db.repositories.clip_review_repository import (
    SqlAlchemyClipReviewRepository,
)
from scout_ai.adapters.db.repositories.clips_repository import SqlAlchemyClipRepository
from scout_ai.adapters.db.repositories.player_kpis_repository import (
    SqlAlchemyPlayerKpisRepository,
)
from scout_ai.adapters.db.repositories.player_repository import (
    SqlAlchemyPlayerRepository,
)
from scout_ai.adapters.db.repositories.player_reviews_repository import (
    SqlAlchemyPlayerReviewsRepository,
)
from scout_ai.adapters.db.repositories.player_stats_repository import (
    SqlAlchemyPlayerStatsRepository,
)
from scout_ai.adapters.db.repositories.sport_catalog_repository import (
    SqlAlchemySportCatalogRepository,
)
from scout_ai.adapters.db.repositories.video_assets_repository import (
    SqlAlchemyVideoAssetsRepository,
)
from scout_ai.application.use_cases.generate_clip_review import (
    GenerateClipReviewUseCase,
)
from scout_ai.application.use_cases.generate_player_summary import (
    GeneratePlayerSummaryUseCase,
)
from scout_ai.application.use_cases.recompute_player_kpis import (
    RecomputePlayerKpisUseCase,
)
from scout_ai.application.use_cases.resolve_clip_analysis_input import (
    ResolveClipAnalysisInputUseCase,
)
from scout_ai.config import Settings
from scout_ai.domain.ports import (
    ClipReviewRepository,
    KpiScorer,
    PlayerSummarizer,
    VideoAnalyzer,
)


@dataclass
class AppContainer:
    """Process-wide singletons assembled once at startup."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    video_analyzer: VideoAnalyzer
    player_summarizer: PlayerSummarizer
    kpi_scorer: KpiScorer

    @classmethod
    def build(cls, settings: Settings) -> AppContainer:
        engine = create_engine(settings.database_url, echo=settings.env == "local")
        gemini = build_gemini_client(settings.gcp_project_id, settings.vertex_location)
        return cls(
            settings=settings,
            engine=engine,
            session_factory=create_session_factory(engine),
            video_analyzer=(
                GeminiEvidenceVideoAnalyzer(gemini, settings.gemini_model)
                if settings.analysis_mode == "evidence"
                else GeminiVideoAnalyzer(gemini, settings.gemini_model)
            ),
            player_summarizer=GeminiPlayerSummarizer(gemini, settings.gemini_model),
            kpi_scorer=GeminiKpiScorer(gemini, settings.gemini_model),
        )

    @classmethod
    def build_local(cls, settings: Settings) -> AppContainer:
        """Wire the app for local mode: local DB, local video files, optional stub AI.

        No Pub/Sub and no GCS. ``gs://<bucket>/<key>`` is mapped to
        ``<local_media_dir>/<bucket>/<key>``, so empty bucket names get local defaults
        (``processed`` / ``tracking``) instead of cloud ones.
        """
        settings = settings.model_copy(
            update={
                "gcs_processed_bucket": settings.gcs_processed_bucket or "processed",
                "gcs_tracking_bucket": settings.gcs_tracking_bucket or "tracking",
            }
        )
        engine = create_engine(settings.database_url, echo=False)

        analyzer: VideoAnalyzer
        summarizer: PlayerSummarizer
        scorer: KpiScorer
        if settings.ai_backend == "gemini":
            gemini = build_local_gemini_client(
                api_key=settings.gemini_api_key,
                project_id=settings.gcp_project_id,
                location=settings.vertex_location,
            )
            local_cls = (
                LocalGeminiEvidenceVideoAnalyzer
                if settings.analysis_mode == "evidence"
                else LocalGeminiVideoAnalyzer
            )
            analyzer = local_cls(gemini, settings.gemini_model, settings.local_media_dir)
            summarizer = GeminiPlayerSummarizer(gemini, settings.gemini_model)
            scorer = GeminiKpiScorer(gemini, settings.gemini_model)
        else:
            # Models are recorded on rows as ``gemini_model``; label stub runs honestly.
            settings = settings.model_copy(update={"gemini_model": STUB_MODEL_NAME})
            analyzer = StubVideoAnalyzer(
                settings.local_media_dir, with_evidence=settings.analysis_mode == "evidence"
            )
            summarizer = StubPlayerSummarizer()
            scorer = StubKpiScorer()

        return cls(
            settings=settings,
            engine=engine,
            session_factory=create_session_factory(engine),
            video_analyzer=analyzer,
            player_summarizer=summarizer,
            kpi_scorer=scorer,
        )

    async def dispose(self) -> None:
        await self.engine.dispose()

    def clip_review_repository(self, session: AsyncSession) -> ClipReviewRepository:
        return SqlAlchemyClipReviewRepository(session)

    def resolve_input_use_case(self, session: AsyncSession) -> ResolveClipAnalysisInputUseCase:
        return ResolveClipAnalysisInputUseCase(
            clips=SqlAlchemyClipRepository(session),
            video_assets=SqlAlchemyVideoAssetsRepository(session),
            processed_bucket=self.settings.gcs_processed_bucket,
            tracking_bucket=self.settings.gcs_tracking_bucket,
        )

    def generate_review_use_case(self, session: AsyncSession) -> GenerateClipReviewUseCase:
        return GenerateClipReviewUseCase(
            analyzer=self.video_analyzer,
            reviews=SqlAlchemyClipReviewRepository(session),
            model_name=self.settings.gemini_model,
            analyses=SqlAlchemyClipAnalysisRepository(session),
        )

    def generate_player_summary_use_case(
        self, session: AsyncSession
    ) -> GeneratePlayerSummaryUseCase:
        return GeneratePlayerSummaryUseCase(
            reviews=SqlAlchemyPlayerReviewsRepository(session),
            summarizer=self.player_summarizer,
            player_stats=SqlAlchemyPlayerStatsRepository(session),
            review_limit=self.settings.player_summary_review_limit,
        )

    def recompute_player_kpis_use_case(
        self, session: AsyncSession
    ) -> RecomputePlayerKpisUseCase:
        return RecomputePlayerKpisUseCase(
            players=SqlAlchemyPlayerRepository(session),
            catalog=SqlAlchemySportCatalogRepository(session),
            evidence=SqlAlchemyClipReviewEvidenceRepository(session),
            scorer=self.kpi_scorer,
            kpis=SqlAlchemyPlayerKpisRepository(session),
            model_name=self.settings.gemini_model,
            evidence_limit=self.settings.kpi_evidence_limit,
        )
