from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from scout_ai.domain.clip_evidence import ClipEvidence
from scout_ai.domain.entities import (
    AIGenerationStatus,
    ClipDetails,
    ClipReviewEvidence,
    Language,
    PlayerReviewItem,
    PlayerSummary,
    VideoAnalysis,
    VideoAssetsDetails,
)
from scout_ai.domain.kpi import KpiDefinition, KpiScore, SportCatalog
from scout_ai.domain.value_objects import ClipAnalysisInput


class ClipRepository(ABC):
    """Persistence boundary for clips."""

    @abstractmethod
    async def get_details(self, clip_id: UUID) -> ClipDetails:
        """Return clip details or raise ClipNotFoundError."""


class VideoAssetsRepository(ABC):
    """Persistence boundary for video assets."""

    @abstractmethod
    async def get_details(self, clip_id: UUID) -> VideoAssetsDetails:
        """Return video assets details or raise VideoAssetsNotFoundError."""


class ClipReviewRepository(ABC):
    @abstractmethod
    async def start_ai_generation(
        self,
        clip_id: UUID,
        club_id: UUID,
        player_id: UUID | None = None,
        match_date: date | None = None,
    ) -> None:
        """Upsert the clip's row into ``processing``, creating it if it's absent."""

    @abstractmethod
    async def get_status(self, clip_id: UUID) -> AIGenerationStatus | None:
        """Current AI generation status for a clip, or ``None`` if no row exists.

        Used as the idempotency check by the Pub/Sub consumer.
        """

    @abstractmethod
    async def mark_ai_generation_status(
        self, clip_id: UUID, status: AIGenerationStatus
    ) -> None: ...

    @abstractmethod
    async def save_ai_review(
        self,
        clip_id: UUID,
        review: str,
        score: Decimal,
        verdict: str,
        model: str,
        player_id: UUID | None = None,
        match_date: date | None = None,
        total_tokens: int | None = None,
    ) -> None: ...


class ClipAnalysisRepository(ABC):
    """Writes the structured evidence behind the player-page B screens."""

    @abstractmethod
    async def save(
        self,
        *,
        clip_id: UUID,
        club_id: UUID,
        evidence: ClipEvidence,
        rating: Decimal,
        model_version: str,
        taxonomy_version: str,
    ) -> UUID:
        """Publish a new analysis version for the clip (older ones become 'superseded').

        All rows (analysis, presence spans, observations, tallies) are written in ONE
        transaction. Returns the new ``clip_analyses.id``.
        """


class VideoAnalyzer(ABC):
    """Runs the AI analysis over a clip's downsized video (read from GCS)."""

    @abstractmethod
    async def analyze(self, data: ClipAnalysisInput) -> VideoAnalysis: ...


# --------------------------------------------------------------------------- #
# Player-summary regeneration flow (topic: ai-regenerate)
# --------------------------------------------------------------------------- #
class PlayerReviewsRepository(ABC):
    """Reads a player's recent scout-touched reviews from ``clip_reviews``."""

    @abstractmethod
    async def latest_reviewed(
        self, player_id: UUID, club_id: UUID, limit: int
    ) -> list[PlayerReviewItem]:
        """Latest ``limit`` reviews (status edited_by_scout/approved), newest first."""


class PlayerSummarizer(ABC):
    """Produces a holistic AI summary for a player from their reviews."""

    @abstractmethod
    async def summarize(
        self, items: list[PlayerReviewItem], language: Language
    ) -> PlayerSummary: ...


class PlayerStatsRepository(ABC):
    """Writes the AI summary onto the player's latest ``player_stats`` row."""

    @abstractmethod
    async def set_latest_ai_summary(self, player_id: UUID, summary: str) -> None:
        """Upsert ``ai_summary`` for the player (insert the row if none exists)."""


# --------------------------------------------------------------------------- #
# KPI recompute flow (topic: ai-kpi-recompute)
# --------------------------------------------------------------------------- #
class PlayerRepository(ABC):
    """Reads player attributes needed to pick a KPI set."""

    @abstractmethod
    async def get_position(self, player_id: UUID, club_id: UUID) -> str:
        """The player's ``position_key`` or raise ``PlayerNotFoundError``."""


class SportCatalogRepository(ABC):
    """Reads the tenant sport's KPI catalog (``club_sports`` -> ``sport_catalog``)."""

    @abstractmethod
    async def get(self, sport_id: UUID, club_id: UUID) -> SportCatalog:
        """The parsed catalog for the club's sport, or raise ``SportCatalogNotFoundError``.

        ``sport_id`` is the ``club_sports.id`` carried on the event.
        """


class ClipReviewEvidenceRepository(ABC):
    """Reads a player's recent scout-touched clip reviews as KPI evidence."""

    @abstractmethod
    async def latest_scout_touched(
        self, player_id: UUID, club_id: UUID, limit: int
    ) -> list[ClipReviewEvidence]:
        """Latest ``limit`` scout-touched reviews (edited_by_scout/accepted), newest first."""


class PlayerKpisRepository(ABC):
    """Writes the AI layer of ``player_kpis`` (never the scout review layer)."""

    @abstractmethod
    async def upsert_ai_scores(
        self,
        *,
        club_id: UUID,
        player_id: UUID,
        scores: list[KpiScore],
        evidence_version: str,
        model: str,
        computed_at: datetime,
    ) -> None:
        """Upsert one row per KPI on ``(player_id, kpi_key)``, AI columns only."""


class KpiScorer(ABC):
    """Scores a catalog KPI set from a player's clip-review evidence."""

    @abstractmethod
    async def score(
        self,
        kpis: list[KpiDefinition],
        evidence: list[ClipReviewEvidence],
        language: Language,
    ) -> list[KpiScore]:
        """One ``KpiScore`` per input KPI (score ``None`` = not observed).

        ``language`` sets the language of each KPI ``rationale`` (the numeric scores and
        ``kpi_key`` are language-independent).
        """
