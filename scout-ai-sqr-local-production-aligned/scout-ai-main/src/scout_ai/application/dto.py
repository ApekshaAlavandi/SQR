from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from scout_ai.domain.entities import Language


@dataclass(slots=True)
class DraftRequested:
    """The ``ai.draft.requested`` event, mapped to primitives for the use case."""

    clip_id: UUID
    club_id: UUID
    sport_id: UUID | None = None
    player_id: UUID | None = None
    language: Language = Language.english


@dataclass(slots=True)
class RegenerateSummary:
    """The ``ai-regenerate`` event — regenerate a player's AI summary."""

    club_id: UUID
    clip_id: UUID
    player_id: UUID
    language: Language = Language.english


@dataclass(slots=True)
class KpiRecomputeRequested:
    """The ``ai-kpi-recompute`` event — recompute a player's AI KPI scores."""

    club_id: UUID
    sport_id: UUID  # club_sports.id for the tenant sport
    player_id: UUID
    language: Language = Language.english
