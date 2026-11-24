from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from scout_ai.application.dto import (
    DraftRequested,
    KpiRecomputeRequested,
    RegenerateSummary,
)
from scout_ai.domain.entities import Language

# The wire ``language`` is a free-form string ("en"/"ar", or absent). Keep it as a plain
# optional string on every event and let ``Language.from_wire`` normalize it (blank /
# unknown -> English) at the ``to_dto`` boundary, so the rest of the app only sees the enum.


class DraftRequestedEvent(BaseModel):
    """The JSON payload published on the topic.

    Unknown fields are ignored so additive changes upstream don't break us.
    """

    model_config = ConfigDict(extra="ignore")

    # eventType: str  # noqa: N815 - matches the wire field name
    clip_id: UUID
    club_id: UUID
    sport_id: UUID | None = None
    player_id: UUID | None = None
    language: str | None = None

    def to_dto(self) -> DraftRequested:
        return DraftRequested(
            clip_id=self.clip_id,
            club_id=self.club_id,
            sport_id=self.sport_id,
            player_id=self.player_id,
            language=Language.from_wire(self.language),
        )


class RegenerateSummaryEvent(BaseModel):
    """Payload on the ``ai-regenerate`` topic."""

    model_config = ConfigDict(extra="ignore")

    club_id: UUID
    clip_id: UUID
    player_id: UUID
    language: str | None = None

    def to_dto(self) -> RegenerateSummary:
        return RegenerateSummary(
            club_id=self.club_id,
            clip_id=self.clip_id,
            player_id=self.player_id,
            language=Language.from_wire(self.language),
        )


class KpiRecomputeEvent(BaseModel):
    """Payload on the ``ai-kpi-recompute`` topic."""

    model_config = ConfigDict(extra="ignore")

    club_id: UUID
    sport_id: UUID  # club_sports.id for the tenant sport
    player_id: UUID
    language: str | None = None

    def to_dto(self) -> KpiRecomputeRequested:
        return KpiRecomputeRequested(
            club_id=self.club_id,
            sport_id=self.sport_id,
            player_id=self.player_id,
            language=Language.from_wire(self.language),
        )
