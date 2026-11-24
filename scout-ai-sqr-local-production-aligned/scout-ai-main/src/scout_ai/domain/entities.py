from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for the annotation
    from scout_ai.domain.clip_evidence import ClipEvidence


class Language(str, Enum):
    """Output language for AI-generated text (clip reviews, player summaries, KPI rationale).

    Carried on every flow's event. Per the upstream contract a missing/blank ``language``
    (e.g. from the deprecated regenerate-ai-draft publisher) — and anything unrecognised —
    is treated as English, so ``from_wire`` never fails.
    """

    english = "en"
    arabic = "ar"

    @classmethod
    def from_wire(cls, raw: str | None) -> Language:
        """Normalize a wire ``language`` value; empty/unknown -> English."""
        if not raw:
            return cls.english
        token = raw.strip().lower()
        return {
            "en": cls.english,
            "english": cls.english,
            "ar": cls.arabic,
            "arabic": cls.arabic,
        }.get(token, cls.english)

    @property
    def label(self) -> str:
        """English name of the language, used inside model prompts."""
        return {"en": "English", "ar": "Arabic"}[self.value]


class ClipStatus(str, Enum):
    pending_upload = "pending_upload"
    uploading = "uploading"
    processing = "processing"
    ready = "ready"
    failed = "failed"


@dataclass(frozen=True)
class ClipDetails:
    id: UUID
    opponent_name: str
    match_date: date
    action_tag_key: str
    minute: int | None
    ident_text: str | None
    upload_status: ClipStatus


@dataclass(frozen=True)
class VideoAssetsDetails:
    id: UUID
    clip_id: UUID
    processed_object_key: str | None
    # Non-null when the clip was tracked (subject marked with a box); points at the
    # annotated video in the tracking bucket. Its presence is the "tracked" signal.
    tracking_object_key: str | None = None


class AIGenerationStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class ReviewStatus(str, Enum):
    pending = "pending"
    edited_by_scout = "edited_by_scout"
    accepted_ai_draft = "accepted_ai_draft"


class AIClipVerdict(str, Enum):
    sign = "sign"
    monitor = "monitor"
    pass_ = "pass"


@dataclass
class ClipReview:
    id: UUID
    clip_id: UUID
    club_id: UUID
    player_id: UUID | None
    match_date: date | None
    ai_review: str | None
    ai_score: Decimal | None
    ai_verdict: str | None
    human_review: str | None
    human_score: Decimal | None
    human_verdict: str | None
    ai_generation_status: AIGenerationStatus
    review_status: ReviewStatus
    ai_model: str | None
    reviewed_by: UUID | None
    reviewed_at: datetime | None
    created_at: datetime
    updated_at: datetime
    # Total tokens the AI draft cost (prompt + output + thoughts); None if the model
    # reported no usage. Only the total is stored — the breakdown is log-only.
    total_tokens: int | None = None


# --------------------------------------------------------------------------- #
# AI analysis results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts the LLM reported for one analysis call.

    Every field is optional: ``usage_metadata`` can be missing on a failed/partial
    response, and individual counts come back ``None`` on models that don't report them
    (``thoughts_tokens`` on non-thinking models, ``cached_tokens`` when nothing was
    cached). Only ``total_tokens`` is persisted (``clip_reviews.ai_total_tokens``); the
    full breakdown is logged for per-clip cost visibility.
    """

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    cached_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class VideoAnalysis:
    review: str
    score: float
    verdict: str
    tokens: TokenUsage | None = None
    # Structured evidence (evidence-first prompt). None => legacy prompt / non-football
    # clip: nothing is written to the clip_analyses family of tables.
    evidence: ClipEvidence | None = None


@dataclass(frozen=True, slots=True)
class PlayerReviewItem:
    """One reviewed clip feeding the player-summary regeneration flow.

    Sourced from ``clip_reviews`` for a player, restricted to scout-touched
    statuses (edited_by_scout / approved), ordered by match date.
    """

    match_date: date | None
    review_status: ReviewStatus
    ai_review: str | None
    ai_score: Decimal | None
    ai_verdict: str | None
    human_review: str | None
    human_score: Decimal | None
    human_verdict: str | None


@dataclass(slots=True)
class PlayerSummary:
    """AI-generated holistic summary for a player (the only output we persist)."""

    summary: str


@dataclass(frozen=True, slots=True)
class ClipReviewEvidence:
    """One scout-touched clip review feeding the KPI-recompute flow.

    Read from ``clip_reviews`` for a player (statuses edited_by_scout /
    accepted_ai_draft), newest first. Separate from ``PlayerReviewItem`` on purpose:
    KPI scoring needs ``clip_id`` and ``updated_at`` to fingerprint the evidence set
    (see ``scout_ai.domain.kpi.evidence_version``), which the summary entity omits.
    """

    clip_id: UUID
    match_date: date | None
    review_status: ReviewStatus
    ai_review: str | None
    ai_score: Decimal | None
    ai_verdict: str | None
    human_review: str | None
    human_score: Decimal | None
    human_verdict: str | None
    updated_at: datetime
