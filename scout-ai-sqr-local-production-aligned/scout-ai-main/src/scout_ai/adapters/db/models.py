from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import Date, Integer, Numeric, SmallInteger, Text, Enum, func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from scout_ai.adapters.db.base import Base
from scout_ai.domain.entities import AIGenerationStatus, ReviewStatus, AIClipVerdict as ClipVerdict


class Clips(Base):
    __tablename__ = "clips"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    opponent_name: Mapped[str] = mapped_column(Text, nullable=False)
    match_date: Mapped[date] = mapped_column(Date, nullable=False)
    action_tag_key: Mapped[str] = mapped_column(Text, nullable=False)
    minute: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    ident_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_status: Mapped[str | None] = mapped_column(nullable=False)


class VideoAssets(Base):
    __tablename__ = "video_assets"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    clip_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True))
    processed_object_key: Mapped[str | None]
    thumbnail_object_key: Mapped[str | None]
    # Set by the upstream tracking pipeline once the subject player has been marked
    # with a bounding-box overlay. Non-null => the clip was tracked, and this key (the
    # annotated video) is analyzed instead of processed_object_key. Lives in its own
    # bucket (SCOUT_GCS_TRACKING_BUCKET); see ResolveClipAnalysisInputUseCase.
    tracking_object_key: Mapped[str | None]


class ClipReviewModel(Base):
    __tablename__ = "clip_reviews"

    id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True)
    clip_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False, unique=True)
    club_id: Mapped[UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)

    # Denormalized from clips so the player-summary flow reads clip_reviews alone.
    player_id: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True, index=True)
    match_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    # AI Draft
    ai_review: Mapped[str | None] = mapped_column(Text)
    ai_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    ai_verdict:  Mapped[ClipVerdict | None] = mapped_column(
    Enum(
        ClipVerdict,
        name="clip_verdict",
        create_type=False,
        # The Postgres type's labels are the enum *values* (sign/monitor/pass),
        # not the member names (pass_ != "pass"). Map by value in both directions.
        values_callable=lambda enum_cls: [m.value for m in enum_cls],
    ),
    nullable=True,
)

    # Human Final
    human_review: Mapped[str | None] = mapped_column(Text)
    human_score: Mapped[Decimal | None] = mapped_column(Numeric(4, 2))
    human_verdict:  Mapped[ClipVerdict | None] = mapped_column(
    Enum(
        ClipVerdict,
        name="clip_verdict",
        create_type=False,
        # See ai_verdict: map by enum value (DB labels are sign/monitor/pass).
        values_callable=lambda enum_cls: [m.value for m in enum_cls],
    ),
    nullable=True,
)

    ai_generation_status: Mapped[AIGenerationStatus] = mapped_column(
        Enum(
            AIGenerationStatus,
            name="ai_generation_status",
            create_type=False,
        ),
        nullable=False,
        server_default="pending",
    )

    review_status: Mapped[ReviewStatus] = mapped_column(
        Enum(
            ReviewStatus,
            name="review_status",
            create_type=False,
        ),
        nullable=False,
        server_default="pending",
    )

    ai_model: Mapped[str | None] = mapped_column(Text)
    # Total tokens the AI draft cost (prompt + output + thoughts), as reported by the
    # model's usage_metadata. Nullable: older rows and models that report no usage.
    ai_total_tokens: Mapped[int | None] = mapped_column(Integer)
    reviewed_by: Mapped[UUID | None] = mapped_column(PgUUID(as_uuid=True))
    reviewed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
