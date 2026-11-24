from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from uuid import UUID

from scout_ai.domain.entities import Language
from scout_ai.domain.exceptions import InvalidGcsUriError


@dataclass(frozen=True, slots=True)
class GcsUri:
    """A parsed Google Cloud Storage location (``gs://bucket/object``)."""

    bucket: str
    object_name: str

    @classmethod
    def parse(cls, raw: str) -> GcsUri:
        if not raw or not raw.startswith("gs://"):
            raise InvalidGcsUriError(f"Expected a gs:// URI, got: {raw!r}")
        without_scheme = raw[len("gs://") :]
        bucket, _, object_name = without_scheme.partition("/")
        if not bucket or not object_name:
            raise InvalidGcsUriError(f"URI must include a bucket and object: {raw!r}")
        return cls(bucket=bucket, object_name=object_name)

    @property
    def uri(self) -> str:
        return f"gs://{self.bucket}/{self.object_name}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.uri


@dataclass(frozen=True, slots=True)
class ClipAnalysisInput:
    """Everything needed to run the AI analysis for one clip.

    Assembled by ``ResolveClipAnalysisInputUseCase`` from the ``video_assets`` and
    ``clips`` repositories; consumed by the Gemini analyzer and the review use case.
    """

    clip_id: UUID
    club_id: UUID
    video_uri: GcsUri  # gs://<bucket>/<key>: processed clip, or the annotated clip when is_tracked
    action_tag_key: str  # the action/skill being scouted in this clip
    ident_text: str | None  # free-text identifier / context for the player
    player_id: UUID | None = None  # denormalized onto clip_reviews for the summary flow
    match_date: date | None = None  # from the clip; denormalized onto clip_reviews
    language: Language = Language.english  # target language for the generated review
    is_tracked: bool = False  # True => video_uri is the annotated (box-tracked) clip
