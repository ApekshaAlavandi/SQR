"""Gemini analyzer for the evidence-first prompt.

Same transport as ``GeminiVideoAnalyzer`` (video part, token accounting, review text
assembly) but with the evidence prompt and a richer, enum-constrained response schema.
The report fields keep feeding ``clip_reviews`` exactly as before; the evidence records
are sanitized and handed back on ``VideoAnalysis.evidence`` for ``clip_analyses`` & co.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from scout_ai.adapters.ai.evidence_prompt import build_evidence_prompt
from scout_ai.adapters.ai.local_video_analyzer import build_inline_video_part
from scout_ai.adapters.ai.video_analyzer import AIClipContentCheck, GeminiVideoAnalyzer
from scout_ai.domain import clip_evidence as ce
from scout_ai.domain import clip_taxonomy as tx
from scout_ai.domain.entities import AIClipVerdict, VideoAnalysis
from scout_ai.domain.value_objects import ClipAnalysisInput

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Response schema (field ORDER is the generation order: records first, report after)
# NOTE: no defaults on purpose - Gemini's response_schema rejects default values.
# --------------------------------------------------------------------------- #
class _PresenceSpanS(BaseModel):
    t0: float
    t1: float
    state: tx.PresenceState


class _ObservationS(BaseModel):
    t0: float
    t1: float
    phase: tx.Phase
    zone: int
    action: tx.ObsAction
    quality: tx.Quality
    outcome: tx.Outcome
    significance: int
    attributes: list[tx.Attribute]
    confidence: float
    text_en: str
    text_ar: str


class _TallyS(BaseModel):
    action: tx.TallyAction
    attempted: int
    successful: int
    agreement: float


class _CountS(BaseModel):
    action: tx.CountOnlyAction
    count: int


class _ShotS(BaseModel):
    t: float
    zone: int
    foot: tx.Foot
    outcome: tx.ShotOutcome
    situation: tx.ShotSituation


class _TouchS(BaseModel):
    t: float
    zone: int


class _IdentS(BaseModel):
    confidence: float
    role: str
    cues: list[str]


class _AreaS(BaseModel):
    area: tx.Area
    confidence: float


class _EvidenceAnalysisSchema(BaseModel):
    content_check: AIClipContentCheck
    clip_duration_s: float
    presence_spans: list[_PresenceSpanS]
    observations: list[_ObservationS]
    action_tallies: list[_TallyS]
    counts_only: list[_CountS]
    shots: list[_ShotS]
    touches: list[_TouchS]
    identification: _IdentS | None
    confidence_by_area: list[_AreaS]
    not_observed: list[tx.NotObserved]
    detailed_analysis: str
    summary: str
    prose_en: str
    prose_ar: str
    score: float = Field(ge=0, le=10)
    verdict: AIClipVerdict

    @field_validator("verdict", mode="before")
    @classmethod
    def _coerce_verdict(cls, v: object) -> AIClipVerdict:
        if isinstance(v, AIClipVerdict):
            return v
        if isinstance(v, str):
            try:
                return AIClipVerdict(v.strip().lower())
            except ValueError:
                return AIClipVerdict.monitor
        return AIClipVerdict.monitor

    @field_validator("content_check", mode="before")
    @classmethod
    def _coerce_content_check(cls, v: object) -> AIClipContentCheck:
        if isinstance(v, AIClipContentCheck):
            return v
        if isinstance(v, str):
            try:
                return AIClipContentCheck(v.strip().lower())
            except ValueError:
                return AIClipContentCheck.unclear
        return AIClipContentCheck.unclear


def to_evidence(r: _EvidenceAnalysisSchema) -> ce.ClipEvidence | None:
    """Schema -> domain evidence, sanitized. None when the clip is not analyzable."""
    if r.content_check != AIClipContentCheck.football:
        return None
    raw = ce.ClipEvidence(
        duration_s=r.clip_duration_s,
        presence_spans=[ce.PresenceSpan(s.t0, s.t1, s.state.value) for s in r.presence_spans],
        observations=[
            ce.Observation(
                o.t0, o.t1, o.phase.value, o.zone, o.action.value, o.quality.value,
                o.outcome.value, o.significance, tuple(a.value for a in o.attributes),
                o.confidence, o.text_en, o.text_ar,
            )
            for o in r.observations
        ],
        action_tallies=[
            ce.ActionTally(t.action.value, t.attempted, t.successful, t.agreement)
            for t in r.action_tallies
        ],
        counts_only=[ce.CountOnly(c.action.value, c.count) for c in r.counts_only],
        shots=[
            ce.Shot(int(s.t), s.zone, s.foot.value, s.outcome.value, s.situation.value)
            for s in r.shots
        ],
        touches=[ce.Touch(int(t.t), t.zone) for t in r.touches],
        identification=(
            {"confidence": r.identification.confidence, "role": r.identification.role,
             "cues": list(r.identification.cues)}
            if r.identification else None
        ),
        confidence_by_area={a.area.value: a.confidence for a in r.confidence_by_area},
        not_observed=[n.value for n in r.not_observed],
        prose_en=r.prose_en,
        prose_ar=r.prose_ar,
    )
    if raw.identification is None:  # football clip without identification: not persistable
        logger.warning("Evidence: football clip returned no identification; not persisting")
        return None
    return ce.sanitize_evidence(raw)


class GeminiEvidenceVideoAnalyzer(GeminiVideoAnalyzer):
    """Evidence-first analysis; review text path is unchanged."""

    async def analyze(self, data: ClipAnalysisInput) -> VideoAnalysis:
        video_part = await self._video_part(data)
        instructions = build_evidence_prompt(data)
        config = types.GenerateContentConfig(
            temperature=0.0,
            top_p=0.1,
            response_mime_type="application/json",
            response_schema=_EvidenceAnalysisSchema,
        )
        logger.info("Gemini evidence analysis start: clip=%s model=%s", data.clip_id, self._model)
        started = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self._model, contents=[video_part, instructions], config=config
        )
        logger.info(
            "Gemini evidence analysis done: clip=%s took=%.2fs",
            data.clip_id, time.monotonic() - started,
        )
        tokens = self._extract_token_usage(response)
        logger.info(
            "Gemini token usage: clip=%s prompt=%s output=%s thoughts=%s cached=%s total=%s",
            data.clip_id, tokens.prompt_tokens, tokens.output_tokens,
            tokens.thoughts_tokens, tokens.cached_tokens, tokens.total_tokens,
        )

        result: _EvidenceAnalysisSchema | None = response.parsed
        if result is None:
            result = _EvidenceAnalysisSchema.model_validate_json(response.text)

        # _build_review only reads content_check / summary / detailed_analysis, so it works
        # on this schema unchanged -> clip_reviews.ai_review is built exactly as before.
        review = self._build_review(result)  # type: ignore[arg-type]
        evidence = to_evidence(result)
        if result.content_check == AIClipContentCheck.football and evidence is None:
            logger.warning("Evidence not persistable for clip=%s (kept review only)", data.clip_id)
        return VideoAnalysis(
            review=review,
            score=float(result.score or 0),
            verdict=result.verdict.value,
            tokens=tokens,
            evidence=evidence,
        )


class LocalGeminiEvidenceVideoAnalyzer(GeminiEvidenceVideoAnalyzer):
    """Local mode: same as production, video read from disk instead of GCS."""

    def __init__(self, client: genai.Client, model: str, media_dir: str | Path) -> None:
        super().__init__(client, model)
        self._media_dir = media_dir

    async def _video_part(self, data: ClipAnalysisInput) -> types.Part:
        return await build_inline_video_part(self._media_dir, data)
