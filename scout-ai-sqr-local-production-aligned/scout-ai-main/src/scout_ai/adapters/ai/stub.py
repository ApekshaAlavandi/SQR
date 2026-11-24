"""Offline, deterministic AI implementations for local mode.

They satisfy the same ports as the Gemini adapters, so every use case and flow runs
unchanged — but nothing leaves the machine and no API key is needed. Output is derived
from the input (a stable hash), so repeated runs are reproducible, and it is clearly
labelled as stub output so it can't be mistaken for a real analysis.
"""

from __future__ import annotations

import hashlib
import logging

from scout_ai.adapters.ai.local_media import resolve_local_path
from scout_ai.domain import clip_evidence as ce
from scout_ai.domain.entities import (
    AIClipVerdict,
    ClipReviewEvidence,
    Language,
    PlayerReviewItem,
    PlayerSummary,
    TokenUsage,
    VideoAnalysis,
)
from scout_ai.domain.kpi import KpiDefinition, KpiScore
from scout_ai.domain.ports import KpiScorer, PlayerSummarizer, VideoAnalyzer
from scout_ai.domain.value_objects import ClipAnalysisInput

logger = logging.getLogger(__name__)

STUB_MODEL_NAME = "stub-local"
_VERDICTS = (AIClipVerdict.pass_, AIClipVerdict.monitor, AIClipVerdict.sign)


def _unit(seed: str) -> float:
    """Stable pseudo-random float in [0, 1) derived from ``seed``."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _tag(language: Language) -> str:
    return "[STUB — no AI call]" if language is Language.english else "[STUB — no AI call] (ar)"


def _stub_evidence(seed: str) -> ce.ClipEvidence:
    """Small, valid, deterministic evidence set so local runs fill every B-screen table."""
    z = 7 + int(_unit(seed + ":zone") * 6)  # zones 7..12 (middle third)
    raw = ce.ClipEvidence(
        duration_s=120,
        presence_spans=[
            ce.PresenceSpan(0, 30, "in"),
            ce.PresenceSpan(30, 50, "out"),
            ce.PresenceSpan(50, 120, "in"),
        ],
        observations=[
            ce.Observation(5, 8, "oa", z, "pass", "good", "success", 2, ("passing_short",), 0.9,
                           "[STUB] Completed a short pass.", "[STUB] أكمل تمريرة قصيرة."),
            ce.Observation(60, 63, "od", z, "positioning", "neutral", "neutral", 1,
                           ("positioning_defensive",), 0.85,
                           "[STUB] Held a central position.", "[STUB] حافظ على موقع مركزي."),
            ce.Observation(70, 74, "ot", z, "carry", "good", "success", 3, ("carrying",), 0.8,
                           "[STUB] Carried the ball forward.", "[STUB] قاد الكرة إلى الأمام."),
        ],
        action_tallies=[
            ce.ActionTally("pass", 1, 1, 0.9),
            ce.ActionTally("carry", 1, 1, 0.8),
        ],
        counts_only=[ce.CountOnly("recoveries", 1)],
        touches=[ce.Touch(5, z), ce.Touch(6, z), ce.Touch(71, z), ce.Touch(73, z)],
        identification={"confidence": 0.9, "role": "central midfielder", "cues": []},
        confidence_by_area={"passing": 0.55, "carrying": 0.55, "positional_discipline": 0.55},
        not_observed=["shooting", "tackling", "aerial_duels"],
        prose_en="At 00:05 the player completed a short pass. At 01:10 the player carried the ball forward.",
        prose_ar="عند 00:05 أكمل اللاعب تمريرة قصيرة. عند 01:10 قاد اللاعب الكرة إلى الأمام.",
    )
    return ce.sanitize_evidence(raw)  # type: ignore[return-value]


class StubVideoAnalyzer(VideoAnalyzer):
    def __init__(self, media_dir: str, with_evidence: bool = True) -> None:
        self._media_dir = media_dir
        self._with_evidence = with_evidence

    async def analyze(self, data: ClipAnalysisInput) -> VideoAnalysis:
        path = resolve_local_path(self._media_dir, data.video_uri)
        found = path.is_file()
        if not found:
            logger.info("Stub analyzer: no video file at %s (continuing; stub needs none)", path)

        u = _unit(f"{data.clip_id}:{data.action_tag_key}:{data.is_tracked}")
        score = round(4.0 + u * 5.0, 1)  # 4.0 – 9.0
        verdict = _VERDICTS[0 if score < 5.5 else 1 if score < 7.5 else 2]
        subject = "the tracked player" if data.is_tracked else (data.ident_text or "the player")
        review = (
            f"# DETAILED ANALYSIS\n\n"
            f"{_tag(data.language)} Canned review of '{data.action_tag_key}' for {subject}.\n\n"
            f"- Video: {data.video_uri.uri} ({'found' if found else 'missing'} locally)\n"
            f"- Tracked clip: {data.is_tracked}\n"
            f"- Score {score}/10, verdict '{verdict.value}'.\n"
        )
        return VideoAnalysis(
            review=review,
            score=score,
            verdict=verdict.value,
            tokens=TokenUsage(total_tokens=0),
            evidence=(
                _stub_evidence(f"{data.clip_id}:{data.is_tracked}") if self._with_evidence else None
            ),
        )


class StubPlayerSummarizer(PlayerSummarizer):
    async def summarize(
        self, items: list[PlayerReviewItem], language: Language
    ) -> PlayerSummary:
        scores = [float(i.human_score if i.human_score is not None else i.ai_score)
                  for i in items if (i.human_score is not None or i.ai_score is not None)]
        avg = f"{sum(scores) / len(scores):.1f}" if scores else "n/a"
        return PlayerSummary(
            summary=(
                f"{_tag(language)} Summary built from {len(items)} scout-reviewed clip(s); "
                f"average score {avg}/10."
            )
        )


class StubKpiScorer(KpiScorer):
    async def score(
        self,
        kpis: list[KpiDefinition],
        evidence: list[ClipReviewEvidence],
        language: Language,
    ) -> list[KpiScore]:
        seed_base = "|".join(sorted(str(e.clip_id) for e in evidence))
        scores: list[KpiScore] = []
        for kpi in kpis:
            u = _unit(f"{seed_base}:{kpi.key}")
            # Every 5th-ish KPI is "not observed" so the NULL-score path gets exercised.
            observed = u >= 0.2
            scores.append(
                KpiScore(
                    kpi_key=kpi.key,
                    category_key=kpi.category_key,
                    score=round(3.0 + u * 6.0, 1) if observed else None,
                    rationale=(
                        f"{_tag(language)} Derived from {len(evidence)} review(s)."
                        if observed
                        else f"{_tag(language)} Not observed in the available reviews."
                    ),
                )
            )
        return scores
