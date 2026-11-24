from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from scout_ai.domain.entities import ClipReviewEvidence, Language
from scout_ai.domain.kpi import KpiDefinition, KpiScore
from scout_ai.domain.ports import KpiScorer

logger = logging.getLogger(__name__)

# Fallback rationale for a KPI the model dropped (see _align), localized so it matches the
# language of the model-written rationales around it. Unknown languages fall back to English.
_NOT_OBSERVED = {
    Language.english: "Not assessable from the available reviews.",
    Language.arabic: "لا يمكن تقييمه من المراجعات المتاحة.",
}


def _language_line(language: Language) -> str:
    return (
        f'OUTPUT LANGUAGE (mandatory): write every "rationale" in {language.label}. This '
        f'overrides the language of the input reviews. Reuse the given "kpi_key" values '
        f'exactly, and write the numeric "score" as a plain number in Western/ASCII digits '
        f"(e.g. 7.5), never Arabic-Indic or other localized digits — it is parsed as a "
        f"number. Plain text only in the rationale, no markdown."
    )

_SYSTEM = """\
You are an elite football scouting analyst. You are given (a) a fixed list of KPIs to
score and (b) a player's most recent scout-reviewed clips — each with an AI draft review
and the scout's final review, their scores and verdicts, the review status, and the match
date (newest first).

The reviews may contain markdown (headings, tables, bullet lists). Read through it for
substance and IGNORE its formatting.

Score EACH KPI in the provided list on a 0–10 scale, weighing scout (human) reviews over
AI drafts ('accepted_ai_draft' = scout endorsed the AI view; 'edited_by_scout' = scout's
corrected view). For each KPI:
- Give a "score" from 0 to 10 (one decimal is fine) ONLY when the reviews contain real
  evidence bearing on that KPI.
- Set "score" to null when the reviews give NO basis to judge that KPI ("not observed").
  Do NOT guess or invent a middling number to fill it in — a null is correct and simply
  drops the KPI from the player's fit.
- Give a short, evidence-anchored "rationale" (1–2 sentences). For a null score, briefly
  say the reviews don't cover it.

Return a score object for EVERY kpi_key in the list and use those keys exactly. Do not add
KPIs that were not requested. Output plain text in the rationale — no markdown.
"""


class _KpiScoreItem(BaseModel):
    kpi_key: str = Field(description="The catalog KPI key being scored.")
    score: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="0–10 score, or null when the reviews give no basis to judge it.",
    )
    rationale: str = Field(description="Short, evidence-anchored justification (plain text).")


class _KpiScoresSchema(BaseModel):
    scores: list[_KpiScoreItem] = Field(description="One entry per requested KPI.")


def _format_kpis(kpis: list[KpiDefinition]) -> str:
    lines = ["KPIs to score (use these exact kpi_key values):"]
    for k in kpis:
        measures = f" — {k.measures}" if k.measures else ""
        lines.append(f"- {k.key} ({k.label}){measures}")
    return "\n".join(lines)


def _format_evidence(evidence: list[ClipReviewEvidence]) -> str:
    blocks: list[str] = []
    for i, it in enumerate(evidence, start=1):
        date = it.match_date.isoformat() if it.match_date else "unknown date"
        blocks.append(
            f"Review {i} (match {date}, status={it.review_status.value}):\n"
            f"  AI: verdict={it.ai_verdict} score={it.ai_score}\n"
            f"  {it.ai_review or '(no AI review)'}\n"
            f"  Scout: verdict={it.human_verdict} score={it.human_score}\n"
            f"  {it.human_review or '(no scout review)'}"
        )
    return "\n\n".join(blocks)


class GeminiKpiScorer(KpiScorer):
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def score(
        self,
        kpis: list[KpiDefinition],
        evidence: list[ClipReviewEvidence],
        language: Language,
    ) -> list[KpiScore]:
        config = types.GenerateContentConfig(
            system_instruction=f"{_SYSTEM}\n{_language_line(language)}",
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=_KpiScoresSchema,
        )
        contents = f"{_format_kpis(kpis)}\n\n---\n\n{_format_evidence(evidence)}"

        logger.info(
            "Gemini KPI scoring start: kpis=%d evidence=%d model=%s",
            len(kpis),
            len(evidence),
            self._model,
        )
        started = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[contents],
            config=config,
        )
        logger.info(
            "Gemini KPI scoring done: kpis=%d evidence=%d took=%.2fs",
            len(kpis),
            len(evidence),
            time.monotonic() - started,
        )

        result: _KpiScoresSchema | None = response.parsed
        if result is None:
            logger.warning("Gemini KPI response not auto-parsed; falling back to JSON parse")
            result = _KpiScoresSchema.model_validate_json(response.text)

        scores = self._align(kpis, result, language)
        logger.debug(
            "KPI scores: %s",
            ", ".join(f"{s.kpi_key}={s.score}" for s in scores),
        )
        return scores

    @staticmethod
    def _align(
        kpis: list[KpiDefinition],
        result: _KpiScoresSchema,
        language: Language,
    ) -> list[KpiScore]:
        """Return exactly one KpiScore per requested KPI.

        The model is constrained to the KPI list, but we still realign by key so a
        dropped or hallucinated entry can't corrupt the output: a KPI the model didn't
        return becomes a NULL "not observed" score, and unknown keys are ignored.
        """
        not_observed = _NOT_OBSERVED.get(language, _NOT_OBSERVED[Language.english])
        by_key = {item.kpi_key: item for item in result.scores}
        scores: list[KpiScore] = []
        for kpi in kpis:
            item = by_key.get(kpi.key)
            if item is None:
                logger.warning("Model omitted KPI %s; recording as not observed", kpi.key)
                scores.append(
                    KpiScore(
                        kpi_key=kpi.key,
                        category_key=kpi.category_key,
                        score=None,
                        rationale=not_observed,
                    )
                )
                continue
            scores.append(
                KpiScore(
                    kpi_key=kpi.key,
                    category_key=kpi.category_key,
                    score=item.score,
                    rationale=item.rationale or not_observed,
                )
            )
        return scores
