from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from scout_ai.domain.entities import Language, PlayerReviewItem, PlayerSummary
from scout_ai.domain.ports import PlayerSummarizer

logger = logging.getLogger(__name__)


def _language_line(language: Language) -> str:
    return (
        f"OUTPUT LANGUAGE (mandatory): write the entire summary in {language.label}. "
        f"This overrides the language of the input reviews (which may be written in another "
        f"language). Plain, continuous prose only — no markdown."
    )

_SYSTEM = """\
You are an elite football scouting analyst. You are given a player's most recent
clip reviews — each with an AI draft review and the scout's final review, their
scores and verdicts, the review status, and the match date (newest first).

The clip reviews may contain markdown formatting (headings, tables, bullet lists).
Treat all of that purely as source material — read through it for substance and
IGNORE its formatting. Do not copy headings, tables, or list structure into your
answer.

Write a single, cohesive scouting summary for the PLAYER (not per clip):
- Weight scout (human) reviews over AI drafts; treat 'approved' as the strongest
  signal and 'edited_by_scout' as the scout's corrected view.
- Identify consistent strengths, recurring weaknesses, and any trend over time.
- Be concise, evidence-based, and professional. Do not invent facts.

Output format: plain, continuous prose only. NO markdown — no headings, no bullet
points, no numbered lists, no tables. The summary is rendered as plain text, so any
formatting markers would appear literally to the reader.
"""


class _PlayerSummarySchema(BaseModel):
    summary: str = Field(description="The holistic scouting summary for the player.")


def _format(items: list[PlayerReviewItem]) -> str:
    blocks: list[str] = []
    for i, it in enumerate(items, start=1):
        date = it.match_date.isoformat() if it.match_date else "unknown date"
        blocks.append(
            f"Review {i} (match {date}, status={it.review_status.value}):\n"
            f"  AI: verdict={it.ai_verdict} score={it.ai_score}\n"
            f"  {it.ai_review or '(no AI review)'}\n"
            f"  Scout: verdict={it.human_verdict} score={it.human_score}\n"
            f"  {it.human_review or '(no scout review)'}"
        )
    return "\n\n".join(blocks)


class GeminiPlayerSummarizer(PlayerSummarizer):
    def __init__(self, client: genai.Client, model: str) -> None:
        self._client = client
        self._model = model

    async def summarize(
        self, items: list[PlayerReviewItem], language: Language
    ) -> PlayerSummary:
        config = types.GenerateContentConfig(
            system_instruction=f"{_SYSTEM}\n{_language_line(language)}",
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=_PlayerSummarySchema,
        )
        started = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[_format(items)],
            config=config,
        )
        logger.info(
            "Gemini player summary done: reviews=%d took=%.2fs",
            len(items),
            time.monotonic() - started,
        )
        result: _PlayerSummarySchema | None = response.parsed
        if result is None:
            result = _PlayerSummarySchema.model_validate_json(response.text)
        return PlayerSummary(summary=result.summary)
