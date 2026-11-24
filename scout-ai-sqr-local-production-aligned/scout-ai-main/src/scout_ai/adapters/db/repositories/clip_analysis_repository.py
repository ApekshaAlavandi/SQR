"""Persists the evidence-first analysis into the B-screen tables.

Tables written (one transaction): clip_analyses, clip_presence_spans,
clip_observations, clip_tallies.  NOT written here: clip_segments (needs cut video
snippets + an object key), and the existing clip_reviews (handled by
ClipReviewRepository exactly as before).

Where the shape of a jsonb column is a UI contract, it is built in ONE place
(``_actions_json`` / ``_confidence_json`` / ``_areas_json`` below) so it is easy to change.
"""

from __future__ import annotations

import json
from decimal import Decimal
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.domain import clip_evidence as ce
from scout_ai.domain.clip_evidence import ClipEvidence
from scout_ai.domain.ports import ClipAnalysisRepository

# Observations below this confidence get low_confidence = true (prompt label "low" < 0.60).
LOW_CONFIDENCE_BELOW = Decimal("0.60")


def _dec(x: float | int | Decimal, places: int = 2) -> Decimal:
    return Decimal(str(round(float(x), places)))


# ---- jsonb shapes (the contract with the UI) ------------------------------------------
def _actions_json(ev: ClipEvidence) -> dict:
    """{"pass": {"attempted": 2, "successful": 2}, "recoveries": {"count": 3}, ...}"""
    out: dict[str, dict] = {
        t.action: {"attempted": t.attempted, "successful": t.successful}
        for t in ev.action_tallies
    }
    for c in ev.counts_only:
        out[c.action] = {"count": c.count}
    return out


def _confidence_json(ev: ClipEvidence) -> dict:
    """Per-action agreement (how sure the model is the counts are right): {"pass": 0.9}."""
    return {t.action: t.agreement for t in ev.action_tallies}


def _areas_json(ev: ClipEvidence) -> dict | None:
    """{"passing": 0.7, "positional_discipline": 0.85}; NULL when nothing was assessable."""
    return dict(ev.confidence_by_area) or None


def _shots_json(ev: ClipEvidence) -> list[dict]:
    return [
        {"t": s.t, "zone": s.zone, "foot": s.foot, "outcome": s.outcome, "situation": s.situation}
        for s in ev.shots
    ]


# ---- SQL ------------------------------------------------------------------------------
_LATEST = text(
    """
    SELECT id, version FROM clip_analyses
     WHERE clip_id = :clip_id AND club_id = :club_id
     ORDER BY version DESC LIMIT 1
     FOR UPDATE
    """
)
_SUPERSEDE = text(
    "UPDATE clip_analyses SET status = 'superseded' "
    "WHERE clip_id = :clip_id AND club_id = :club_id AND status = 'published'"
)
_INSERT_ANALYSIS = text(
    """
    INSERT INTO clip_analyses
        (club_id, clip_id, lane, version, parent_version_id, status, rating_value,
         duration_s, minutes_in_frame, identification, confidence_by_area, not_observed,
         prose_en, prose_ar, model_version, taxonomy_version, published_at)
    VALUES
        (:club_id, :clip_id, 'club', :version, :parent, 'published', :rating,
         :duration_s, :minutes_in_frame, CAST(:identification AS jsonb),
         CAST(:areas AS jsonb), CAST(:not_observed AS text[]),
         :prose_en, :prose_ar, :model_version, :taxonomy_version, now())
    RETURNING id
    """
)
_INSERT_SPAN = text(
    "INSERT INTO clip_presence_spans (analysis_id, club_id, t0, t1, state) "
    "VALUES (:analysis_id, :club_id, :t0, :t1, :state)"
)
_INSERT_OBS = text(
    """
    INSERT INTO clip_observations
        (club_id, analysis_id, seq, t0, t1, phase, zone, action, quality, significance,
         attributes, outcome, confidence, low_confidence, text_en, text_ar, cited_in_prose)
    VALUES
        (:club_id, :analysis_id, :seq, :t0, :t1, :phase, :zone, :action, :quality,
         :significance, CAST(:attributes AS text[]), :outcome, :confidence,
         :low_confidence, :text_en, :text_ar, :cited)
    """
)
_INSERT_TALLY = text(
    """
    INSERT INTO clip_tallies
        (analysis_id, club_id, touches, by_minute, by_zone, actions, shots, confidence)
    VALUES
        (:analysis_id, :club_id, :touches, CAST(:by_minute AS integer[]),
         CAST(:by_zone AS jsonb), CAST(:actions AS jsonb), CAST(:shots AS jsonb),
         CAST(:confidence AS jsonb))
    """
)


class SqlAlchemyClipAnalysisRepository(ClipAnalysisRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

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
        s, ev = self._session, evidence
        try:
            latest = (await s.execute(_LATEST, {"clip_id": clip_id, "club_id": club_id})).first()
            version = (latest.version + 1) if latest else 1
            parent = latest.id if latest else None
            # The partial unique index allows ONE published row per clip: retire it first.
            await s.execute(_SUPERSEDE, {"clip_id": clip_id, "club_id": club_id})

            analysis_id: UUID = (
                await s.execute(
                    _INSERT_ANALYSIS,
                    {
                        "club_id": club_id,
                        "clip_id": clip_id,
                        "version": version,
                        "parent": parent,
                        "rating": _dec(min(10.0, max(0.0, float(rating))), 1),
                        "duration_s": _dec(ev.duration_s),
                        "minutes_in_frame": _dec(ce.minutes_in_frame(ev)),
                        "identification": json.dumps(ev.identification or {}),
                        "areas": json.dumps(_areas_json(ev)) if _areas_json(ev) else None,
                        "not_observed": ev.not_observed,
                        "prose_en": ev.prose_en,
                        "prose_ar": ev.prose_ar,
                        "model_version": model_version,
                        "taxonomy_version": taxonomy_version,
                    },
                )
            ).scalar_one()

            await s.execute(
                _INSERT_SPAN,
                [
                    {"analysis_id": analysis_id, "club_id": club_id,
                     "t0": _dec(sp.t0), "t1": _dec(sp.t1), "state": sp.state}
                    for sp in ev.presence_spans
                ],
            )

            if ev.observations:
                prose = ev.prose_en or ""
                await s.execute(
                    _INSERT_OBS,
                    [
                        {
                            "club_id": club_id, "analysis_id": analysis_id, "seq": i,
                            "t0": _dec(o.t0), "t1": _dec(o.t1), "phase": o.phase,
                            "zone": o.zone, "action": o.action, "quality": o.quality,
                            "significance": o.significance, "attributes": list(o.attributes),
                            "outcome": o.outcome, "confidence": _dec(o.confidence),
                            "low_confidence": _dec(o.confidence) < LOW_CONFIDENCE_BELOW,
                            "text_en": o.text_en, "text_ar": o.text_ar,
                            "cited": ce.format_mmss(o.t0) in prose,
                        }
                        for i, o in enumerate(ev.observations, start=1)
                    ],
                )

            await s.execute(
                _INSERT_TALLY,
                {
                    "analysis_id": analysis_id, "club_id": club_id,
                    "touches": len(ev.touches),
                    "by_minute": ce.touches_by_minute(ev),
                    "by_zone": json.dumps(ce.touches_by_zone(ev)),
                    "actions": json.dumps(_actions_json(ev)),
                    "shots": json.dumps(_shots_json(ev)),
                    "confidence": json.dumps(_confidence_json(ev)),
                },
            )
            await s.commit()
            return analysis_id
        except Exception:
            await s.rollback()
            raise
