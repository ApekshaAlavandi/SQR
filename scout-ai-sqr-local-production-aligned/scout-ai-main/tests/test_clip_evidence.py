"""Evidence-first analysis: prompt, schema, sanitizer, and (opt-in) real-Postgres persistence."""

from __future__ import annotations

import json
import os
import re
from uuid import UUID

import pytest

from scout_ai.adapters.ai.evidence_analyzer import _EvidenceAnalysisSchema, to_evidence
from scout_ai.adapters.ai.evidence_prompt import EVIDENCE_PROMPT, build_evidence_prompt
from scout_ai.domain import clip_evidence as ce
from scout_ai.domain import clip_taxonomy as tx
from scout_ai.domain.entities import Language
from scout_ai.domain.value_objects import ClipAnalysisInput, GcsUri

CLIP = UUID("00000000-0000-4000-8000-000000000102")
CLUB = UUID("00000000-0000-4000-8000-0000000000c1")


def _input(tracked=True, language=Language.english) -> ClipAnalysisInput:
    return ClipAnalysisInput(
        clip_id=CLIP, club_id=CLUB, video_uri=GcsUri.parse("gs://tracking/x.mp4"),
        action_tag_key="pass", ident_text=None if tracked else "number 8",
        is_tracked=tracked, language=language,
    )


# --- prompt ---------------------------------------------------------------------------
@pytest.mark.parametrize("tracked", [True, False])
@pytest.mark.parametrize("language", [Language.english, Language.arabic])
def test_prompt_renders_with_no_leftover_placeholders(tracked, language):
    text = build_evidence_prompt(_input(tracked, language))
    assert not re.findall(r"\{[a-z_0-9]+\}", text)
    for token in ("oa, ot, od, dt, sp", "in, out, off", "good, work, neutral", "1–18"):
        assert token in text
    assert '"touches": [' in text  # {{ }} in the JSON contract collapsed to { }


def test_tracked_prompt_uses_player_description_verbatim():
    text = build_evidence_prompt(_input(True))
    assert "Target Player: the tracked player\n" in text


def test_template_is_intact():
    assert EVIDENCE_PROMPT.count("6.7 touches") == 1
    assert EVIDENCE_PROMPT.rstrip().endswith("Never wrap it in Markdown code fences.")


# --- response schema ------------------------------------------------------------------
def test_response_schema_has_no_defaults_and_keeps_generation_order():
    schema = _EvidenceAnalysisSchema.model_json_schema()
    assert "default" not in json.dumps(schema)  # Gemini rejects defaults
    order = list(schema["properties"])
    assert order[:3] == ["content_check", "clip_duration_s", "presence_spans"]
    assert order.index("observations") < order.index("detailed_analysis")


# --- the sample response from the new prompt ------------------------------------------
def _sample() -> dict:
    ob = lambda t0, t1, phase, action, q, out, sig, attrs, text: {  # noqa: E731
        "t0": t0, "t1": t1, "phase": phase, "zone": 8, "action": action, "quality": q,
        "outcome": out, "significance": sig, "attributes": attrs, "confidence": 0.9,
        "text_en": text, "text_ar": "نص",
    }
    return {
        "content_check": "football", "clip_duration_s": 191,
        "presence_spans": [
            {"t0": 0, "t1": 6, "state": "in"}, {"t0": 6, "t1": 27, "state": "out"},
            {"t0": 27, "t1": 34, "state": "in"}, {"t0": 34, "t1": 63, "state": "out"},
            {"t0": 63, "t1": 77, "state": "in"}, {"t0": 77, "t1": 135, "state": "out"},
            {"t0": 135, "t1": 146, "state": "in"}, {"t0": 146, "t1": 154, "state": "out"},
            {"t0": 154, "t1": 170, "state": "in"}, {"t0": 170, "t1": 186, "state": "out"},
            {"t0": 186, "t1": 191, "state": "in"},
        ],
        "observations": [
            ob(1, 3, "oa", "pass", "good", "success", 2, ["passing_short", "first_touch"], "Passed."),
            ob(27, 30, "oa", "pass", "good", "success", 2, ["passing_progressive"], "Passed."),
            ob(63, 66, "od", "positioning", "neutral", "neutral", 1, ["positioning_defensive"], "Held."),
        ],
        "action_tallies": [{"action": "pass", "attempted": 2, "successful": 2, "agreement": 0.9}],
        "counts_only": [], "shots": [],
        "touches": [{"t": 1, "zone": 8}, {"t": 28, "zone": 8}],
        "identification": {"confidence": 0.9, "role": "central midfielder", "cues": []},
        "confidence_by_area": [{"area": "passing", "confidence": 0.7}],
        "not_observed": ["shooting", "dribbling"],
        "detailed_analysis": "# DETAILED ANALYSIS\n\nx", "summary": "• Player:\n  the tracked player (high)",
        "prose_en": "At 00:01 the player passed. At 00:27 the player passed.", "prose_ar": "ar",
        "score": 6.5, "verdict": "monitor",
    }


def test_sample_parses_and_sanitizes_cleanly():
    ev = to_evidence(_EvidenceAnalysisSchema.model_validate(_sample()))
    assert ev is not None and ev.warnings == []
    assert len(ev.observations) == 3 and len(ev.touches) == 2
    assert sum(ce.touches_by_minute(ev)) == sum(ce.touches_by_zone(ev).values()) == 2
    assert len(ce.touches_by_minute(ev)) == 4  # 191 s -> 4 one-minute buckets


def test_not_football_is_not_persistable():
    d = _sample() | {"content_check": "not_football", "identification": None}
    assert to_evidence(_EvidenceAnalysisSchema.model_validate(d)) is None


def test_unknown_enum_value_from_model_fails_schema_validation():
    d = _sample()
    d["observations"][0]["phase"] = "attack"
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        _EvidenceAnalysisSchema.model_validate(d)


# --- sanitizer ------------------------------------------------------------------------
def _raw(**kw) -> ce.ClipEvidence:
    base = dict(
        duration_s=100,
        presence_spans=[ce.PresenceSpan(0, 40, "in"), ce.PresenceSpan(40, 100, "out")],
    )
    return ce.ClipEvidence(**(base | kw))


def _obs(t0, t1, zone=8, action="pass", outcome="success"):
    return ce.Observation(t0, t1, "oa", zone, action, "good", outcome, 1, (), 0.9, "x", "س")


def test_drops_records_outside_in_spans_or_with_bad_zone():
    ev = ce.sanitize_evidence(
        _raw(
            observations=[_obs(10, 12), _obs(45, 47), _obs(20, 22, zone=19), _obs(38, 42)],
            touches=[ce.Touch(10, 8), ce.Touch(50, 8), ce.Touch(11, 0)],
            shots=[ce.Shot(60, 14, "left", "goal", "open_play")],
        )
    )
    assert [o.t0 for o in ev.observations] == [10]
    assert [t.t for t in ev.touches] == [10]
    assert ev.shots == []
    assert sum("dropped" in w for w in ev.warnings) == 6


def test_spans_are_repaired_to_cover_the_whole_clip():
    ev = ce.sanitize_evidence(
        _raw(presence_spans=[ce.PresenceSpan(5, 20, "in"), ce.PresenceSpan(15, 30, "out"),
                             ce.PresenceSpan(50, 120, "in")])
    )
    s = ev.presence_spans
    assert s[0].t0 == 0 and s[-1].t1 == 100
    assert all(a.t1 == b.t0 for a, b in zip(s, s[1:], strict=False))
    assert all(sp.t1 > sp.t0 for sp in s)


def test_tallies_are_reconciled_with_observations():
    ev = ce.sanitize_evidence(
        _raw(
            observations=[_obs(1, 2), _obs(3, 4), _obs(5, 6, action="carry")],
            action_tallies=[ce.ActionTally("pass", 1, 0, 0.9), ce.ActionTally("tackle", 0, 0, 0.9)],
        )
    )
    tallies = {t.action: t for t in ev.action_tallies}
    assert (tallies["pass"].attempted, tallies["pass"].successful) == (2, 2)
    assert tallies["carry"].attempted == 1  # observation without a tally gets one
    assert "tackle" not in tallies


def test_zero_duration_is_not_persistable():
    assert ce.sanitize_evidence(ce.ClipEvidence(duration_s=0)) is None


def test_mmss_matches_the_prompt_table():
    assert [ce.format_mmss(s) for s in (4, 60, 108, 121, 219)] == [
        "00:04", "01:00", "01:48", "02:01", "03:39"]


# --- real Postgres (opt-in) -----------------------------------------------------------
pytestmark_db = pytest.mark.skipif(
    not os.environ.get("SCOUT_TEST_DATABASE_URL"),
    reason="set SCOUT_TEST_DATABASE_URL to a LOCAL Postgres",
)


@pytestmark_db
async def test_draft_persists_evidence_and_versions_on_redo(monkeypatch):
    from sqlalchemy import text

    from scout_ai.adapters.ai.stub import STUB_MODEL_NAME  # noqa: F401
    from scout_ai.config import Settings
    from scout_ai.transport import flows
    from scout_ai.transport.container import AppContainer
    from scout_ai.transport.local import bootstrap

    settings = Settings(database_url=os.environ["SCOUT_TEST_DATABASE_URL"], ai_backend="stub")
    c = AppContainer.build_local(settings)
    try:
        await bootstrap.init_db(c.engine)
        await bootstrap.seed(c.engine)
        payload = json.dumps({"clip_id": str(bootstrap.CLIP_TRACKED), "club_id": str(bootstrap.CLUB_ID),
                              "player_id": str(bootstrap.PLAYER_ID)}).encode()
        async with c.session_factory() as s:
            await s.execute(text("UPDATE clip_reviews SET ai_generation_status='pending' "
                                 "WHERE clip_id=:c"), {"c": bootstrap.CLIP_TRACKED})
            await s.commit()
        for _ in range(2):
            outcome = await flows.FLOW_BUILDERS["draft"](c)(payload)
            assert outcome is flows.ProcessOutcome.PROCESSED
            async with c.session_factory() as s:
                await s.execute(text("UPDATE clip_reviews SET ai_generation_status='pending' "
                                     "WHERE clip_id=:c"), {"c": bootstrap.CLIP_TRACKED})
                await s.commit()
        async with c.session_factory() as s:
            rows = (await s.execute(text(
                "SELECT version, status FROM clip_analyses WHERE clip_id=:c ORDER BY version"),
                {"c": bootstrap.CLIP_TRACKED})).all()
            assert [(r.version, r.status) for r in rows] == [(1, "superseded"), (2, "published")]
            n = (await s.execute(text(
                "SELECT count(*) FROM clip_observations o JOIN clip_analyses a ON a.id=o.analysis_id "
                "WHERE a.clip_id=:c AND a.status='published'"), {"c": bootstrap.CLIP_TRACKED})).scalar_one()
            assert n == 3
            t = (await s.execute(text(
                "SELECT touches FROM clip_tallies t JOIN clip_analyses a ON a.id=t.analysis_id "
                "WHERE a.clip_id=:c AND a.status='published'"), {"c": bootstrap.CLIP_TRACKED})).scalar_one()
            assert t == 4
    finally:
        await c.dispose()
