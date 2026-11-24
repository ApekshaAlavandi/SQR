"""Static contract checks for the SQR B-02/B-03/B-17/B-18/B-19 requirements."""

from pathlib import Path

from scout_ai.domain.clip_taxonomy import TAXONOMY_VERSION
from scout_ai.transport.local.schema import SCHEMA_STATEMENTS
from scout_ai.transport.local.schema_ext import EXTENSION_STATEMENTS


ROOT = Path(__file__).parents[1]


def _ddl() -> str:
    return "\n".join((*SCHEMA_STATEMENTS, *EXTENSION_STATEMENTS))


def test_required_b_screen_tables_are_present():
    ddl = _ddl()
    for table in (
        "clips",
        "video_assets",
        "processing_jobs",
        "clip_reviews",
        "clip_analyses",
        "clip_presence_spans",
        "clip_observations",
        "clip_tallies",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in ddl


def test_b_screen_required_columns_are_present():
    ddl = _ddl()
    required = {
        "clips": ["match_date", "opponent_name", "ident_text", "position_played",
                  "half", "footage_class", "competition", "venue", "upload_status"],
        "video_assets": ["processed_object_key", "tracking_object_key"],
        "processing_jobs": ["status", "chunks_done", "model_version", "taxonomy_version"],
        "clip_reviews": ["review_status", "ai_score", "human_score", "ai_review", "human_review"],
        "clip_analyses": ["rating_value", "duration_s", "minutes_in_frame", "coverage",
                          "prose_en", "prose_ar", "model_version", "taxonomy_version"],
        "clip_presence_spans": ["t0", "t1", "state"],
        "clip_observations": ["t0", "t1", "zone", "action", "quality", "significance",
                              "attributes", "outcome", "confidence", "text_en", "text_ar"],
        "clip_tallies": ["touches", "by_minute", "by_zone", "actions", "shots", "confidence"],
    }
    for table, columns in required.items():
        assert table in ddl
        for col in columns:
            assert col in ddl, f"{table}.{col} missing from local DDL"


def test_version_contract_is_explicit():
    jobs = (ROOT / "src/scout_ai/transport/local/jobs.py").read_text()
    assert "model_version" in jobs
    assert "taxonomy_version" in jobs
    assert "TAXONOMY_VERSION" in jobs
    assert TAXONOMY_VERSION


def test_b19_does_not_invent_unconfirmed_sentence_tables():
    doc = (ROOT / "B_SCREEN_REQUIREMENTS.md").read_text()
    assert "prose_en" in doc and "prose_ar" in doc
    assert "proposed" in doc.lower()
