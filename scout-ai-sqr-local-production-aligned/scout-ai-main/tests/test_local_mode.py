"""Tests for local mode (``scout-ai-local``): no Pub/Sub, no GCS, optional DB."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from uuid import UUID

import pytest

from scout_ai.adapters.ai import local_video_analyzer as lva
from scout_ai.adapters.ai.local_media import resolve_local_path
from scout_ai.adapters.ai.stub import StubKpiScorer, StubVideoAnalyzer
from scout_ai.adapters.ai.video_analyzer import _VideoAnalysisSchema
from scout_ai.config import Settings
from scout_ai.domain.entities import Language
from scout_ai.domain.exceptions import LocalVideoNotFoundError
from scout_ai.domain.kpi import KpiDefinition
from scout_ai.domain.value_objects import ClipAnalysisInput, GcsUri
from scout_ai.transport.local import cli

CLIP = UUID("00000000-0000-4000-8000-000000000101")
CLUB = UUID("00000000-0000-4000-8000-0000000000c1")


def _input(uri: str = "gs://processed/demo/clip.mp4", tracked: bool = False) -> ClipAnalysisInput:
    return ClipAnalysisInput(
        clip_id=CLIP,
        club_id=CLUB,
        video_uri=GcsUri.parse(uri),
        action_tag_key="pass",
        ident_text="number 8",
        is_tracked=tracked,
    )


# --- gs:// -> disk mapping ------------------------------------------------------------
def test_gcs_uri_maps_to_media_dir(tmp_path):
    path = resolve_local_path(tmp_path, GcsUri.parse("gs://tracking/a/b.mp4"))
    assert path == (tmp_path / "tracking" / "a" / "b.mp4").resolve()


def test_path_traversal_is_rejected(tmp_path):
    with pytest.raises(LocalVideoNotFoundError):
        resolve_local_path(tmp_path, GcsUri(bucket="processed", object_name="../../etc/passwd"))


# --- Gemini analyzer, local video source (fake client; no network) --------------------
class _FakeGemini:
    def __init__(self):
        self.calls = []
        self.aio = SimpleNamespace(models=SimpleNamespace(generate_content=self._generate))

    async def _generate(self, *, model, contents, config):
        self.calls.append(contents)
        parsed = _VideoAnalysisSchema(
            content_check="football",
            detailed_analysis="# DETAILED ANALYSIS\nbody",
            summary="short summary",
            score=7.5,
            verdict="monitor",
        )
        return SimpleNamespace(parsed=parsed, text="", usage_metadata=None)


async def test_local_analyzer_sends_file_bytes_inline(tmp_path):
    video = tmp_path / "processed" / "demo" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00fake-mp4-bytes")
    client = _FakeGemini()

    result = await lva.LocalGeminiVideoAnalyzer(client, "m", tmp_path).analyze(_input())

    part = client.calls[0][0]
    assert part.inline_data.data == b"\x00fake-mp4-bytes"  # bytes, not a gs:// reference
    assert part.file_data is None
    assert (result.score, result.verdict) == (7.5, "monitor")


async def test_local_analyzer_missing_file(tmp_path):
    with pytest.raises(LocalVideoNotFoundError, match="not found"):
        await lva.LocalGeminiVideoAnalyzer(_FakeGemini(), "m", tmp_path).analyze(_input())


async def test_local_analyzer_rejects_oversize(tmp_path, monkeypatch):
    video = tmp_path / "processed" / "demo" / "clip.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x" * 10)
    monkeypatch.setattr(lva, "MAX_INLINE_BYTES", 5)
    with pytest.raises(ValueError, match="inline"):
        await lva.LocalGeminiVideoAnalyzer(_FakeGemini(), "m", tmp_path).analyze(_input())


# --- stubs ----------------------------------------------------------------------------
async def test_stub_analyzer_is_deterministic_and_valid(tmp_path):
    a = await StubVideoAnalyzer(str(tmp_path)).analyze(_input())
    b = await StubVideoAnalyzer(str(tmp_path)).analyze(_input())
    assert a == b
    assert 0 <= a.score <= 10 and a.verdict in {"sign", "monitor", "pass"}


async def test_stub_scorer_returns_one_score_per_kpi_with_some_unobserved():
    kpis = [KpiDefinition(f"k{i}", "cat", f"K{i}", "") for i in range(40)]
    scores = await StubKpiScorer().score(kpis, [], Language.english)
    assert [s.kpi_key for s in scores] == [k.key for k in kpis]
    assert any(s.score is None for s in scores) and any(s.score is not None for s in scores)


# --- guards / parsing -----------------------------------------------------------------
def test_remote_db_is_refused_unless_allowed():
    remote = Settings(database_url="postgresql+asyncpg://u:p@prod.example.com/x")
    with pytest.raises(cli.CliError):
        cli._assert_local_db(remote, allow_remote=False)
    cli._assert_local_db(remote, allow_remote=True)
    cli._assert_local_db(Settings(database_url="postgresql+asyncpg://u:p@localhost/x"), False)


def test_cli_rejects_bad_uuid():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["draft", "--clip-id", "nope", "--club-id", str(CLUB)])


def test_local_mode_needs_no_pubsub_package():
    """Importing + building local mode must work with google.cloud.pubsub blocked."""
    code = textwrap.dedent(
        """
        import sys
        sys.modules["google.cloud.pubsub_v1"] = None   # any import of it raises ImportError
        sys.modules["google.cloud.storage"] = None
        import scout_ai.transport.local.cli
        from scout_ai.config import Settings
        from scout_ai.transport.container import AppContainer
        c = AppContainer.build_local(Settings())
        assert c.settings.gcs_processed_bucket == "processed"
        assert c.settings.gemini_model == "stub-local"
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# --- end to end against a real local Postgres (opt-in) --------------------------------
@pytest.mark.skipif(
    not os.environ.get("SCOUT_TEST_DATABASE_URL"),
    reason="set SCOUT_TEST_DATABASE_URL to a LOCAL Postgres to run the end-to-end demo",
)
def test_demo_end_to_end(monkeypatch):
    from scout_ai.config import get_settings

    monkeypatch.setenv("SCOUT_DATABASE_URL", os.environ["SCOUT_TEST_DATABASE_URL"])
    monkeypatch.setenv("SCOUT_AI_BACKEND", "stub")
    get_settings.cache_clear()
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["demo"])
        assert exc.value.code == 0
    finally:
        get_settings.cache_clear()
