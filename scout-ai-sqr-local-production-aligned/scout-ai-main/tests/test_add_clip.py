"""Tests for ``scout-ai-local add-clip`` (file staging + argument rules)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from scout_ai.transport.local import cli


def test_stage_video_copies_and_never_overwrites(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"data")
    dest = tmp_path / "media" / "processed" / "custom" / "a.mp4"

    cli._stage_video(src, dest, "stub")
    assert dest.read_bytes() == b"data"

    dest.write_bytes(b"EXISTING")  # someone else's video
    with pytest.raises(cli.CliError, match="already exists"):
        cli._stage_video(src, dest, "stub")
    assert dest.read_bytes() == b"EXISTING"  # untouched


def test_stage_video_rejects_missing_and_non_mp4(tmp_path):
    with pytest.raises(cli.CliError, match="not found"):
        cli._stage_video(tmp_path / "nope.mp4", tmp_path / "d.mp4", "stub")
    mov = tmp_path / "clip.mov"
    mov.write_bytes(b"x")
    with pytest.raises(cli.CliError, match="only .mp4"):
        cli._stage_video(mov, tmp_path / "d.mp4", "stub")


def test_stage_video_oversize_blocks_gemini_but_only_warns_for_stub(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "MAX_INLINE_BYTES", 5)
    src = tmp_path / "big.mp4"
    src.write_bytes(b"x" * 10)

    with pytest.raises(cli.CliError, match="ffmpeg"):
        cli._stage_video(src, tmp_path / "g.mp4", "gemini")
    assert not (tmp_path / "g.mp4").exists()  # nothing copied on rejection

    cli._stage_video(src, tmp_path / "s.mp4", "stub")
    assert (tmp_path / "s.mp4").exists()
    assert "warning" in capsys.readouterr().err


def test_add_clip_parser_defaults_to_demo_club_and_player():
    args = cli.build_parser().parse_args(["add-clip", "--video", "v.mp4", "--ident-text", "x"])
    assert args.club_id == str(cli.bootstrap.CLUB_ID)
    assert args.player_id == str(cli.bootstrap.PLAYER_ID)
    assert args.position == "CM" and args.draft is False and args.tracked_video is None


@pytest.mark.parametrize(
    "argv, message",
    [
        (["add-clip", "--ident-text", "x"], "give --video"),
        (["add-clip", "--video", "v.mp4"], "--ident-text"),
        (["add-clip", "--video", "v.mp4", "--ident-text", "x", "--position", "XX"], "unknown"),
    ],
)
def test_add_clip_argument_rules(argv, message, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code != 0
    assert message in capsys.readouterr().err


@pytest.mark.skipif(
    not os.environ.get("SCOUT_TEST_DATABASE_URL"),
    reason="set SCOUT_TEST_DATABASE_URL to a LOCAL Postgres to run the add-clip end-to-end test",
)
def test_add_clip_end_to_end(monkeypatch, tmp_path: Path):
    from scout_ai.config import get_settings

    video, tracked = tmp_path / "v.mp4", tmp_path / "t.mp4"
    video.write_bytes(b"v")
    tracked.write_bytes(b"t")
    monkeypatch.setenv("SCOUT_DATABASE_URL", os.environ["SCOUT_TEST_DATABASE_URL"])
    monkeypatch.setenv("SCOUT_AI_BACKEND", "stub")
    monkeypatch.setenv("SCOUT_LOCAL_MEDIA_DIR", str(tmp_path / "media"))
    get_settings.cache_clear()
    try:
        for extra in (["--ident-text", "number 8"], ["--tracked-video", str(tracked)]):
            with pytest.raises(SystemExit) as exc:
                cli.main(["add-clip", "--video", str(video), "--draft", *extra])
            assert exc.value.code == 0
        staged = sorted(p.name for p in (tmp_path / "media").rglob("*.mp4"))
        assert len(staged) == 3  # 2 processed copies + 1 tracked copy
    finally:
        get_settings.cache_clear()
