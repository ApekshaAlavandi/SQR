"""``scout-ai-local``: run scout-ai's flows from the command line, no Pub/Sub, no GCS.

Each flow command builds the same JSON payload a publisher would put on the topic and hands
it to the *same* processor the Pub/Sub worker uses (``scout_ai.transport.flows``), so the
idempotency check, error classification and failure marking behave exactly as in production.

Exit codes mirror what the Pub/Sub consumer would do with the message:
    0  PROCESSED or SKIPPED  -> would ack
    2  FAILED                -> would ack and drop (bad payload / non-retryable error)
    1  exception raised      -> would nack and be redelivered (retryable)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import make_url

from scout_ai.adapters.ai.local_media import resolve_local_path
from scout_ai.adapters.ai.local_video_analyzer import MAX_INLINE_BYTES
from scout_ai.config import Settings, get_settings
from scout_ai.domain.exceptions import VideoNotReadyError
from scout_ai.domain.value_objects import GcsUri
from scout_ai.transport import flows
from scout_ai.transport.container import AppContainer
from scout_ai.transport.flows import ProcessOutcome
from scout_ai.transport.local import bootstrap, jobs

logger = logging.getLogger("scout_ai.local")

_LOCAL_HOSTS = {None, "", "localhost", "127.0.0.1", "::1", "db", "postgres", "host.docker.internal"}
# The regenerate event requires a clip_id, but the summary flow never reads it.
_NIL_UUID = "00000000-0000-0000-0000-000000000000"

EXIT_OK, EXIT_RETRYABLE, EXIT_DROPPED = 0, 1, 2


class CliError(Exception):
    """A user-facing error: print the message, exit non-zero, no traceback."""


# --------------------------------------------------------------------------- #
# Safety
# --------------------------------------------------------------------------- #
def _assert_local_db(settings: Settings, allow_remote: bool) -> None:
    """Local mode creates tables and writes demo rows — refuse a non-local database."""
    host = make_url(settings.database_url).host
    if host in _LOCAL_HOSTS or allow_remote:
        return
    raise CliError(
        f"SCOUT_DATABASE_URL points at '{host}', which is not a local host. "
        "Local mode writes to the database, so it refuses by default. "
        "Point SCOUT_DATABASE_URL at your local Postgres, or pass --allow-remote-db "
        "if you really mean it."
    )


# --------------------------------------------------------------------------- #
# Reporting (read back what the flow wrote)
# --------------------------------------------------------------------------- #
def _print_rows(title: str, rows: Sequence[dict[str, Any]]) -> None:
    print(f"\n── {title} " + "─" * max(0, 60 - len(title)))
    if not rows:
        print("(no rows)")
        return
    for row in rows:
        for key, value in row.items():
            rendered = "" if value is None else str(value)
            if "\n" in rendered:
                rendered = rendered.replace("\n", "\n      ")
            print(f"  {key:>18}: {rendered}")
        print()


async def _query(container: AppContainer, sql: str, **params: Any) -> list[dict[str, Any]]:
    async with container.session_factory() as session:
        result = await session.execute(text(sql), params)
        rows = [dict(r) for r in result.mappings().all()]
        await session.commit()  # no-op for SELECT; needed for UPDATE ... RETURNING
        return rows


async def _report_draft(container: AppContainer, clip_id: str) -> None:
    rows = await _query(
        container,
        "SELECT ai_generation_status, review_status, ai_verdict, ai_score, ai_model, "
        "ai_total_tokens, ai_review FROM clip_reviews WHERE clip_id = :c",
        c=UUID(clip_id),
    )
    _print_rows("clip_reviews", rows)


async def _report_analysis(container: AppContainer, clip_id: str) -> None:
    """What the B screens will read for this clip (evidence tables)."""
    rows = await _query(
        container,
        "SELECT a.version, a.status, a.rating_value, a.duration_s, a.minutes_in_frame, "
        "a.coverage, a.taxonomy_version, a.model_version, "
        "(SELECT count(*) FROM clip_presence_spans s WHERE s.analysis_id = a.id) AS spans, "
        "(SELECT count(*) FROM clip_observations o WHERE o.analysis_id = a.id) AS observations, "
        "t.touches, t.by_minute, t.actions, t.shots "
        "FROM clip_analyses a LEFT JOIN clip_tallies t ON t.analysis_id = a.id "
        "WHERE a.clip_id = :c ORDER BY a.version DESC",
        c=UUID(clip_id),
    )
    _print_rows("clip_analyses (+ spans / observations / tallies)", rows)


async def _report_job(container: AppContainer, job_id: UUID | None) -> None:
    """The local processing_jobs row the draft ran under (what the B screens trace back to)."""
    if job_id is None:
        return
    rows = await _query(container, jobs.JOB_REPORT_SQL, job=job_id)
    _print_rows("processing_jobs", rows)


async def _report_summary(container: AppContainer, player_id: str) -> None:
    rows = await _query(
        container,
        "SELECT ai_summary, updated_at FROM player_stats WHERE player_id = :p",
        p=UUID(player_id),
    )
    _print_rows("player_stats", rows)


async def _report_kpis(container: AppContainer, player_id: str) -> None:
    rows = await _query(
        container,
        "SELECT kpi_key, category_key, ai_score, ai_rationale, ai_model, "
        "left(evidence_version, 19) AS evidence_version FROM player_kpis "
        "WHERE player_id = :p ORDER BY category_key, kpi_key",
        p=UUID(player_id),
    )
    _print_rows("player_kpis", rows)


# --------------------------------------------------------------------------- #
# Flow execution
# --------------------------------------------------------------------------- #
async def _run_flow(
    container: AppContainer, flow: str, payload: dict[str, Any]
) -> tuple[int, ProcessOutcome | None]:
    """Run one flow on one payload. Returns (exit_code, outcome-or-None-on-exception)."""
    process = flows.FLOW_BUILDERS[flow](container)
    logger.info("Running flow=%s payload=%s", flow, json.dumps(payload, default=str))
    try:
        outcome = await process(json.dumps(payload, default=str).encode("utf-8"))
    except VideoNotReadyError as exc:
        # In production this is nacked and retried once downsizing finishes.
        print(
            f"\n✗ {exc}\n"
            "  (retryable: the clip has no processed_object_key or tracking_object_key yet)"
        )
        return EXIT_RETRYABLE, None
    except Exception as exc:  # noqa: BLE001 - mirrors "nack": report and exit non-zero
        logger.exception("Flow %s raised (would be NACKed / retried in production)", flow)
        print(f"\n✗ {type(exc).__name__}: {exc}")
        return EXIT_RETRYABLE, None

    print(f"\n✓ outcome: {outcome.name}")
    return (EXIT_DROPPED if outcome is ProcessOutcome.FAILED else EXIT_OK), outcome


async def _force_reset(container: AppContainer, clip_id: str) -> None:
    """Let a completed draft be regenerated (the flow skips completed/processing clips)."""
    async with container.session_factory() as session:
        await session.execute(
            text("UPDATE clip_reviews SET ai_generation_status = 'pending' WHERE clip_id = :c"),
            {"c": UUID(clip_id)},
        )
        await session.commit()


def _draft_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "clip_id": args.clip_id,
        "club_id": args.club_id,
        "language": args.language,
    }
    if args.player_id:
        payload["player_id"] = args.player_id
    return payload


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
async def cmd_init_db(container: AppContainer, args: argparse.Namespace) -> int:
    await bootstrap.init_db(container.engine)
    print("✓ schema ready")
    return EXIT_OK


async def cmd_seed(container: AppContainer, args: argparse.Namespace) -> int:
    await bootstrap.init_db(container.engine)
    await bootstrap.seed(container.engine)
    media = Path(container.settings.local_media_dir)
    print(
        f"""✓ demo data loaded

  club   : {bootstrap.CLUB_ID}
  sport  : {bootstrap.SPORT_ID}   (club_sports.id)
  player : {bootstrap.PLAYER_ID}
  clips  : {bootstrap.CLIP_UNTRACKED}  (untracked, needs a draft)
           {bootstrap.CLIP_TRACKED}  (tracked, needs a draft)
           {bootstrap.CLIP_REVIEWED_A}  (already scout-reviewed)
           {bootstrap.CLIP_REVIEWED_B}  (already scout-reviewed)

Videos are read from {media.resolve()}/<bucket>/<key>, e.g.
  {media}/processed/demo/clip1.mp4
  {media}/tracking/demo/clip2_tracked.mp4
The stub AI backend does not need the files; SCOUT_AI_BACKEND=gemini does.

Try:
  scout-ai-local draft --clip-id {bootstrap.CLIP_UNTRACKED} \\
      --club-id {bootstrap.CLUB_ID} --player-id {bootstrap.PLAYER_ID}
  scout-ai-local summary --club-id {bootstrap.CLUB_ID} \\
      --player-id {bootstrap.PLAYER_ID}
  scout-ai-local kpis --club-id {bootstrap.CLUB_ID} \\
      --sport-id {bootstrap.SPORT_ID} --player-id {bootstrap.PLAYER_ID}"""
    )
    return EXIT_OK


async def cmd_draft(container: AppContainer, args: argparse.Namespace) -> int:
    if args.force:
        await _force_reset(container, args.clip_id)
    payload = _draft_payload(args)
    # Stand-in for the upstream job tracker: processing_jobs row + clip_analyses.job_id.
    job_id = await jobs.start_job(
        container, club_id=args.club_id, clip_id=args.clip_id, payload=payload
    )
    code, outcome = await _run_flow(container, "draft", payload)
    await jobs.finish_job(container, job_id, clip_id=args.clip_id, outcome=outcome)
    await _report_draft(container, args.clip_id)
    await _report_analysis(container, args.clip_id)
    await _report_job(container, job_id)
    return code


async def cmd_summary(container: AppContainer, args: argparse.Namespace) -> int:
    payload = {
        "club_id": args.club_id,
        "clip_id": args.clip_id or _NIL_UUID,
        "player_id": args.player_id,
        "language": args.language,
    }
    code, _ = await _run_flow(container, "regenerate", payload)
    await _report_summary(container, args.player_id)
    return code


async def cmd_kpis(container: AppContainer, args: argparse.Namespace) -> int:
    payload = {
        "club_id": args.club_id,
        "sport_id": args.sport_id,
        "player_id": args.player_id,
        "language": args.language,
    }
    code, _ = await _run_flow(container, "kpi_compute", payload)
    await _report_kpis(container, args.player_id)
    return code


async def cmd_approve(container: AppContainer, args: argparse.Namespace) -> int:
    """Simulate the scout UI: mark a completed AI draft accepted (or edited).

    The summary and KPI flows only read scout-touched reviews, so locally this is how a
    freshly generated draft becomes evidence.
    """
    edited = any(v is not None for v in (args.human_review, args.human_score, args.human_verdict))
    rows = await _query(
        container,
        """
        UPDATE clip_reviews
           SET review_status = CAST(:status AS review_status),
               review_source = CAST(:source AS review_source),
               human_review  = COALESCE(:h_review, human_review),
               human_score   = COALESCE(:h_score, human_score),
               human_verdict = COALESCE(CAST(:h_verdict AS clip_verdict), human_verdict),
               reviewed_at   = now(),
               updated_at    = now()
         WHERE clip_id = :c AND ai_generation_status = 'completed'
        RETURNING clip_id, review_status
        """,
        status="edited_by_scout" if edited else "accepted_ai_draft",
        source="human" if edited else "ai",
        h_review=args.human_review,
        h_score=None if args.human_score is None else Decimal(str(args.human_score)),
        h_verdict=args.human_verdict,
        c=UUID(args.clip_id),
    )
    if not rows:
        raise CliError(
            f"No completed AI review for clip {args.clip_id}. Run `draft` for it first."
        )
    print(f"✓ clip {args.clip_id} -> {rows[0]['review_status']}")
    return EXIT_OK


async def cmd_send(container: AppContainer, args: argparse.Namespace) -> int:
    """Publish a raw payload straight into a flow — the closest thing to a Pub/Sub message."""
    raw = Path(args.payload_file).read_text() if args.payload_file else args.payload
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CliError(f"payload is not valid JSON: {exc}") from exc
    code, _ = await _run_flow(container, args.flow, payload)
    return code


def _run_prefix() -> str:
    """How the user launched us, so printed follow-up commands can be pasted as-is."""
    try:
        rel = Path(sys.executable).resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return "python -m scout_ai.transport.local"
    return f"{'.\\' if os.name == 'nt' else './'}{rel} -m scout_ai.transport.local"


def _stage_video(src: Path, dest: Path, backend: str) -> None:
    """Validate a user video and copy it to where the flows will read it."""
    if not src.is_file():
        raise CliError(f"video not found: {src}")
    if src.suffix.lower() != ".mp4":
        raise CliError(f"{src.name}: only .mp4 is supported (the video is sent as video/mp4).")
    size = src.stat().st_size
    if size > MAX_INLINE_BYTES:
        msg = (
            f"{src.name} is {size / 1_048_576:.1f} MB; Gemini inline limit is "
            f"{MAX_INLINE_BYTES // 1_048_576} MB. Shrink it, e.g.:\n"
            f'  ffmpeg -i "{src}" -vf "scale=-2:720,fps=15" -c:v libx264 -crf 32 '
            f"-preset fast -an -y smaller.mp4"
        )
        if backend == "gemini":
            raise CliError(msg)
        print(f"warning: {msg}", file=sys.stderr)
    if dest.exists():
        # Never overwrite (and, on a failed registration, never delete) someone's video.
        raise CliError(f"a video already exists at {dest}; pick another --clip-id.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


async def cmd_add_clip(container: AppContainer, args: argparse.Namespace) -> int:
    """Register YOUR clip: copy the video(s) in, create the clip/video/player rows."""
    if not args.video and not args.tracked_video:
        raise CliError("give --video (normal clip) and/or --tracked-video (box-annotated clip).")
    if not args.tracked_video and not args.ident_text:
        raise CliError(
            "an untracked clip is identified by text only: add --ident-text "
            '"<who to look for, as it appears in the video>".'
        )
    known_positions = {p for b in bootstrap._CATALOG["buckets"] for p in b["positions"]}
    if args.position not in known_positions:
        valid = ", ".join(sorted(known_positions))
        raise CliError(f"unknown --position '{args.position}'. Use one of: {valid}")

    settings = container.settings
    clip_id = UUID(args.clip_id) if args.clip_id else uuid4()
    club_id, player_id = UUID(args.club_id), UUID(args.player_id)
    processed_key = f"custom/{clip_id}.mp4" if args.video else None
    tracking_key = f"custom/{clip_id}_tracked.mp4" if args.tracked_video else None

    staged: list[Path] = []
    try:
        if args.video:
            dest = resolve_local_path(
                settings.local_media_dir, GcsUri(settings.gcs_processed_bucket, processed_key)
            )
            _stage_video(Path(args.video), dest, settings.ai_backend)
            staged.append(dest)
        if args.tracked_video:
            dest = resolve_local_path(
                settings.local_media_dir, GcsUri(settings.gcs_tracking_bucket, tracking_key)
            )
            _stage_video(Path(args.tracked_video), dest, settings.ai_backend)
            staged.append(dest)

        await bootstrap.init_db(container.engine)
        sport_id = await bootstrap.create_clip(
            container.engine,
            club_id=club_id,
            player_id=player_id,
            player_name=args.player_name,
            position_key=args.position,
            clip_id=clip_id,
            opponent_name=args.opponent,
            match_date=args.match_date,
            action_tag_key=args.action,
            minute=args.minute,
            ident_text=args.ident_text,
            processed_object_key=processed_key,
            tracking_object_key=tracking_key,
            language=args.language,
            position_played=args.position_played,
            half=args.half,
            footage_class=args.footage_class,
            competition=args.competition,
            venue=args.venue,
        )
    except Exception as exc:
        for path in staged:  # don't leave orphan files behind a failed registration
            path.unlink(missing_ok=True)
        if isinstance(exc, ValueError):
            raise CliError(str(exc)) from exc
        raise

    mode = "TRACKED (box-annotated video; --ident-text is ignored)" if tracking_key else "UNTRACKED"
    media = Path(settings.local_media_dir).resolve()
    video_path = media / settings.gcs_processed_bucket / processed_key if processed_key else "-"
    tracked_path = media / settings.gcs_tracking_bucket / tracking_key if tracking_key else "-"
    run = _run_prefix()
    print(
        f"""✓ clip registered: {mode}

  clip   : {clip_id}
  club   : {club_id}
  player : {player_id}  ({args.position})
  sport  : {sport_id}   (club_sports.id, for the kpis command)
  video  : {video_path}
  tracked: {tracked_path}

Next:
  {run} draft --clip-id {clip_id} --club-id {club_id} --player-id {player_id}
  {run} approve --clip-id {clip_id}
  {run} summary --club-id {club_id} --player-id {player_id}
  {run} kpis --club-id {club_id} --sport-id {sport_id} --player-id {player_id}"""
    )
    if args.draft:
        return await cmd_draft(
            container,
            argparse.Namespace(
                clip_id=str(clip_id),
                club_id=str(club_id),
                player_id=str(player_id),
                language=args.language,
                force=False,
            ),
        )
    return EXIT_OK


async def cmd_demo(container: AppContainer, args: argparse.Namespace) -> int:
    """init-db → seed → draft ×2 → approve ×2 → summary → kpis, end to end."""
    await bootstrap.init_db(container.engine)
    await bootstrap.seed(container.engine)
    club, player = str(bootstrap.CLUB_ID), str(bootstrap.PLAYER_ID)
    worst = EXIT_OK

    async def step(title: str, coro) -> None:
        nonlocal worst
        print(f"\n=== {title} " + "=" * max(0, 56 - len(title)))
        worst = max(worst, await coro)

    for clip in (bootstrap.CLIP_UNTRACKED, bootstrap.CLIP_TRACKED):
        ns = argparse.Namespace(
            clip_id=str(clip), club_id=club, player_id=player, language=args.language, force=True
        )
        await step(f"draft {clip}", cmd_draft(container, ns))
        await step(
            f"approve {clip}",
            cmd_approve(
                container,
                argparse.Namespace(
                    clip_id=str(clip), human_review=None, human_score=None, human_verdict=None
                ),
            ),
        )
    await step(
        "summary",
        cmd_summary(
            container,
            argparse.Namespace(
                club_id=club, player_id=player, clip_id=None, language=args.language
            ),
        ),
    )
    await step(
        "kpis",
        cmd_kpis(
            container,
            argparse.Namespace(
                club_id=club,
                sport_id=str(bootstrap.SPORT_ID),
                player_id=player,
                language=args.language,
            ),
        ),
    )
    print("\n" + ("✓ demo finished" if worst == EXIT_OK else "✗ demo finished with errors"))
    return worst


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a valid UUID: {value!r}") from exc


def _add_language(p: argparse.ArgumentParser) -> None:
    p.add_argument("--language", default="en", help="Output language: en (default) or ar.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scout-ai-local",
        description="Run scout-ai flows locally: only Postgres required (no Pub/Sub, no GCS).",
    )
    parser.add_argument(
        "--allow-remote-db",
        action="store_true",
        help="Allow a non-local SCOUT_DATABASE_URL (local mode creates tables and writes rows).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Create the local schema (idempotent).").set_defaults(
        func=cmd_init_db
    )
    sub.add_parser("seed", help="Create the schema and load demo data (idempotent).").set_defaults(
        func=cmd_seed
    )

    p = sub.add_parser("draft", help="Flow: AI review of one clip (ai.draft.requested).")
    p.add_argument("--clip-id", required=True, type=_uuid)
    p.add_argument("--club-id", required=True, type=_uuid)
    p.add_argument("--player-id", type=_uuid)
    p.add_argument(
        "--force", action="store_true", help="Regenerate even if the review is already completed."
    )
    _add_language(p)
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("summary", help="Flow: regenerate a player's AI summary (ai-regenerate).")
    p.add_argument("--club-id", required=True, type=_uuid)
    p.add_argument("--player-id", required=True, type=_uuid)
    p.add_argument(
        "--clip-id", type=_uuid, help="Required by the event schema, ignored by the flow."
    )
    _add_language(p)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("kpis", help="Flow: recompute a player's AI KPI scores (ai-kpi-recompute).")
    p.add_argument("--club-id", required=True, type=_uuid)
    p.add_argument("--sport-id", required=True, type=_uuid, help="club_sports.id")
    p.add_argument("--player-id", required=True, type=_uuid)
    _add_language(p)
    p.set_defaults(func=cmd_kpis)

    p = sub.add_parser(
        "approve", help="Simulate a scout accepting (or editing) a completed AI draft."
    )
    p.add_argument("--clip-id", required=True, type=_uuid)
    p.add_argument("--human-review", help="Giving any --human-* option marks it edited_by_scout.")
    p.add_argument("--human-score", type=float)
    p.add_argument("--human-verdict", choices=["sign", "monitor", "pass"])
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("send", help="Feed a raw JSON message to a flow, like a Pub/Sub publish.")
    p.add_argument("--flow", required=True, choices=sorted(flows.FLOW_BUILDERS))
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--payload", help="JSON message body.")
    group.add_argument("--payload-file", help="Path to a file containing the JSON message body.")
    p.set_defaults(func=cmd_send)

    p = sub.add_parser(
        "add-clip",
        help="Register your own clip: copies the video in and creates the clip/video/player rows.",
    )
    p.add_argument("--video", help="Normal (untracked) .mp4 of the clip.")
    p.add_argument(
        "--tracked-video",
        help=".mp4 with the target player boxed. Given: the clip is TRACKED and --ident-text "
        "is not used.",
    )
    p.add_argument(
        "--ident-text", help="Who to look for, e.g. 'Morocco midfielder, red shirt, number 8'."
    )
    p.add_argument("--action", default="pass", help="Action tag, e.g. pass, tackle, shot.")
    p.add_argument("--opponent", default="Local Opponent")
    p.add_argument("--match-date", type=date.fromisoformat, default=date.today(),
                   help="YYYY-MM-DD (default: today).")
    p.add_argument("--minute", type=int)
    p.add_argument("--club-id", type=_uuid, default=str(bootstrap.CLUB_ID))
    p.add_argument("--player-id", type=_uuid, default=str(bootstrap.PLAYER_ID),
                   help="Existing or new player (default: the demo player).")
    p.add_argument("--player-name", help="Used only if the player is created now.")
    p.add_argument("--position", default="CM",
                   help="Used only if the player is created now (GK CB LB RB CDM CM CAM LW RW ST).")
    p.add_argument("--clip-id", type=_uuid, help="Default: a new random UUID.")
    # clips.* columns added upstream (stored as given; see transport/local/schema_ext.py)
    p.add_argument("--position-played", help="Position in THIS match (default: --position).")
    p.add_argument("--half", type=int, choices=[1, 2], help="Match half the clip is from.")
    p.add_argument("--footage-class", choices=["short", "half", "full"], default="short",
                   help="Default: short. NOTE: scout-ai has no chunked analysis for half/full yet.")
    p.add_argument("--competition", help="e.g. 'Saudi Pro League'.")
    p.add_argument("--venue", help="e.g. 'King Fahd Stadium'.")
    p.add_argument("--draft", action="store_true", help="Also run the draft flow right away.")
    _add_language(p)
    p.set_defaults(func=cmd_add_clip)

    p = sub.add_parser("demo", help="End to end: seed, draft, approve, summary, KPIs.")
    _add_language(p)
    p.set_defaults(func=cmd_demo)
    return parser


async def _amain(args: argparse.Namespace, settings: Settings) -> int:
    container = AppContainer.build_local(settings)
    try:
        return await args.func(container, args)
    finally:
        await container.dispose()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        _assert_local_db(settings, args.allow_remote_db)
        db = make_url(settings.database_url).render_as_string(hide_password=True)
        print(
            f"[local mode] db={db}  ai={settings.ai_backend}  "
            f"media={Path(settings.local_media_dir).resolve()}",
            file=sys.stderr,
        )
        code = asyncio.run(_amain(args, settings))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = EXIT_RETRYABLE
    except OSError as exc:  # e.g. Postgres not running
        print(
            f"error: could not reach the database ({exc}). "
            "Is Postgres up? Try: docker compose -f docker-compose.local.yml up -d",
            file=sys.stderr,
        )
        code = EXIT_RETRYABLE
    sys.exit(code)


if __name__ == "__main__":
    main()
