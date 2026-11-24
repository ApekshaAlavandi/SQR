"""Local stand-in for the upstream job tracker (``processing_jobs``).

In production the upstream service creates the ``processing_jobs`` row, and the B screens
read ``clip_analyses.job_id`` / ``processing_jobs.model_version`` / ``taxonomy_version``.
scout-ai's own code does not write ``processing_jobs`` (nor ``clip_analyses.job_id``), so
locally the CLI plays that role around the draft flow:

    start_job  -> row with status 'running' and model/taxonomy versions pinned
    (flow runs)
    finish_job -> status succeeded/failed/cancelled and links the new analysis via job_id

Nothing here touches production code paths. Failures are logged and never block the flow.
``chunk_plan`` / ``chunks_done`` are left at their defaults: scout-ai has no chunked
(half / full footage) analysis yet.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from scout_ai.domain.clip_taxonomy import TAXONOMY_VERSION

from sqlalchemy import text

if TYPE_CHECKING:
    from scout_ai.transport.container import AppContainer
    from scout_ai.transport.flows import ProcessOutcome

logger = logging.getLogger("scout_ai.local.jobs")

JOB_TYPE = "ai_analysis"


async def start_job(
    container: AppContainer, *, club_id: str, clip_id: str, payload: dict[str, Any]
) -> UUID | None:
    """Create a 'running' ai_analysis job for the clip. Returns None if it cannot (non-fatal)."""
    try:
        async with container.session_factory() as session:
            job_id = (
                await session.execute(
                    text(
                        "INSERT INTO processing_jobs (club_id, clip_id, job_type, status, "
                        "idempotency_key, payload, attempt_count, started_at, "
                        "model_version, taxonomy_version) "
                        "VALUES (:club, :clip, CAST(:jt AS processing_job_type), 'running', "
                        ":key, CAST(:payload AS jsonb), 1, now(), :model, :taxonomy) RETURNING id"
                    ),
                    {
                        "club": UUID(club_id),
                        "clip": UUID(clip_id),
                        "jt": JOB_TYPE,
                        "key": f"local:{JOB_TYPE}:{clip_id}:{uuid4()}",
                        "payload": json.dumps(payload, default=str),
                        "model": container.settings.gemini_model,
                        "taxonomy": TAXONOMY_VERSION,
                    },
                )
            ).scalar_one()
            await session.commit()
            return job_id
    except Exception as exc:  # noqa: BLE001 - tracking must never break the flow itself
        logger.warning("could not create a local processing_jobs row: %s", exc)
        return None


async def finish_job(
    container: AppContainer,
    job_id: UUID | None,
    *,
    clip_id: str,
    outcome: ProcessOutcome | None,
) -> None:
    """Close the job and link the analysis the flow just wrote (if it wrote one)."""
    if job_id is None:
        return
    name = outcome.name if outcome is not None else "EXCEPTION"
    status = {"PROCESSED": "succeeded", "SKIPPED": "cancelled"}.get(name, "failed")
    error = None if status != "failed" else f"flow outcome: {name}"
    try:
        async with container.session_factory() as session:
            # Only an analysis created after this job started belongs to it.
            await session.execute(
                text(
                    "UPDATE clip_analyses SET job_id = :job "
                    "WHERE clip_id = :clip AND status = 'published' AND job_id IS NULL "
                    "AND created_at >= (SELECT started_at FROM processing_jobs WHERE id = :job)"
                ),
                {"job": job_id, "clip": UUID(clip_id)},
            )
            await session.execute(
                text(
                    "UPDATE processing_jobs SET status = CAST(:st AS processing_job_status), "
                    "finished_at = now(), last_error = :err, "
                    "model_version = (SELECT a.model_version FROM clip_analyses a "
                    "                  WHERE a.job_id = processing_jobs.id LIMIT 1), "
                    "taxonomy_version = (SELECT a.taxonomy_version FROM clip_analyses a "
                    "                     WHERE a.job_id = processing_jobs.id LIMIT 1) "
                    "WHERE id = :job"
                ),
                {"st": status, "err": error, "job": job_id},
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not finish local processing_jobs row %s: %s", job_id, exc)


JOB_REPORT_SQL = (
    "SELECT j.job_type, j.status, j.attempt_count, j.model_version, j.taxonomy_version, "
    "j.chunks_done, j.chunk_plan, j.last_error, "
    "(SELECT count(*) FROM clip_analyses a WHERE a.job_id = j.id) AS linked_analyses "
    "FROM processing_jobs j WHERE j.id = :job"
)
