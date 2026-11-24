# scout-ai — production-aligned local runner

This repository keeps the **production Pub/Sub worker and the local CLI on the same
application/use-case/AI/database code path**. Only the transport and media edge changes.

- Production: `scout-ai-consumer` → Pub/Sub → `transport/flows.py` → Gemini/GCS/Postgres.
- Local: `scout-ai-local` → the same `transport/flows.py` → Gemini/filesystem/Postgres.
- No Pub/Sub is required for local execution.
- `Dockerfile` remains the production image.
- `Dockerfile.local` is a CLI image for local/container testing.
- `run_tracked_clip.py` is a convenience wrapper for the tracked-video flow.

The evidence-first analysis mode populates `clip_analyses`, `clip_presence_spans`,
`clip_observations`, and `clip_tallies` in addition to the existing `clip_reviews` path.

See `B_SCREEN_REQUIREMENTS.md` for the B-02/B-03/B-17/B-18/B-19 source mapping.

### scout-ai

Service that runs AI analysis over clip videos using Gemini. In production it is a Google
Pub/Sub worker (`scout-ai-consumer`) that reads videos from GCS and writes to Postgres.

## Local mode (`scout-ai-local`)

Run every flow from your terminal with **only Postgres** as a dependency — no Pub/Sub, no GCS,
and (by default) no Gemini.

It swaps just the cloud edges; the flow logic is the production code:

| Production            | Local mode                                                         |
|-----------------------|--------------------------------------------------------------------|
| Pub/Sub message       | CLI command builds the same JSON payload and hands it to the same flow processor (`transport/flows.py`) |
| `gs://bucket/key`     | file at `$SCOUT_LOCAL_MEDIA_DIR/<bucket>/<key>` (default `./local_media`) |
| Gemini on Vertex AI   | `SCOUT_AI_BACKEND=stub` (offline, default) or `gemini` (API key)   |

### Quick start

```bash
docker compose -f docker-compose.local.yml up -d     # local Postgres
uv sync                                              # or: pip install -e .
cp .env.example .env                                 # optional; defaults already match

uv run scout-ai-local demo                           # schema + seed + all flows, end to end
```

Or step by step (`seed` prints ready-to-paste commands with the demo IDs):

```bash
uv run scout-ai-local seed
uv run scout-ai-local draft   --clip-id <clip> --club-id <club> --player-id <player>
uv run scout-ai-local approve --clip-id <clip>       # stand in for the scout UI
uv run scout-ai-local summary --club-id <club> --player-id <player>
uv run scout-ai-local kpis    --club-id <club> --sport-id <club_sports.id> --player-id <player>
```

| Command    | Flow / purpose                                                          |
|------------|-------------------------------------------------------------------------|
| `init-db`  | Create the local schema (idempotent)                                    |
| `seed`     | `init-db` + demo club, player, catalog and clips (idempotent)           |
| `draft`    | `ai.draft.requested` — AI review of one clip (`--force` to redo a completed one) |
| `summary`  | `ai-regenerate` — player AI summary                                     |
| `kpis`     | `ai-kpi-recompute` — player KPI scores                                  |
| `approve`  | Mark a completed draft as scout-accepted (or edited with `--human-*`). The summary and KPI flows only read scout-touched reviews, so this is how a fresh draft becomes evidence |
| `send`     | Feed any raw JSON message to a flow: `send --flow draft --payload '{...}'` |
| `demo`     | Everything above, in order                                              |

Exit codes mirror what the Pub/Sub consumer would do: `0` processed/skipped (ack), `2` dropped
(bad payload / non-retryable error), `1` retryable error (nack).

### Using real Gemini locally

```bash
export SCOUT_AI_BACKEND=gemini SCOUT_GEMINI_API_KEY=...
mkdir -p local_media/processed/demo && cp my_clip.mp4 local_media/processed/demo/clip1.mp4
uv run scout-ai-local draft --clip-id ... --club-id ... --force
```

Videos are sent inline (no GCS), so clips must be under ~19 MB — the pipeline's downsized
videos normally are. Tracked clips are read from `local_media/tracking/<tracking_object_key>`.
The prompts and response parsing are identical to production.

### Notes

- **The local schema is an approximation.** In production the tables are owned by the upstream
  service; `transport/local/schema.py` recreates only the columns scout-ai touches. If upstream
  changes, update it.
- **Safety:** local commands create tables and write rows, so they refuse a non-local
  `SCOUT_DATABASE_URL` unless you pass `--allow-remote-db`.
- Stub output is deterministic and clearly labelled `[STUB — no AI call]`; rows are stamped
  `ai_model = stub-local`.
- Tests: `uv run pytest` (add `SCOUT_TEST_DATABASE_URL=<local postgres>` to include the
  end-to-end demo test).
