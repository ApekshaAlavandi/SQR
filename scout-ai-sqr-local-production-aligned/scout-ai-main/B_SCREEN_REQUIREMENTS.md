# SQR B-screen backend requirements

This repository is the production-aligned `scout-ai` backend with a local CLI transport.
The B-screen requirements below are derived from the supplied SQR screen/data-mapping
documents. The repository owns the analysis records and local stand-in for upstream job
tracking; it does not implement the frontend.

## B-02 — Player summary / across-clips analysis

Required clip-level sources:
- `clips`: player/position association and clip metadata.
- `clip_analyses`: `rating_value`, `minutes_in_frame`, `coverage`.
- `clip_observations`: timestamped attributes, `quality`, `significance`, `t0`.
- `clip_tallies`: `actions`, `by_zone`, `touches`.
- `clip_reviews`: accepted/scout-edited review state and score.
- `player_stats` / `player_kpis`: existing summary/KPI data.

Derived examples:
- successful action rate per 10 minutes = successful count / `minutes_in_frame` * 10.
- evidence count = distinct clips containing the attribute.
- zone heatmap = sum `clip_tallies.by_zone` over eligible clips.
- touches/minutes summary = sum touches and total `minutes_in_frame`.

Cohort/percentile/profile-history outputs remain outside this service where the source
mapping assigns them to the insight service.

## B-03 — Player timeline + selected clip

Required sources:
- `clips`: date, opponent, position, half, footage class and status.
- `video_assets`: processed/tracked object key for playback.
- `processing_jobs`: analysis status, cancellation and pinned model/taxonomy versions.
- `clip_reviews`: AI/human review state and text.
- `clip_analyses`: rating, duration, minutes in frame, coverage, prose and versions.
- `clip_presence_spans`: presence lane.
- `clip_observations`: moments, zones, actions, quality, significance, confidence and text.
- `clip_tallies`: touches, per-minute/per-zone totals, actions and shots.
- `clip_segments`: evidence snippets are supported by the schema, but this service does
  not cut/write the video snippets yet.

Derived examples:
- coverage = `minutes_in_frame * 60 / duration_s`.
- moment position on timeline = observation timestamp / clip duration.
- presence lane = `clip_presence_spans` ordered by `t0`.

## B-17 — Timeline list view

B-17 uses the same clip data as B-03; it is a different presentation, not a second
analysis dataset.

Required sources:
- `clips.match_date`, `clips.opponent_name`, `clips.competition`,
  `clips.position_played`, `clips.upload_status`.
- `clip_analyses.minutes_in_frame`, `rating_value`.
- `clip_observations.t0`, `quality`, `significance`.
- `clip_reviews.review_status`.
- `processing_jobs.status`.
- `video_assets.processed_object_key` for Play.

Derived examples:
- total clips = count player clips.
- pending review = count `clip_reviews.review_status = pending`.
- analysing = clips/jobs in the analysis state.
- moment strip position = `t0 / duration_s`.
- rating display = AI rating until scout confirmation; mark provisional while pending.

## B-18 — Upload dialog

The local CLI implements the backend-equivalent creation path:
`add-clip` / `run_tracked_clip.py`.

Persisted fields include:
- video → `video_assets.processed_object_key` or `tracking_object_key`.
- match date → `clips.match_date`.
- "How to spot you/him" → `clips.ident_text` for untracked clips.
- position in this clip → `clips.position_played`.
- optional match metadata → `clips.opponent_name`, `competition`, `venue`, `half`,
  `footage_class`.
- creation/job state → `clips.upload_status`, `processing_jobs.*`.

Billing/credit reservation is owned by the billing service and is deliberately not
duplicated in this repository.

## B-19 — Generated text / translations

Current clip-level storage used by this service:
- English generated analysis → `clip_analyses.prose_en`.
- Arabic generated analysis → `clip_analyses.prose_ar`.
- Existing review text remains in `clip_reviews.ai_review` / `human_review`.

The supplied mapping marks sentence-level `text_segment` / `text_translation` as
**proposed**, not part of migration 005. This repository therefore does not invent those
tables or silently diverge from production. Once the production contract is confirmed,
they can be added behind an explicit migration.

## Versioning rule

`processing_jobs.model_version` and `processing_jobs.taxonomy_version` are pinned when
the local job is accepted. The same values are persisted on `clip_analyses`. This keeps
the B-screen evidence reproducible and explains why the two tables legitimately contain
the same version identifiers: the job records what was selected for execution; the
analysis records what was actually published.

## Production boundary

- Production transport: `scout-ai-consumer` → Pub/Sub → `transport/flows.py`.
- Local transport: `scout-ai-local` → the same `transport/flows.py`.
- Local video source: filesystem mapping of `gs://bucket/key`.
- Production video source: GCS.
- Local `processing_jobs` row is a stand-in for the upstream job tracker.
- The evidence tables and analysis use case are shared by both transports.
