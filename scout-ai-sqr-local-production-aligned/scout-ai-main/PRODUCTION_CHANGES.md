# Evidence-first analysis: what changed (and what to port to the Pub/Sub repo)

Existing screens are untouched: `clip_reviews` (review text, score, verdict) is still written by
the same code path. The new code only ADDS rows to `clip_analyses`, `clip_presence_spans`,
`clip_observations`, `clip_tallies` (B-screen data). `clip_segments` is NOT written (needs cut
video snippets + an object key).

Switch: `SCOUT_ANALYSIS_MODE=evidence|legacy` (default `evidence`). `legacy` = old behaviour.

## New files (copy as-is to prod)
| File | Purpose |
|---|---|
| `src/scout_ai/domain/clip_taxonomy.py` | All enum tokens + `TAXONOMY_VERSION`. Single source for prompt, Gemini schema, sanitizer |
| `src/scout_ai/domain/clip_evidence.py` | Evidence dataclasses + `sanitize_evidence` (drops records the DB would reject) |
| `src/scout_ai/adapters/ai/evidence_prompt_template.txt` | Your new prompt (placeholders filled in code) |
| `src/scout_ai/adapters/ai/evidence_prompt.py` | Fills the placeholders |
| `src/scout_ai/adapters/ai/evidence_analyzer.py` | Gemini call with enum-constrained schema; returns review + evidence |
| `src/scout_ai/adapters/db/repositories/clip_analysis_repository.py` | One-transaction insert into the 4 tables; versions/supersedes old analysis |
| `tests/test_clip_evidence.py` | Unit tests + opt-in Postgres test |

## Modified files
| File | Change | In prod patch? |
|---|---|---|
| `domain/entities.py` | `VideoAnalysis.evidence` (optional) | yes |
| `domain/ports.py` | `ClipAnalysisRepository` port | yes |
| `application/use_cases/generate_clip_review.py` | after saving the review, saves evidence (non-fatal on failure) | yes |
| `config.py` | `analysis_mode` | yes |
| `transport/container.py` | picks evidence analyzer, wires repository | yes (`build()` only) |
| `adapters/ai/video_analyzer.py` | `_video_part()` seam (already in local; prod lacks it) | yes |
| `adapters/ai/local_video_analyzer.py`, `adapters/ai/stub.py`, `transport/local/{schema,bootstrap,cli}.py` | local-only (DDL for new tables, stub evidence, CLI report) | no |
| `transport/local/schema_ext.py` (new), `transport/local/jobs.py` (new) | local-only: DDL parity with `schema_new.sql` (extended `clips`, `processing_jobs`, `video_assets`, `clip_reviews.review_source`, `clip_segments`) and a stand-in for the upstream job tracker | no |

`prod_port.patch` (next to this file) is these prod changes as a `patch -p1` against your Pub/Sub repo.
No DB migration is needed in prod (tables exist in schema_new.sql); the service DB role needs
INSERT/UPDATE/SELECT on the 4 tables.
