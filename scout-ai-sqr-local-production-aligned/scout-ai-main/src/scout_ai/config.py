from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCOUT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "local"
    log_level: str = "INFO"

    # Postgres (async SQLAlchemy URL).
    database_url: str = "postgresql+asyncpg://scout:scout@localhost:5432/scout"

    # --- Gemini (Vertex AI) ---
    gcp_project_id: str = ""
    vertex_location: str = "global"
    gemini_model: str = "gemini-3-flash-preview"

    # GCS bucket
    gcs_processed_bucket: str = ""
    # Annotated (box-tracked) video lives in its own bucket; used when a clip's
    # video_assets.tracking_object_key is set. See ResolveClipAnalysisInputUseCase.
    gcs_tracking_bucket: str = ""

    # --- Pub/Sub (pull consumers) ---
    pubsub_project_id: str = ""
    pubsub_subscription: str = ""  # "ai-draft-requested-sub"
    pubsub_regenerate_subscription: str = ""  # "ai-regenerate-sub"
    pubsub_kpi_recompute_subscription: str = ""  # "ai-kpi-recompute-sub"
    pubsub_max_messages: int = 4

    # Which clip-analysis prompt the draft flow uses:
    #   "evidence" - evidence-first prompt; review text -> clip_reviews (as before) AND
    #                structured evidence -> clip_analyses & friends (B screens).
    #   "legacy"   - the previous prompt; clip_reviews only. Instant rollback switch.
    analysis_mode: Literal["evidence", "legacy"] = "evidence"

    player_summary_review_limit: int = 5
    kpi_evidence_limit: int = 10

    # --- Local mode (``scout-ai-local`` CLI) ---
    # These are read ONLY by local mode; the Pub/Sub worker ignores them.
    #
    # Which AI implementation local mode wires in:
    #   "stub"   - deterministic canned output, no network, no API key (default: the only
    #              external dependency of a local run is then the Postgres database).
    #   "gemini" - real Gemini. Needs ``gemini_api_key`` (Gemini Developer API) or
    #              ``gcp_project_id`` + Application Default Credentials (Vertex AI).
    ai_backend: Literal["stub", "gemini"] = "stub"
    gemini_api_key: str = ""
    # Stand-in for GCS: ``gs://<bucket>/<key>`` is read from ``<local_media_dir>/<bucket>/<key>``.
    local_media_dir: str = "./local_media"

    # @model_validator(mode="after")
    # def _export_google_credentials(self) -> Settings:
    #     if self.google_application_credentials:
    #         os.environ.setdefault(
    #             "GOOGLE_APPLICATION_CREDENTIALS", self.google_application_credentials
    #         )
    #     return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
