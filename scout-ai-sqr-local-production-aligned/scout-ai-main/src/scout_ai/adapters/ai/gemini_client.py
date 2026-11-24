from __future__ import annotations

from google import genai
from google.genai import types


def build_gemini_client(project_id: str, location: str) -> genai.Client:
    return genai.Client(vertexai=True, project=project_id, location=location)


def build_local_gemini_client(
    *, api_key: str, project_id: str, location: str
) -> genai.Client:
    """Gemini client for local mode.

    An API key (Gemini Developer API) is the zero-GCP option; otherwise fall back to Vertex
    AI via Application Default Credentials. Videos are sent inline in both cases, so no GCS
    bucket is involved.
    """
    if api_key:
        # Preview models (e.g. gemini-3-flash-preview) are only served under v1beta.
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
    if project_id:
        return build_gemini_client(project_id, location)
    raise ValueError(
        "SCOUT_AI_BACKEND=gemini needs SCOUT_GEMINI_API_KEY (or SCOUT_GCP_PROJECT_ID with "
        "Application Default Credentials). Use SCOUT_AI_BACKEND=stub to run offline."
    )