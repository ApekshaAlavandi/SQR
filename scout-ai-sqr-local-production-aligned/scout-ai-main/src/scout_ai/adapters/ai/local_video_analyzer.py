"""Gemini video analyzer that reads the clip from local disk instead of GCS."""

from __future__ import annotations

import asyncio
from pathlib import Path

from google import genai
from google.genai import types

from scout_ai.adapters.ai.local_media import resolve_local_path
from scout_ai.adapters.ai.video_analyzer import GeminiVideoAnalyzer
from scout_ai.domain.exceptions import LocalVideoNotFoundError
from scout_ai.domain.value_objects import ClipAnalysisInput

# Gemini rejects requests above ~20 MB total; keep headroom for the prompt.
MAX_INLINE_BYTES = 19 * 1024 * 1024


async def build_inline_video_part(media_dir: str | Path, data: ClipAnalysisInput) -> types.Part:
    """Read the clip from local disk and wrap its bytes as a Gemini ``Part`` (shared by the
    legacy and the evidence local analyzers)."""
    path = resolve_local_path(media_dir, data.video_uri)
    if not path.is_file():
        raise LocalVideoNotFoundError(data.video_uri.uri, str(path))
    size = path.stat().st_size
    if size > MAX_INLINE_BYTES:
        raise ValueError(
            f"{path} is {size / 1_048_576:.1f} MB; local mode sends video inline "
            f"(limit {MAX_INLINE_BYTES // 1_048_576} MB). Use a downsized clip."
        )
    content = await asyncio.to_thread(path.read_bytes)
    return types.Part.from_bytes(data=content, mime_type="video/mp4")


class LocalGeminiVideoAnalyzer(GeminiVideoAnalyzer):
    """Same prompt, schema and parsing as production — only the video source differs.

    The video is sent as inline bytes rather than a ``gs://`` reference, so neither GCS nor
    GCS credentials are needed. Clips are already downsized upstream; a clip over the inline
    limit fails with a clear message instead of a confusing API error.
    """

    def __init__(self, client: genai.Client, model: str, media_dir: str | Path) -> None:
        super().__init__(client, model)
        self._media_dir = media_dir

    async def _video_part(self, data: ClipAnalysisInput) -> types.Part:
        return await build_inline_video_part(self._media_dir, data)
