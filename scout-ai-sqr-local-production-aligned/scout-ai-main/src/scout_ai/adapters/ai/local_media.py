"""Local stand-in for GCS: map ``gs://<bucket>/<key>`` onto a directory on disk."""

from __future__ import annotations

from pathlib import Path

from scout_ai.domain.exceptions import LocalVideoNotFoundError
from scout_ai.domain.value_objects import GcsUri


def resolve_local_path(media_dir: str | Path, uri: GcsUri) -> Path:
    """``<media_dir>/<bucket>/<object_name>`` — refusing anything that escapes ``media_dir``."""
    root = Path(media_dir).expanduser().resolve()
    path = (root / uri.bucket / uri.object_name).resolve()
    if not path.is_relative_to(root):
        raise LocalVideoNotFoundError(uri.uri, str(path))
    return path
