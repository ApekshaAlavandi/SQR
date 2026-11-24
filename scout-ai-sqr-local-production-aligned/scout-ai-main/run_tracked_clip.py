"""Run one tracked football clip locally, without Pub/Sub.

The tracked video is assumed to already contain the target-player bounding box.
The wrapper uses the production-aligned local CLI, so the same evidence prompt,
sanitizer, persistence and processing-job semantics are exercised.

Examples (PowerShell):
    python run_tracked_clip.py --video .\tracked.mp4 --draft
    python run_tracked_clip.py --video .\tracked.mp4 --club-id <uuid> --player-id <uuid> --draft

Use ``--help`` to see all supported clip metadata options.
"""

from __future__ import annotations

import sys

from scout_ai.transport.local.cli import main


def _argv() -> list[str]:
    args = sys.argv[1:]
    if "--video" not in args:
        raise SystemExit("error: run_tracked_clip.py requires --video <tracked.mp4>")
    i = args.index("--video")
    if i + 1 >= len(args):
        raise SystemExit("error: --video requires a file path")
    video = args[i + 1]
    # The wrapper deliberately maps --video to the tracked-video input of the canonical CLI.
    args = args[:i] + ["--tracked-video", video] + args[i + 2:]
    return ["add-clip", *args]


if __name__ == "__main__":
    main(_argv())
