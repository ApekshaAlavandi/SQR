"""Builds the evidence-first prompt (template: ``evidence_prompt_template.txt``).

The template text is the prompt exactly as authored; this module only fills its
placeholders. Mode-dependent fragments (tracked vs untracked identity rules, gate
question, ...) are reused from ``video_analyzer`` so both prompts stay in sync, and every
enumerated token list comes from ``domain.clip_taxonomy`` (same source as the response
schema and the DB sanitizer).
"""

from __future__ import annotations

from pathlib import Path

from scout_ai.adapters.ai.video_analyzer import (
    _language_directive,
    _mode_fragments,
    effective_video_fps,
)
from scout_ai.domain import clip_taxonomy as tx
from scout_ai.domain.value_objects import ClipAnalysisInput

_TEMPLATE_PATH = Path(__file__).with_name("evidence_prompt_template.txt")
EVIDENCE_PROMPT: str = _TEMPLATE_PATH.read_text(encoding="utf-8")

TRACKED_PLAYER = "the tracked player"


def _presence_definition(is_tracked: bool) -> str:
    if is_tracked:
        return (
            "- in  = the tracking box is on the target player and the player is visible.\n"
            "- out = the tracking box is absent, the player is not visible, or the camera\n"
            "        is elsewhere (replay, close-up, other players).\n"
            "- off = the target player is off the pitch."
        )
    return (
        "- in  = the described target player is visible and identifiable.\n"
        "- out = the target is not visible, cannot be identified, or the camera is elsewhere.\n"
        "- off = the target player is off the pitch."
    )


def _video_sampling_note() -> str:
    """Tell the model how densely the video is sampled (same value that is sent to Gemini)."""
    fps = effective_video_fps()
    if fps is None:
        return ""
    return (
        "VIDEO SAMPLING\n"
        f"The supplied video is sampled at about {fps:g} frames per second. Before recording\n"
        "any touch, shot, or zone change, inspect the frames immediately before and after\n"
        "the suspected moment. Time is always seconds from the start of the clip, never a\n"
        "frame count."
    )


def build_evidence_prompt(data: ClipAnalysisInput) -> str:
    # Verbatim (no .upper()): the prompt requires "the tracked player" to be written as is.
    player_description = (
        TRACKED_PLAYER if data.is_tracked else (data.ident_text or "the primary player featured in the clip")
    )
    return EVIDENCE_PROMPT.format(
        player_description=player_description,
        language_directive=_language_directive(data.language),
        video_sampling_note=_video_sampling_note(),
        presence_definition=_presence_definition(data.is_tracked),
        zone_min=tx.ZONE_MIN,
        zone_max=tx.ZONE_MAX,
        presence_values=tx.csv(tx.PresenceState),
        phase_values=tx.csv(tx.Phase),
        action_values=tx.csv(tx.ObsAction),
        quality_values=tx.csv(tx.Quality),
        outcome_values=tx.csv(tx.Outcome),
        attribute_values=tx.csv(tx.Attribute),
        tally_action_values=tx.csv(tx.TallyAction),
        count_only_values=tx.csv(tx.CountOnlyAction),
        foot_values=tx.csv(tx.Foot),
        shot_outcome_values=tx.csv(tx.ShotOutcome),
        situation_values=tx.csv(tx.ShotSituation),
        area_values=tx.csv(tx.Area),
        not_observed_values=tx.csv(tx.NotObserved),
        **_mode_fragments(data.is_tracked),
    )