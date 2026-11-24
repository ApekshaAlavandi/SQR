"""Closed vocabularies for the evidence-first clip analysis (single source of truth).

Used in three places, so they can never drift apart:
  1. the prompt placeholders ({phase_values}, {action_values}, ...),
  2. the Gemini response schema (enum-constrained output),
  3. the sanitizer that runs before anything is written to Postgres.

Bump ``TAXONOMY_VERSION`` whenever a token is added/renamed/removed; it is stored on
every ``clip_analyses`` row so the UI/analytics know which vocabulary a row used.

!! If the front-end already fixed its own tokens for the B screens, edit them HERE and
!! nowhere else.
"""

from __future__ import annotations

from enum import Enum

TAXONOMY_VERSION = "2026.11-v1"

ZONE_MIN = 1
ZONE_MAX = 18  # 6 bands x 3 lanes; matches chk_obs_zone in the DB


class Phase(str, Enum):  # chk_obs_phase
    oa = "oa"  # organized attack
    ot = "ot"  # offensive transition
    od = "od"  # organized defense
    dt = "dt"  # defensive transition
    sp = "sp"  # set pieces


class Quality(str, Enum):  # chk_obs_quality
    good = "good"
    work = "work"
    neutral = "neutral"


class Outcome(str, Enum):  # chk_obs_outcome
    success = "success"
    fail = "fail"
    neutral = "neutral"


class PresenceState(str, Enum):  # chk_presence_state
    in_ = "in"
    out = "out"
    off = "off"


class ObsAction(str, Enum):
    pass_ = "pass"
    cross = "cross"
    carry = "carry"
    dribble = "dribble"
    shot = "shot"
    tackle = "tackle"
    interception = "interception"
    clearance = "clearance"
    aerial_duel = "aerial_duel"
    duel = "duel"
    press = "press"
    recovery_run = "recovery_run"
    off_ball_run = "off_ball_run"
    positioning = "positioning"


class TallyAction(str, Enum):
    """Classes counted as attempted/successful (stored in clip_tallies.actions)."""

    pass_ = "pass"
    cross = "cross"
    carry = "carry"
    dribble = "dribble"
    tackle = "tackle"
    press = "press"
    aerial_duel = "aerial_duel"
    duel = "duel"


class CountOnlyAction(str, Enum):
    interceptions = "interceptions"
    clearances = "clearances"
    recoveries = "recoveries"
    possession_losses = "possession_losses"


class Attribute(str, Enum):
    first_touch = "first_touch"
    passing_short = "passing_short"
    passing_long = "passing_long"
    passing_progressive = "passing_progressive"
    dribbling = "dribbling"
    carrying = "carrying"
    shooting = "shooting"
    crossing = "crossing"
    tackling = "tackling"
    interception = "interception"
    clearance = "clearance"
    aerial = "aerial"
    pressing = "pressing"
    off_ball_movement = "off_ball_movement"
    positioning_offensive = "positioning_offensive"
    positioning_defensive = "positioning_defensive"
    recovery_run = "recovery_run"
    body_shape = "body_shape"
    ball_retention = "ball_retention"


class Foot(str, Enum):
    left = "left"
    right = "right"
    head = "head"


class ShotOutcome(str, Enum):
    goal = "goal"
    saved = "saved"
    blocked = "blocked"
    off_target = "off_target"
    post = "post"
    unclear = "unclear"


class ShotSituation(str, Enum):
    open_play = "open_play"
    counter = "counter"
    set_piece = "set_piece"
    penalty = "penalty"


class Area(str, Enum):
    """confidence_by_area keys."""

    first_touch = "first_touch"
    passing = "passing"
    crossing = "crossing"
    dribbling = "dribbling"
    carrying = "carrying"
    shooting = "shooting"
    tackling = "tackling"
    interceptions = "interceptions"
    pressing = "pressing"
    aerial_ability = "aerial_ability"
    tactical_intelligence = "tactical_intelligence"
    positional_discipline = "positional_discipline"


class NotObserved(str, Enum):
    passing = "passing"
    crosses = "crosses"
    dribbling = "dribbling"
    carrying = "carrying"
    shooting = "shooting"
    tackling = "tackling"
    interceptions = "interceptions"
    clearances = "clearances"
    aerial_duels = "aerial_duels"
    pressing = "pressing"
    set_pieces = "set_pieces"


def values(enum_cls: type[Enum]) -> list[str]:
    return [m.value for m in enum_cls]


def as_set(enum_cls: type[Enum]) -> frozenset[str]:
    return frozenset(values(enum_cls))


def csv(enum_cls: type[Enum]) -> str:
    """Comma-separated tokens, as injected into the prompt placeholders."""
    return ", ".join(values(enum_cls))
