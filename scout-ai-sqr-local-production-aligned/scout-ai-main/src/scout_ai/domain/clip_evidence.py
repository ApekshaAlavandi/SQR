"""Structured evidence extracted from one clip, plus the pre-persistence sanitizer.

``ClipEvidence`` is what the evidence-first prompt returns (minus the free-text report,
which keeps flowing into ``clip_reviews`` as before). ``sanitize_evidence`` enforces the
prompt's own audit rules (Section 3.9 / 3.10) and the DB CHECK constraints in code, so a
single bad record from the model is dropped (and logged) instead of failing the whole
transaction and sending the Pub/Sub message into a retry loop.

Pure Python: no pydantic, no SQLAlchemy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scout_ai.domain import clip_taxonomy as tx


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PresenceSpan:
    t0: float
    t1: float
    state: str


@dataclass(frozen=True, slots=True)
class Observation:
    t0: float
    t1: float
    phase: str
    zone: int
    action: str
    quality: str
    outcome: str
    significance: int
    attributes: tuple[str, ...]
    confidence: float
    text_en: str
    text_ar: str


@dataclass(frozen=True, slots=True)
class ActionTally:
    action: str
    attempted: int
    successful: int
    agreement: float


@dataclass(frozen=True, slots=True)
class CountOnly:
    action: str
    count: int


@dataclass(frozen=True, slots=True)
class Shot:
    t: int
    zone: int
    foot: str
    outcome: str
    situation: str


@dataclass(frozen=True, slots=True)
class Touch:
    t: int
    zone: int


@dataclass(slots=True)
class ClipEvidence:
    duration_s: float
    presence_spans: list[PresenceSpan] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    action_tallies: list[ActionTally] = field(default_factory=list)
    counts_only: list[CountOnly] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    touches: list[Touch] = field(default_factory=list)
    identification: dict | None = None
    confidence_by_area: dict[str, float] = field(default_factory=dict)
    not_observed: list[str] = field(default_factory=list)
    prose_en: str | None = None
    prose_ar: str | None = None
    # Human-readable notes about what the sanitizer dropped/repaired (logged, not stored).
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers shared with the persistence layer
# --------------------------------------------------------------------------- #
def format_mmss(seconds: float) -> str:
    """Seconds -> MM:SS (the prompt's Section 3.3 formula)."""
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


def minutes_in_frame(ev: ClipEvidence) -> float:
    """Total time the target was 'in' (minutes, 2 dp, never above the clip length)."""
    secs = sum(sp.t1 - sp.t0 for sp in ev.presence_spans if sp.state == "in")
    secs = min(secs, ev.duration_s)
    return round(secs / 60.0, 2)


def minute_buckets(duration_s: float) -> int:
    return max(1, math.ceil(duration_s / 60.0))


def touches_by_minute(ev: ClipEvidence) -> list[int]:
    n = minute_buckets(ev.duration_s)
    out = [0] * n
    for t in ev.touches:
        out[min(t.t // 60, n - 1)] += 1
    return out


def touches_by_zone(ev: ClipEvidence) -> dict[str, int]:
    """Dense map '1'..'18' -> touch count (sums to len(touches))."""
    out = {str(z): 0 for z in range(tx.ZONE_MIN, tx.ZONE_MAX + 1)}
    for t in ev.touches:
        out[str(t.zone)] += 1
    return out


# --------------------------------------------------------------------------- #
# Sanitizer
# --------------------------------------------------------------------------- #
_PHASES = tx.as_set(tx.Phase)
_QUALITIES = tx.as_set(tx.Quality)
_OUTCOMES = tx.as_set(tx.Outcome)
_STATES = tx.as_set(tx.PresenceState)
_ACTIONS = tx.as_set(tx.ObsAction)
_ATTRS = tx.as_set(tx.Attribute)
_TALLY = tx.as_set(tx.TallyAction)
_COUNT = tx.as_set(tx.CountOnlyAction)
_FEET = tx.as_set(tx.Foot)
_SHOT_OUT = tx.as_set(tx.ShotOutcome)
_SHOT_SIT = tx.as_set(tx.ShotSituation)
_AREAS = tx.as_set(tx.Area)
_NOT_OBS = tx.as_set(tx.NotObserved)


def _clamp01(x: float) -> float:
    return round(min(1.0, max(0.0, float(x))), 2)


def _zone_ok(z: int) -> bool:
    return tx.ZONE_MIN <= z <= tx.ZONE_MAX


def _repair_spans(spans: list[PresenceSpan], duration: float, warn: list[str]) -> list[PresenceSpan]:
    """Chronological, non-overlapping, gap-free, from 0 to exactly ``duration``."""
    cleaned: list[PresenceSpan] = []
    for sp in sorted(spans, key=lambda s: (s.t0, s.t1)):
        if sp.state not in _STATES:
            warn.append(f"span dropped: bad state {sp.state!r}")
            continue
        t0, t1 = max(0.0, sp.t0), min(duration, sp.t1)
        if cleaned:
            t0 = max(t0, cleaned[-1].t1)  # resolve overlap
        if t1 <= t0:
            warn.append(f"span dropped: empty after clamp ({sp.t0}-{sp.t1})")
            continue
        if cleaned and t0 > cleaned[-1].t1:  # fill gap
            cleaned.append(PresenceSpan(cleaned[-1].t1, t0, "out"))
            warn.append(f"gap {cleaned[-2].t1}-{t0} filled with 'out'")
        cleaned.append(PresenceSpan(t0, t1, sp.state))
    if not cleaned:
        return [PresenceSpan(0.0, duration, "out")]
    if cleaned[0].t0 > 0:
        cleaned.insert(0, PresenceSpan(0.0, cleaned[0].t0, "out"))
    if cleaned[-1].t1 < duration:
        last = cleaned[-1]
        if last.state == "out":
            cleaned[-1] = PresenceSpan(last.t0, duration, "out")
        else:
            cleaned.append(PresenceSpan(last.t1, duration, "out"))
    # merge neighbours with the same state (PK is (analysis_id, t0); keeps the lane tidy)
    merged: list[PresenceSpan] = []
    for sp in cleaned:
        if merged and merged[-1].state == sp.state:
            merged[-1] = PresenceSpan(merged[-1].t0, sp.t1, sp.state)
        else:
            merged.append(sp)
    return merged


def sanitize_evidence(raw: ClipEvidence) -> ClipEvidence | None:
    """Return a DB-safe copy of ``raw`` or ``None`` if there is nothing persistable.

    Rules (all from the prompt's Section 3.9/3.10 or the DB constraints):
      * duration must be > 0
      * spans: repaired to cover [0, duration] contiguously
      * observations / touches / shots must sit entirely inside an 'in' span
      * tokens must belong to the taxonomy, zone 1..18, t0 < t1, significance 1..3
      * tallies: attempted > 0, successful <= attempted, duplicates merged, and raised to
        at least the number of matching observations (prompt Section 6.4 reconciliation)
    Dropped/repaired items are listed in ``warnings``.
    """
    warn: list[str] = []
    dur = float(raw.duration_s or 0)
    if dur <= 0:
        return None

    spans = _repair_spans(raw.presence_spans, dur, warn)
    in_spans = [s for s in spans if s.state == "in"]

    def inside(t0: float, t1: float) -> bool:
        return any(s.t0 <= t0 and t1 <= s.t1 for s in in_spans)

    # observations -------------------------------------------------------------
    obs: list[Observation] = []
    for o in sorted(raw.observations, key=lambda x: x.t0):
        why = None
        if o.phase not in _PHASES or o.quality not in _QUALITIES or o.outcome not in _OUTCOMES:
            why = "bad phase/quality/outcome"
        elif o.action not in _ACTIONS:
            why = f"bad action {o.action!r}"
        elif not _zone_ok(o.zone):
            why = f"bad zone {o.zone}"
        elif not (0 <= o.t0 < o.t1 <= dur):
            why = "bad time range"
        elif not 1 <= o.significance <= 3:
            why = "bad significance"
        elif not inside(o.t0, o.t1):
            why = "outside an 'in' span"
        elif not o.text_en.strip() or not o.text_ar.strip():
            why = "empty text"
        if why:
            warn.append(f"observation {o.t0}-{o.t1} {o.action} dropped: {why}")
            continue
        attrs = tuple(dict.fromkeys(a for a in o.attributes if a in _ATTRS))
        obs.append(
            Observation(
                o.t0, o.t1, o.phase, o.zone, o.action, o.quality, o.outcome,
                o.significance, attrs, _clamp01(o.confidence), o.text_en.strip(), o.text_ar.strip(),
            )
        )

    # touches ------------------------------------------------------------------
    touches: list[Touch] = []
    for t in sorted(raw.touches, key=lambda x: x.t):
        if not (0 <= t.t <= dur) or not _zone_ok(t.zone) or not inside(t.t, t.t):
            warn.append(f"touch t={t.t} zone={t.zone} dropped (range/zone/not 'in')")
            continue
        touches.append(Touch(int(t.t), int(t.zone)))

    # shots --------------------------------------------------------------------
    shots: list[Shot] = []
    for s in sorted(raw.shots, key=lambda x: x.t):
        if (
            not (0 <= s.t <= dur) or not _zone_ok(s.zone) or not inside(s.t, s.t)
            or s.foot not in _FEET or s.outcome not in _SHOT_OUT or s.situation not in _SHOT_SIT
        ):
            warn.append(f"shot t={s.t} dropped (range/zone/not 'in'/token)")
            continue
        shots.append(Shot(int(s.t), int(s.zone), s.foot, s.outcome, s.situation))

    # tallies ------------------------------------------------------------------
    merged: dict[str, list[float]] = {}
    for tl in raw.action_tallies:
        if tl.action not in _TALLY or tl.attempted <= 0:
            warn.append(f"tally {tl.action!r} dropped")
            continue
        m = merged.setdefault(tl.action, [0, 0, 1.0])
        m[0] += tl.attempted
        m[1] += min(tl.successful, tl.attempted)
        m[2] = min(m[2], _clamp01(tl.agreement))
    for o in obs:  # an observation of a tally class must have a tally (prompt 6.4)
        if o.action in _TALLY:
            merged.setdefault(o.action, [0, 0, 0.5])
    for action, m in merged.items():
        mine = [o for o in obs if o.action == action]
        if m[0] < len(mine):
            warn.append(f"tally {action}: attempted raised {int(m[0])}->{len(mine)}")
            m[0] = len(mine)
        wins = sum(1 for o in mine if o.outcome == "success")
        if m[1] < wins:
            warn.append(f"tally {action}: successful raised {int(m[1])}->{wins}")
            m[1] = wins
        m[1] = min(m[1], m[0])
    tallies = [ActionTally(a, int(m[0]), int(m[1]), round(m[2], 2)) for a, m in merged.items() if m[0] > 0]

    counts: dict[str, int] = {}
    for c in raw.counts_only:
        if c.action in _COUNT and c.count > 0:
            counts[c.action] = counts.get(c.action, 0) + c.count
    counts_only = [CountOnly(a, n) for a, n in counts.items()]

    # areas / not observed / identification -------------------------------------
    areas = {a: _clamp01(c) for a, c in raw.confidence_by_area.items() if a in _AREAS}
    not_obs = list(dict.fromkeys(n for n in raw.not_observed if n in _NOT_OBS))

    ident = None
    if raw.identification is not None:
        ident = {
            "confidence": _clamp01(raw.identification.get("confidence", 0)),
            "role": str(raw.identification.get("role", "") or ""),
            "cues": [str(c) for c in raw.identification.get("cues", []) or []],
        }

    return ClipEvidence(
        duration_s=dur,
        presence_spans=spans,
        observations=obs,
        action_tallies=tallies,
        counts_only=counts_only,
        shots=shots,
        touches=touches,
        identification=ident,
        confidence_by_area=areas,
        not_observed=not_obs,
        prose_en=(raw.prose_en or "").strip() or None,
        prose_ar=(raw.prose_ar or "").strip() or None,
        warnings=warn,
    )
