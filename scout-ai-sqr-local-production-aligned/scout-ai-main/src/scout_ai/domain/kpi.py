"""KPI catalog domain: bucket/category/KPI value objects, position→KPI selection,
the scorer's per-KPI output, and the deterministic evidence fingerprint.

All pure — no I/O. The catalog is parsed from ``sport_catalog.default_config`` by the
adapter and handed here as typed value objects; selecting which KPIs apply to a player
and fingerprinting the evidence set are domain rules, so they live here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scout_ai.domain.entities import ClipReviewEvidence


@dataclass(frozen=True, slots=True)
class CatalogBucket:
    """A position bucket (GK/DEF/MID/FWD) and the raw positions it groups."""

    key: str
    positions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KpiCategory:
    """A KPI grouping (e.g. ``passing``) and the buckets it applies to."""

    key: str
    buckets: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class KpiDefinition:
    """One catalog KPI to score, resolved for a player.

    ``category_key`` mirrors the KPI's category and is denormalized onto
    ``player_kpis.category_key``. ``label`` / ``measures`` are English-language hints
    fed to the model so it knows what each KPI captures.
    """

    key: str
    category_key: str
    label: str
    measures: str


@dataclass(frozen=True, slots=True)
class SportCatalog:
    """The parsed KPI taxonomy for one sport (buckets, categories, KPIs)."""

    buckets: tuple[CatalogBucket, ...]
    categories: tuple[KpiCategory, ...]
    kpis: tuple[KpiDefinition, ...]

    def bucket_for_position(self, position_key: str) -> str | None:
        """The bucket key whose ``positions`` contains ``position_key`` (or None)."""
        for bucket in self.buckets:
            if position_key in bucket.positions:
                return bucket.key
        return None

    def select_for_position(
        self, position_key: str
    ) -> tuple[str | None, list[KpiDefinition]]:
        """Resolve the player's bucket and the KPIs that apply to it.

        A KPI applies when its category lists the player's bucket. Returns
        ``(bucket_key, kpis)``; ``(None, [])`` when the position maps to no bucket.
        """
        bucket_key = self.bucket_for_position(position_key)
        if bucket_key is None:
            return None, []

        buckets_by_category = {c.key: c.buckets for c in self.categories}
        selected = [
            kpi
            for kpi in self.kpis
            if bucket_key in buckets_by_category.get(kpi.category_key, ())
        ]
        return bucket_key, selected


@dataclass(frozen=True, slots=True)
class KpiScore:
    """The scorer's verdict for one KPI.

    ``score`` is ``None`` when the evidence does not let the KPI be assessed
    ("not observed") — persisted as a NULL ``ai_score`` that drops out of fit.
    """

    kpi_key: str
    category_key: str
    score: float | None
    rationale: str


def evidence_version(evidence: list[ClipReviewEvidence]) -> str:
    """Deterministic fingerprint of the evidence (clip) set.

    A hash over the sorted ``(clip_id, updated_at)`` pairs — same clips in the same
    review state always yield the same fingerprint; a new or edited clip changes it.
    Never uses a timestamp or run id, so re-running over unchanged evidence is a no-op
    and divergence-on-read (scout's ``reviewed_evidence_version`` vs this) stays honest.
    """
    parts = sorted(
        f"{item.clip_id}:{item.updated_at.isoformat()}" for item in evidence
    )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
