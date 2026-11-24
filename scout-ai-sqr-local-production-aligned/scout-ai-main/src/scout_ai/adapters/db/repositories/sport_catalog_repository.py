from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scout_ai.domain.exceptions import SportCatalogNotFoundError
from scout_ai.domain.kpi import (
    CatalogBucket,
    KpiCategory,
    KpiDefinition,
    SportCatalog,
)
from scout_ai.domain.ports import SportCatalogRepository

logger = logging.getLogger(__name__)

# Join the tenant sport (club_sports) to the global taxonomy (sport_catalog) and pull
# the three config fragments the KPI flow needs. sport_id is club_sports.id.
_GET_CATALOG = text(
    """
    SELECT sc.default_config -> 'buckets'        AS buckets,
           sc.default_config -> 'kpi_categories' AS kpi_categories,
           sc.default_config -> 'kpis'           AS kpis
    FROM club_sports cs
    JOIN sport_catalog sc ON sc.sport_key = cs.sport_key
    WHERE cs.id = :sport_id AND cs.club_id = :club_id
    """
)


def _as_list(value: Any) -> list[dict[str, Any]]:
    """Normalize a jsonb column into a list of dicts.

    asyncpg may hand back jsonb as a JSON string or as already-decoded objects
    depending on codec config, so tolerate both (and NULL).
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value)
    return list(value) if isinstance(value, list) else []


class SqlAlchemySportCatalogRepository(SportCatalogRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, sport_id: UUID, club_id: UUID) -> SportCatalog:
        logger.debug("sport_catalog lookup: sport=%s club=%s", sport_id, club_id)
        row = (
            await self._session.execute(
                _GET_CATALOG, {"sport_id": sport_id, "club_id": club_id}
            )
        ).first()
        if row is None:
            logger.warning(
                "No club_sports/sport_catalog join for sport=%s club=%s", sport_id, club_id
            )
            raise SportCatalogNotFoundError(str(sport_id))

        buckets = tuple(
            CatalogBucket(
                key=b["key"],
                positions=tuple(b.get("positions", [])),
            )
            for b in _as_list(row.buckets)
            if b.get("key")
        )
        categories = tuple(
            KpiCategory(
                key=c["key"],
                buckets=tuple(c.get("buckets", [])),
            )
            for c in _as_list(row.kpi_categories)
            if c.get("key")
        )
        kpis = tuple(
            KpiDefinition(
                key=k["key"],
                category_key=k.get("category", ""),
                label=k.get("label_en") or k["key"],
                measures=k.get("measures", ""),
            )
            for k in _as_list(row.kpis)
            if k.get("key")
        )
        if not buckets or not categories or not kpis:
            # A row exists but a fragment is empty/malformed — the player will end up
            # with no KPIs. Surface it loudly since it's a likely misconfiguration.
            logger.warning(
                "Catalog for sport=%s parsed with empty fragment(s): "
                "buckets=%d categories=%d kpis=%d",
                sport_id,
                len(buckets),
                len(categories),
                len(kpis),
            )
        return SportCatalog(buckets=buckets, categories=categories, kpis=kpis)
