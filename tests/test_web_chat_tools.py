"""The agentic search's searches (`miroir.web.toolsearch`) and tool arguments
(`miroir.web.chat_tools`) on the tiny index."""

from __future__ import annotations

import itertools

import pytest

from miroir.eval.bm25 import tokens
from miroir.records.bases import Base, base_of, place_key
from miroir.records.model import Collection
from miroir.web import facets
from miroir.web.engine import Engine, PhotoCosines
from miroir.web.toolsearch import SearchKind, ToolFilters, ToolQuery, ToolSearch
from tests import web_helpers as h


def brute_mask(engine: Engine, filters: ToolFilters) -> list[str]:
    """The entities matching every field, one record at a time (the mask's old loop)."""
    catalog = engine.catalog
    out: list[str] = []
    for row, entity in enumerate(catalog.entity_ids):
        records = catalog.entity_records(row)
        base = base_of(Collection(records[0].collection), records[0].entity_id)
        communes = {place_key(r.commune) for r in records if r.commune}
        depts = {place_key(r.departement) for r in records if r.departement}
        domains = {w for r in records for w in facets_words((*r.domain, *r.denomination))}
        words = {w for r in records for w in facets_words(r.period)}
        broad = {p for r in records for p in facets.periods(r.period)}
        asked_words = set(facets_words(filters.periods)) - {"siecle"}
        if (
            (not filters.bases or base in filters.bases)
            and (not filters.communes or {place_key(c) for c in filters.communes} & communes)
            and (not filters.departements or {place_key(d) for d in filters.departements} & depts)
            and (not filters.domains or set(facets_words(filters.domains)) & domains)
            and (
                not filters.periods
                or set(facets.periods(filters.periods)) & broad
                or asked_words & words
            )
        ):
            out.append(entity)
    return out


def facets_words(values: tuple[str, ...]) -> list[str]:
    return tokens(" ".join(values))


FIELDS = [
    ToolFilters(bases=(Base.JOCONDE,)),
    ToolFilters(bases=(Base.MEMOIRE, Base.ROSE_VALLAND)),
    ToolFilters(communes=("toulouse",)),
    ToolFilters(communes=("Rouen", "Toulouse")),
    ToolFilters(departements=("Seine Maritime",)),
    ToolFilters(domains=("architecture",)),
    ToolFilters(domains=("céramique", "vase")),
    ToolFilters(periods=("18e siècle",)),
    ToolFilters(periods=("années 1960",)),
]


@pytest.mark.parametrize(
    "filters",
    [
        *FIELDS,
        *(
            ToolFilters(
                bases=a.bases or b.bases,
                communes=a.communes or b.communes,
                departements=a.departements or b.departements,
                domains=a.domains or b.domains,
                periods=a.periods or b.periods,
            )
            for a, b in itertools.combinations(FIELDS, 2)
        ),
    ],
)
def test_the_filter_mask_matches_every_field_as_records_say(filters: ToolFilters) -> None:
    engine, _ = h.engine()
    mask = ToolSearch(engine).mask(filters)
    got = [engine.catalog.entity_ids[i] for i, keep in enumerate(mask) if keep]
    assert got == brute_mask(engine, filters)


def test_the_photos_cosines_are_worked_out_once_for_its_searches_and_looks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each image search and look re-embedded the same photo on the GPU."""
    engine, _ = h.engine()
    real = engine.photo_cosines
    embedded: list[bytes] = []

    def counted(data: bytes) -> PhotoCosines:
        embedded.append(data)
        return real(data)

    monkeypatch.setattr(engine, "photo_cosines", counted)
    search = ToolSearch(engine)
    photo, other = h.photo(90), h.photo(130)
    first = search.run(ToolQuery(SearchKind.IMAGE, image=photo, a7_weight=1.0))
    again = search.run(ToolQuery(SearchKind.IMAGE, image=photo, a7_weight=0.0))
    search.run(ToolQuery(SearchKind.VIEW, image=photo, entity_ids=("PA1",)))
    search.run(ToolQuery(SearchKind.IMAGE, image=other))
    assert len(embedded) == 2
    assert [x.entity_id for x in first.hits] == [x.entity_id for x in again.hits]
