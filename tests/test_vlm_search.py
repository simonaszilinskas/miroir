from __future__ import annotations

from miroir.config import Settings
from miroir.eval.vlm import query_text, text_gallery
from miroir.jobs.describe import Description
from miroir.records.build import build
from miroir.records.store import read_records


def test_query_text_doubles_name_and_commune_and_can_drop_visible_text() -> None:
    d = Description(kind="église", name="Saint-Martin", commune="Beaune", visible_text="MAIRIE")
    assert query_text(d) == "Saint-Martin Saint-Martin église Beaune Beaune MAIRIE"
    assert "MAIRIE" not in query_text(d, visible_text=False)
    assert query_text(None) == ""


def test_text_search_ranks_the_record_named_first(
    settings: Settings,
) -> None:
    build(settings)
    records = read_records(settings.records_parquet, settings.images_parquet)
    per_entity = {r.entity_id: sum(x.entity_id == r.entity_id for x in records) for r in records}
    with_two = next(
        r for r in records if len(r.images) >= 2 and r.title and per_entity[r.entity_id] == 1
    )
    image_ids = [i.image_id for r in records for i in r.images]
    gallery = text_gallery(records, image_ids)
    asked = with_two.images[0].image_id
    texts = {asked: f"{with_two.title} {with_two.description}"}

    per_image = gallery.index.scores(texts[asked])[gallery.record_rows]
    assert gallery.entity_ids[int(per_image.argmax())] == with_two.entity_id
