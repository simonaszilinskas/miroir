"""A tiny website index for tests: three entities, five images, made-up vectors."""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from miroir.web.catalog import Catalog, CatalogData, CatalogImage, CatalogRecord, Source
from miroir.web.engine import Engine
from miroir.web.fake import NumpyGallery
from miroir.web.routing import Bucket, Guess, PhotoType, Routing

Floats = NDArray[np.float32]


def record(ref: str, entity: str, collection: str, **fields: object) -> CatalogRecord:
    values: dict[str, object] = {
        "title": f"Titre {ref}",
        "description": "",
        "denomination": (),
        "domain": (),
        "authors": (),
        "period": (),
        "region": None,
        "departement": None,
        "commune": None,
    }
    values.update(fields)
    return CatalogRecord(
        ref=ref,
        collection=collection,
        entity_id=entity,
        pop_url=f"https://pop.culture.gouv.fr/notice/{collection}/{ref}",
        **values,  # pyright: ignore[reportArgumentType]
    )


def image(image_id: str, entity: str, ref: str | None, *, cover: bool = True, kind: str = "main"):
    return CatalogImage(
        image_id=image_id,
        entity_id=entity,
        ref=ref,
        url=f"https://example.org/{image_id}.jpg",
        image_type=kind,
        is_cover=cover,
        source=Source.POP if ref else Source.COMMONS,
        licence="" if ref else "CC BY-SA 4.0",
        author="" if ref else "Une autrice",
    )


def catalog(*, pop_images: bool = True) -> Catalog:
    """PA1: a building with two Mémoire photos (two records) and a Commons photo.
    OBJ1: a Joconde object with a cover and a back. OBJ2: a Rose Valland work.
    `pop_images=False`: PA1's two photos come from Commons too (it has no POP image)."""
    return Catalog(
        CatalogData(
            images=(
                image("m1", "PA1", "AP1" if pop_images else None),
                image("m2", "PA1", "AP2" if pop_images else None),
                image("j1", "OBJ1", "J1"),
                image("j2", "OBJ1", "J1", cover=False, kind="back"),
                image("r1", "OBJ2", "RV1"),
                image("commons-c1", "PA1", None, cover=False),
            ),
            records=(
                record(
                    "AP1",
                    "PA1",
                    "memoire",
                    title="Église Saint-Pierre",
                    commune="Toulouse",
                    departement="Haute-Garonne",
                    region="Occitanie",
                    domain=("Architecture",),
                    period=("années 1960",),
                ),
                record(
                    "AP2",
                    "PA1",
                    "memoire",
                    title="Église Saint-Pierre, portail",
                    commune="Toulouse",
                    departement="Haute-Garonne",
                    region="Midi-Pyrénées",
                    domain=("architecture",),
                ),
                record(
                    "J1",
                    "OBJ1",
                    "joconde",
                    title="",
                    denomination=("vase",),
                    commune="Rouen",
                    departement="Seine-Maritime",
                    region="Normandie",
                    domain=("céramique",),
                    period=("18e siècle",),
                ),
                record("RV1", "OBJ2", "rose-valland", title="Nature morte au lièvre"),
            ),
        )
    )


def unit_rows(n: int, dim: int, seed: int = 1) -> Floats:
    rng = np.random.default_rng(seed)
    rows = np.asarray(rng.normal(0.0, 1.0, (n, dim)), dtype=np.float32)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


class RowEncoders:
    """Photos embed as a chosen gallery row (the row number is the photo's first byte)."""

    def __init__(self, matrix: Floats, text_row: int = 4) -> None:
        self.matrix = matrix
        self.text_row = text_row
        self.last_photo: bytes | None = None

    def image(self, jpeg: bytes) -> tuple[Floats, Floats]:
        self.last_photo = jpeg
        row = int(np.asarray(PILImage.open(io.BytesIO(jpeg)).convert("L")).mean() // 40)
        return self.matrix[row], self.matrix[row]

    def text(self, text: str) -> Floats:
        del text
        return self.matrix[self.text_row]


def photo(shade: int, size: tuple[int, int] = (64, 48)) -> bytes:
    """A flat grey photo whose shade picks gallery row shade // 40 in `RowEncoders`."""
    out = io.BytesIO()
    PILImage.new("RGB", size, (shade, shade, shade)).save(out, format="PNG")
    return out.getvalue()


def routing(building_cutoff: float = 0.9, object_cutoff: float = 0.9) -> Routing:
    return Routing(
        buckets=(
            Bucket("building", 1.0, building_cutoff, PhotoType.BUILDING),
            Bucket("object", 0.2, object_cutoff, PhotoType.OBJECT),
        )
    )


def guess(dim: int, building: bool) -> Guess:
    """A guess that always says building (or always object)."""
    return Guess(
        mean=np.zeros(dim),
        scale=np.ones(dim),
        coef=np.zeros(dim),
        bias=10.0 if building else -10.0,
    )


def engine(*, building: bool = True, cutoff: float = 0.9) -> tuple[Engine, RowEncoders]:
    matrix = unit_rows(6, 8)
    encoders = RowEncoders(matrix)
    built = Engine(
        catalog=catalog(),
        routing=routing(cutoff, cutoff),
        guess=guess(8, building),
        encoders=encoders,
        gallery=NumpyGallery(matrix),
    )
    return built, encoders
