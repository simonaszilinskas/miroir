"""Transformed image searches: the agent crops, turns, mirrors or greys the visitor's photo
before an image search (`miroir.images.transform`, `chat_tools.image_search`)."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Mapping, Sequence

import pytest
from PIL import Image as PILImage

from miroir.images.transform import CELLS, Transform, TransformError, apply, padded
from miroir.outlines import Box
from miroir.web.catalog import CatalogRecord
from miroir.web.chat_tools import (
    MAX_TRANSFORMS,
    Signature,
    ToolName,
    ToolOutcome,
    run_tool,
    running_label,
)
from miroir.web.nearby import Position
from miroir.web.toolsearch import Hit, SearchKind, ToolHits, ToolQuery


def colour_photo(size: tuple[int, int] = (200, 100)) -> bytes:
    """Left half red, right half blue."""
    image = PILImage.new("RGB", size, (220, 30, 30))
    image.paste((30, 30, 220), (size[0] // 2, 0, size[0], size[1]))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def opened(data: bytes) -> PILImage.Image:
    return PILImage.open(io.BytesIO(data)).convert("RGB")


# --- the image operations ----------------------------------------------------------------------


def test_a_crop_keeps_the_part_asked_for() -> None:
    out = opened(apply(colour_photo(), Transform(crop=(0.0, 0.0, 0.5, 1.0)), 1024))
    assert out.size == (100, 100)
    r, _, b = out.getpixel((50, 50))  # pyright: ignore[reportGeneralTypeIssues]
    assert r > 150 > b  # red half only


def test_turn_mirror_and_grey() -> None:
    turned = opened(apply(colour_photo(), Transform(rotate=90), 1024))
    assert turned.size == (100, 200)
    r, _, b = turned.getpixel((50, 10))  # pyright: ignore[reportGeneralTypeIssues]
    assert r > b  # clockwise: the left (red) half comes on top
    mirrored = opened(apply(colour_photo(), Transform(mirror=True), 1024))
    r, _, b = mirrored.getpixel((10, 50))  # pyright: ignore[reportGeneralTypeIssues]
    assert b > r
    grey = opened(apply(colour_photo(), Transform(grayscale=True, stretch=True), 1024))
    r, g, b = grey.getpixel((10, 50))  # pyright: ignore[reportGeneralTypeIssues]
    assert abs(r - g) <= 2
    assert abs(g - b) <= 2


def test_the_long_side_is_capped() -> None:
    assert (
        max(opened(apply(colour_photo((2000, 1000)), Transform(grayscale=True), 512)).size) == 512
    )


def test_padding_stays_in_the_photo_and_tiny_boxes_are_refused() -> None:
    assert padded((0.0, 0.1, 0.5, 0.5), 0.1) == pytest.approx((0.0, 0.06, 0.55, 0.54))
    with pytest.raises(TransformError):
        padded((0.5, 0.1, 0.4, 0.5), 0.0)
    with pytest.raises(TransformError):
        padded((0.1, 0.1, 0.12, 0.5), 0.0)
    with pytest.raises(TransformError):
        apply(colour_photo(), Transform(rotate=45), 512)


def test_the_grid_cells_cover_the_photo_with_overlap() -> None:
    assert len(CELLS) == 9
    assert CELLS["top-left"] == (0.0, 0.0, 0.4, 0.4)
    assert CELLS["bottom-right"] == (0.6, 0.6, 1.0, 1.0)


# --- the tool ----------------------------------------------------------------------------------

OUTLINES = (Box(0.1, 0.2, 0.5, 0.8, 0.9, "sculpture"), Box(0.6, 0.1, 0.9, 0.5, 0.8, "painting"))
HIT = Hit(
    "PM1", 0.9, "palissy", "Retable", "Vétheuil", "", "", "palissy", "Palissy", "", None, None,
    None, 1, (),
)  # fmt: skip


class Ctx:
    """The conversation as the tools see it, with what a search was given."""

    def __init__(self, photo: bytes | None = None, outlines: Sequence[Box] = OUTLINES) -> None:
        self._photo = photo if photo is not None else colour_photo()
        self._outlines = outlines
        self.queries: list[ToolQuery] = []
        self.kept: dict[str, bytes] = {}
        self.transforms = 0
        self.signatures: list[Signature] = []

    @property
    def photo(self) -> bytes | None:
        return self._photo

    @property
    def region(self) -> bytes | None:
        return None

    @property
    def position(self) -> Position | None:
        return None

    async def search(self, query: ToolQuery) -> ToolHits:
        self.queries.append(query)
        return ToolHits((HIT,))

    def remember(self, hits: Sequence[Hit]) -> None:
        del hits

    def seen(self, entity_id: str) -> bool:
        return entity_id == "PM1"

    async def thumbnail(self, hit: Hit) -> str | None:
        del hit
        return None

    def repeats(self, signature: Signature) -> int | None:
        for n, earlier in enumerate(self.signatures, start=1):
            if earlier.same(signature):
                return n
        self.signatures.append(signature)
        return None

    def viewed_image(self, entity_id: str) -> str | None:
        del entity_id
        return None

    def checked(self, entity_id: str) -> bool:
        del entity_id
        return False

    def title_of(self, entity_id: str) -> str:
        return entity_id

    def records(self, entity_id: str) -> Sequence[CatalogRecord]:
        del entity_id
        return ()

    async def outlines(self) -> Sequence[Box]:
        return self._outlines

    def keep_query_image(self, jpeg: bytes) -> str:
        key = f"query-{len(self.kept) + 1}"
        self.kept[key] = jpeg
        return key

    def use_transform(self) -> bool:
        if self.transforms >= MAX_TRANSFORMS:
            return False
        self.transforms += 1
        return True


def search(ctx: Ctx, transform: Mapping[str, object] | None, **extra: object) -> ToolOutcome:
    args: dict[str, object] = {"why": "Je vise une partie.", **extra}
    if transform is not None:
        args["transform"] = transform
    return asyncio.run(run_tool(ctx, ToolName.IMAGE_SEARCH.value, json.dumps(args)))


def test_an_outline_crop_in_black_and_white_searches_that_part_grey() -> None:
    ctx = Ctx()
    outcome = search(ctx, {"outline": 1, "grayscale": True, "pad": 0})
    assert not outcome.error, outcome.text
    assert outcome.step is not None
    assert outcome.step.label.startswith(
        "Recadre sur le contour 1 (sculpture), passe en noir et blanc, puis cherche par image"
    )
    searched = opened(ctx.queries[0].image or b"")
    assert searched.size == (80, 60)  # 0.4 x 0.6 of 200 x 100
    r, _, b = searched.getpixel((10, 30))  # pyright: ignore[reportGeneralTypeIssues]
    assert abs(r - b) <= 2  # grey
    assert ctx.queries[0].kind is SearchKind.IMAGE
    assert "cropped to outline 1 (sculpture), in black and white" in outcome.text
    assert [t.image_id for t in outcome.thumbnails] == ["query-1"]
    assert max(opened(ctx.kept["query-1"]).size) <= 384


@pytest.mark.parametrize(
    ("transform", "label"),
    [
        ({"cell": "top-left"}, "Recadre sur la case en haut à gauche"),
        ({"box": [0.1, 0.1, 0.6, 0.9]}, "Recadre sur une partie de la photo"),
        ({"rotate": 90, "mirror": True}, "Tourne d’un quart de tour à droite, retourne en miroir"),
        ({"stretch": True}, "Passe en noir et blanc contrasté"),
    ],
)
def test_each_transform_says_what_it_did_in_french(
    transform: Mapping[str, object], label: str
) -> None:
    outcome = search(Ctx(), transform)
    assert outcome.step is not None
    assert outcome.step.label.startswith(label)


def test_no_transform_searches_the_whole_photo_as_before() -> None:
    ctx = Ctx()
    outcome = search(ctx, None)
    assert outcome.step is not None
    assert outcome.step.label == "Recherche par image (réglage automatique)"
    assert ctx.queries[0].image == ctx.photo
    assert not outcome.thumbnails
    assert ctx.transforms == 0
    assert search(ctx, {}).error  # the same search again: refused as a repeat


@pytest.mark.parametrize(
    ("transform", "error"),
    [
        ({"outline": 3}, "outline must be a number from 1 to 2"),
        ({"outline": 1, "cell": "top-left"}, "not several"),
        ({"box": [0.5, 0.1, 0.2, 0.9]}, "within 0 to 1"),
        ({"box": [0.1, 0.1]}, "box must be"),
        ({"cell": "corner"}, "unknown cell"),
        ({"rotate": 45}, "rotate must be one of"),
        ({"zoom": 2}, "unknown transform key"),
        ({"outline": 1, "pad": 0.9}, "pad must be"),
    ],
)
def test_bad_transforms_come_back_as_errors_the_model_can_read(
    transform: Mapping[str, object], error: str
) -> None:
    ctx = Ctx()
    outcome = search(ctx, transform)
    assert outcome.error
    assert error in outcome.text
    assert not ctx.queries


def test_without_outlines_an_outline_crop_is_refused() -> None:
    outcome = search(Ctx(outlines=()), {"outline": 1})
    assert "found no outline" in outcome.text


def test_transformed_searches_are_capped_per_message() -> None:
    ctx = Ctx()
    cells = list(CELLS)
    for cell in cells[:MAX_TRANSFORMS]:
        assert not search(ctx, {"cell": cell}).error
    refused = search(ctx, {"cell": cells[MAX_TRANSFORMS]})
    assert refused.error
    assert "no transformed searches left" in refused.text
    assert len(ctx.queries) == MAX_TRANSFORMS


def test_the_running_step_names_the_transform() -> None:
    args = {"transform": {"outline": 3, "grayscale": True}}
    assert running_label("image_search", args) == (
        "Recadre sur le contour 3, passe en noir et blanc, puis cherche par image…"
    )
    assert running_label("image_search", {}) is None
    assert running_label("text_search", args) is None
