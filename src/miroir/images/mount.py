"""Find the picture inside an archive mount: a print glued on a card with a typed caption.

No model: the card is a plain, light, even paper, and the picture is the one large block of
"not paper" on it. The steps, on a small grey copy of the photo:

1. The paper's tone is the median of a band along the photo's edges; the band must be
   light and even, or there is no mount.
2. Each cell of a coarse grid counts as picture when most of its pixels differ from the paper.
   Typed or written lines are thin, so a caption leaves most of its cells paper.
3. The largest connected block of picture cells, holes and all, gives a box; its edges are
   then moved to the exact rows and columns where the paper stops.
4. The box must be a real part of the photo (not a sliver, not nearly all of it), and what
   lies outside it must be mostly paper, with writing on it: cells with a few thin marks
   (`TEXT_CELLS`), little dark area (a museum object's shadow or scale bar is dark).
5. The photo is grey (`MAX_CHROMA`): Mémoire prints and their cards are black and white,
   while a museum object on a plain background, which passes 1 to 4, is in colour.

On the tune halves (2026-10-13) this finds a mount on 133 of 138 T1 archive photos, and on
3 of 488 T5 object photos, 4 of 291 T2b, 1 of 99 T3, none of 516 T2 or 150 T8c.

`find_picture` returns the box in the photo's own pixels, or None for a photo with no mount
(a phone photo of a building, a bare scan, an object on a plain background).
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageOps

from miroir.images.colour import chroma_spread

#: The photo is studied at this long side, in pixels.
WORK_SIDE = 384
#: Cell side of the coarse grid, in work pixels.
CELL = 8
#: Width of the edge band that gives the paper's tone, as a share of the short side.
BAND = 0.03
#: The paper is at least this light (0-255) ...
PAPER_MIN = 140.0
#: ... and this even: share of the band within `INK_DIFF` of its median.
PAPER_EVEN = 0.7
#: A pixel is not paper when it differs from the paper's tone by more than this.
INK_DIFF = 28.0
#: A cell is picture when this share of its pixels is not paper.
CELL_INK = 0.5
#: A row or column of the box's edge is picture when this share of it is not paper.
EDGE_INK = 0.5
#: The picture covers between these shares of the photo.
MIN_AREA, MAX_AREA = 0.08, 0.9
#: Outside the box, at least this share of the pixels is paper.
OUTSIDE_PAPER = 0.8
#: Outside the box, a cell holds writing when this share of its pixels is not paper ...
TEXT_INK = (0.04, 0.5)
#: ... and a caption fills this many such cells (a T1 card: 35 to 341).
TEXT_CELLS = (25, 300)
#: Outside the box, at most this share of its pixels is darker than the paper (T1: 3% median).
MAX_DARK_OUTSIDE = 0.08
#: The photo's chroma spread (`images.colour`) is below this: a grey print.
MAX_CHROMA = 4.0

Floats = NDArray[np.float32]
Bools = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Picture:
    """Where the picture lies in the photo, in its pixels (left, top, right, bottom)."""

    left: int
    top: int
    right: int
    bottom: int
    width: int
    """Width and height of the whole photo."""
    height: int

    @property
    def area_share(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top) / (self.width * self.height)

    def scaled(self, width: int, height: int) -> Picture:
        """The same box on a copy of the photo at another size."""
        sx, sy = width / self.width, height / self.height
        return Picture(
            left=round(self.left * sx),
            top=round(self.top * sy),
            right=round(self.right * sx),
            bottom=round(self.bottom * sy),
            width=width,
            height=height,
        )


def find_picture(jpeg: bytes) -> Picture | None:
    """The picture inside the photo's mount, or None when the photo has no mount."""
    if chroma_spread(jpeg) >= MAX_CHROMA:
        return None
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        upright = ImageOps.exif_transpose(opened)
        width, height = upright.size
        small = upright.convert("L")
    small.thumbnail((WORK_SIDE, WORK_SIDE), PILImage.Resampling.BILINEAR)
    grey = np.asarray(small, dtype=np.float32)
    box = picture_box(grey)
    if box is None:
        return None
    top, bottom, left, right = box
    found = Picture(left, top, right, bottom, grey.shape[1], grey.shape[0])
    return found.scaled(width, height)


def crop_picture(jpeg: bytes, picture: Picture) -> bytes:
    """The picture alone, as a JPEG (upright, at the photo's own resolution)."""
    with PILImage.open(io.BytesIO(jpeg)) as opened:
        upright = ImageOps.exif_transpose(opened)
        box = picture.scaled(*upright.size)
        cut = upright.crop((box.left, box.top, box.right, box.bottom)).convert("RGB")
    out = io.BytesIO()
    cut.save(out, format="JPEG", quality=92)
    return out.getvalue()


def picture_box(grey: Floats) -> tuple[int, int, int, int] | None:
    """(top, bottom, left, right) of the picture in a grey image, or None (see module doc)."""
    h, w = grey.shape
    paper = _paper_tone(grey) if min(h, w) >= 4 * CELL else None
    if paper is None:
        return None
    ink = np.abs(grey - paper) > INK_DIFF
    block = _largest_block(_cells(ink))
    if block is None:
        return None
    rows, cols = block
    top, bottom = rows[0] * CELL, min((rows[1] + 1) * CELL, h)
    left, right = cols[0] * CELL, min((cols[1] + 1) * CELL, w)
    top, bottom = _edges(ink[:, left:right].mean(axis=1, dtype=np.float32), top, bottom)
    left, right = _edges(ink[top:bottom].mean(axis=0, dtype=np.float32), left, right)
    inside = np.zeros_like(ink)
    inside[top:bottom, left:right] = True
    dark = grey < paper - INK_DIFF
    return (top, bottom, left, right) if _mount_around(ink, dark, inside) else None


def _mount_around(ink: Bools, dark: Bools, inside: Bools) -> bool:
    """The picture's box is a real part of the photo, on paper that carries writing."""
    if not MIN_AREA <= inside.mean() <= MAX_AREA:
        return False
    outside = ~inside
    pixels = int(outside.sum())
    paper = 1 - int((ink & outside).sum()) / pixels
    dark_share = int((dark & outside).sum()) / pixels
    return (
        paper >= OUTSIDE_PAPER
        and dark_share <= MAX_DARK_OUTSIDE
        and TEXT_CELLS[0] <= _text_cells(ink, inside) <= TEXT_CELLS[1]
    )


def _text_cells(ink: Bools, inside: Bools) -> int:
    """Cells wholly outside the picture whose ink share looks like writing."""
    share = _cell_share(ink)
    touched = _cell_share(inside) > 0
    low, high = TEXT_INK
    return int(((share > low) & (share < high) & ~touched).sum())


def _paper_tone(grey: Floats) -> float | None:
    h, w = grey.shape
    band = max(2, round(BAND * min(h, w)))
    edge = np.concatenate(
        [
            grey[:band].ravel(),
            grey[-band:].ravel(),
            grey[:, :band].ravel(),
            grey[:, -band:].ravel(),
        ]
    )
    tone = float(np.median(edge))
    if tone < PAPER_MIN or (np.abs(edge - tone) <= INK_DIFF).mean() < PAPER_EVEN:
        return None
    return tone


def _cell_share(mask: Bools) -> Floats:
    h, w = mask.shape
    rows, cols = h // CELL, w // CELL
    grid = mask[: rows * CELL, : cols * CELL].reshape(rows, CELL, cols, CELL)
    return np.asarray(grid.mean(axis=(1, 3)), dtype=np.float32)


def _cells(ink: Bools) -> Bools:
    return np.asarray(_cell_share(ink) >= CELL_INK)


def _largest_block(cells: Bools) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Row and column span of the largest 4-connected group of picture cells."""
    seen = np.zeros_like(cells)
    best: tuple[int, tuple[tuple[int, int], tuple[int, int]]] | None = None
    rows, cols = cells.shape
    for r0, c0 in zip(*np.nonzero(cells), strict=True):
        if seen[r0, c0]:
            continue
        stack = [(int(r0), int(c0))]
        seen[r0, c0] = True
        size = 0
        r_min = r_max = int(r0)
        c_min = c_max = int(c0)
        while stack:
            r, c = stack.pop()
            size += 1
            r_min, r_max = min(r_min, r), max(r_max, r)
            c_min, c_max = min(c_min, c), max(c_max, c)
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and cells[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        if best is None or size > best[0]:
            best = (size, ((r_min, r_max), (c_min, c_max)))
    return None if best is None else best[1]


def _edges(profile: Floats, start: int, stop: int) -> tuple[int, int]:
    """Move a span's ends to where `profile` (ink share per row or column) crosses
    `EDGE_INK`, looking one cell either side of each end."""
    n = len(profile)
    lo = max(0, start - CELL)
    hi = min(n, start + CELL)
    inside = np.flatnonzero(profile[lo:hi] >= EDGE_INK)
    new_start = lo + int(inside[0]) if inside.size else start
    lo = max(0, stop - CELL)
    hi = min(n, stop + CELL)
    inside = np.flatnonzero(profile[lo:hi] >= EDGE_INK)
    new_stop = lo + int(inside[-1]) + 1 if inside.size else stop
    return new_start, max(new_stop, new_start + 1)
