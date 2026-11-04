"""A photo someone sends, decoded once: its pixels upright and its chroma spread.

The site reads a visitor's photo for the mount, the chroma and the query copy; each used to
decode the upload again. `open_photo` decodes it once and refuses what it cannot read with
`UnreadablePhotoError`, the one error the site turns into "Cette image n'a pas pu être lue".
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image as PILImage
from PIL import ImageOps

from miroir.images.colour import chroma_of

#: Largest photo a visitor may send, in pixels. The page sends at most 1,024 px a side; a
#: small file can still claim a huge size and take gigabytes to decode.
MAX_UPLOAD_PIXELS = 40_000_000
#: Formats a visitor's photo may be in (the page sends JPEG; tests and tools send PNG).
UPLOAD_FORMATS = ("JPEG", "PNG", "WEBP")
#: What Pillow raises on a file it cannot decode.
_PILLOW_ERRORS = (OSError, SyntaxError, EOFError, ValueError, PILImage.DecompressionBombError)


class UnreadablePhotoError(ValueError):
    """The bytes are not a photo that can be read, or one too large to decode."""


@dataclass(frozen=True, slots=True)
class OpenedPhoto:
    upright: PILImage.Image
    """The decoded pixels, turned as the camera's EXIF orientation says."""
    chroma: float
    """`colour.chroma_spread` of the photo as stored (before turning)."""


def open_photo(
    data: bytes,
    *,
    max_pixels: int | None = None,
    formats: tuple[str, ...] | None = None,
) -> OpenedPhoto:
    """Decode `data` once. `max_pixels` and `formats` bound what a visitor may send; the
    defaults (None) take any image Pillow reads, as gallery files and eval photos need."""
    try:
        image = PILImage.open(io.BytesIO(data), formats=formats)
        if max_pixels is not None and image.width * image.height > max_pixels:
            msg = f"photo of {image.width} x {image.height} px is over {max_pixels:,} px"
            raise UnreadablePhotoError(msg)
        image.load()
        chroma = chroma_of(image)
        ImageOps.exif_transpose(image, in_place=True)
    except _PILLOW_ERRORS as err:
        if isinstance(err, UnreadablePhotoError):
            raise
        msg = f"cannot read the photo: {err}"
        raise UnreadablePhotoError(msg) from err
    return OpenedPhoto(upright=image, chroma=chroma)


def open_upload(data: bytes) -> OpenedPhoto:
    """A visitor's photo: at most `MAX_UPLOAD_PIXELS`, in one of `UPLOAD_FORMATS`."""
    return open_photo(data, max_pixels=MAX_UPLOAD_PIXELS, formats=UPLOAD_FORMATS)
