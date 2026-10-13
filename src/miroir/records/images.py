"""Image rules shared by all mappers: ids, URLs, types and skips (technical spec §3.3)."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import StrEnum
from urllib.parse import quote

from miroir.config import POP_ASSET_HOST
from miroir.records.model import ImageType


class ImageSkip(StrEnum):
    """Why an image path from an export does not become an image."""

    PLACEHOLDER = "placeholder"
    """Empty path, or the shared 1x1 GIF the export puts in `imageUrl` when there is no image."""
    ON_REQUEST = "on_request"
    """`sur_demande.jpg`: POP shows the image on request only."""
    DUPLICATE = "duplicate"
    """Same path listed twice in one record."""
    NOT_AN_IMAGE = "not_an_image"
    """A web address, a bare file name or a file that is no picture (PDF, web page). The full
    export's lists hold a few, from the bucket listing or old Mérimée links."""
    RENAMED_COPY = "renamed_copy"
    """Another copy of an image the record already holds, under a name the POP bucket gives
    copies: `M500145_0001074_P.JPG` or `M040000_x_1.jpg` next to `0001074.jpg`, or the same
    name in other letter case. Only full-export Joconde records have them (spec §3.4)."""


#: File extensions that are no picture. Others are kept: some real images have odd or no
#: extension ("…_v1", "portrait. lebasque").
NOT_IMAGE_EXTENSIONS = frozenset(
    {"pdf", "avi", "mp4", "mov", "zip", "htm", "html", "doc", "docx", "txt", "xml"}
)


# First match wins. Specific content (label, inscription) beats where it is (back),
# which beats how it is framed (detail): "detail-etiquette-revers" is a label.
_TYPE_RULES: tuple[tuple[ImageType, re.Pattern[str]], ...] = (
    (ImageType.LABEL, re.compile(r"etiquette|cachet|marque")),
    (ImageType.INSCRIPTION, re.compile(r"inscription|signature")),
    # "dos" alone is too short to match as a substring: skip "calvados", "dossier".
    # The Louvre names views F-ENS / F-DET (front, whole / detail) and R-ENS / R-DET (back).
    (
        ImageType.BACK,
        re.compile(r"revers|verso|(?<!calva)dos(?![a-z])|(?<![a-z])r-(?:ens|det)(?![a-z])"),
    ),
    (ImageType.DETAIL, re.compile(r"detail|(?<![a-z])f-det(?![a-z])")),
)


def image_type_from_path(path: str) -> ImageType:
    """Tag an image from keywords in its file name (accents and case ignored)."""
    name = _fold(path.rsplit("/", 1)[-1])
    for image_type, pattern in _TYPE_RULES:
        if pattern.search(name):
            return image_type
    return ImageType.MAIN


def skip_reason(path: str) -> ImageSkip | None:
    """Return why this export image path is skipped, or None to keep it."""
    if not path.strip() or path.startswith("data:"):
        return ImageSkip.PLACEHOLDER
    name = path.rsplit("/", 1)[-1]
    extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if "://" in path or "/" not in path.strip("/") or extension in NOT_IMAGE_EXTENSIONS:
        return ImageSkip.NOT_AN_IMAGE
    if path.rsplit("/", 1)[-1].lower() == "sur_demande.jpg":
        return ImageSkip.ON_REQUEST
    return None


_COPY_NAME_RE = re.compile(r"^m\d+_(.+?)(?:_(?:p|v|\d+))?$")


def copy_key(path: str) -> str:
    """The name an image and its bucket copies share: `M500145_0001074_P.JPG` and
    `0001074.jpg` both give "0001074". Letter case, accents and extension are ignored."""
    stem = _fold(path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
    match = _COPY_NAME_RE.match(stem)
    return match.group(1) if match else stem


def is_copy_name(path: str) -> bool:
    """True for a name the bucket gives copies (`M<digits>_<name>_P.JPG`)."""
    return _COPY_NAME_RE.match(_fold(path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]) is not None


def asset_url(path: str) -> str:
    """Percent-encoded URL of an export image path on the POP asset bucket."""
    return f"{POP_ASSET_HOST}/{quote(path.lstrip('/'), safe='/')}"


def make_image_id(ref: str, url: str) -> str:
    """Stable id: the same record and URL always give the same id, whatever the row order."""
    return hashlib.sha1(f"{ref}\n{url}".encode(), usedforsecurity=False).hexdigest()[:16]


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))
