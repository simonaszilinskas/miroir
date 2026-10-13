"""What we read from Wikidata and the Commons API, parsed into plain values.

Raw JSON stays in this module; everything after it uses `WikidataItem` and `CommonsFile`.
"""

from __future__ import annotations

import html
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import batched
from urllib.parse import unquote, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, JsonValue

from miroir.commons.client import MAXLAG_S, WikimediaClient
from miroir.config import COMMONS_API_URL, WIKIDATA_SPARQL_URL

logger = logging.getLogger(__name__)

#: Wikidata properties that hold French heritage ids.
MERIMEE = "P380"
JOCONDE = "P347"

#: Width of the thumbnail asked for: a standard Commons size, larger than the 512 px models see.
THUMB_WIDTH = 960
#: Files per category listing. The newest come first.
CATEGORY_FILES = 50
#: Titles per imageinfo call (the API's limit).
TITLES_PER_CALL = 50

_ENTITY = "http://www.wikidata.org/entity/"
_FILE_PATH = "http://commons.wikimedia.org/wiki/Special:FilePath/"
_TAG = re.compile(r"<[^>]+>")
_SPACES = re.compile(r"\s+")
_POP_REF = re.compile(r"/notice/(?:merimee|joconde|palissy|memoire)/([A-Za-z0-9]+)")

#: Metadata fields read from each file page.
_EXTMETADATA = (
    "LicenseShortName",
    "LicenseUrl",
    "Artist",
    "Credit",
    "DateTimeOriginal",
    "Categories",
    "ImageDescription",
)


@dataclass(frozen=True, slots=True)
class WikidataItem:
    qid: str
    ids: tuple[str, ...]
    """Values of the heritage-id property (P380 or P347), sorted."""
    images: tuple[str, ...]
    """P18 files, as "File:..." titles."""
    categories: tuple[str, ...]
    """P373 Commons categories, as "Category:..." titles."""


@dataclass(frozen=True, slots=True)
class CommonsFile:
    title: str
    page_url: str
    url: str
    thumb_url: str
    width: int
    height: int
    mime: str
    licence: str
    licence_url: str
    author: str
    credit: str
    taken: str
    """EXIF DateTimeOriginal, else the date the file page gives."""
    latitude: float | None
    longitude: float | None
    """EXIF GPS only."""
    camera: str
    """EXIF make and model, empty when the file has none."""
    categories: tuple[str, ...]
    description: str
    pop_refs: tuple[str, ...] = ()
    """POP records the file page links to (Mérimée or Joconde template): what it shows."""

    @property
    def year(self) -> int | None:
        match = re.search(r"(1[89]\d\d|20\d\d)", self.taken)
        return int(match[1]) if match else None


@dataclass(frozen=True, slots=True)
class CategoryListing:
    files: tuple[CommonsFile, ...]
    subcategories: tuple[str, ...]


# --- Wikidata -------------------------------------------------------------------


class _Value(BaseModel):
    value: str


class _Bindings(BaseModel):
    bindings: list[dict[str, _Value]]


class _Sparql(BaseModel):
    results: _Bindings


def items_with(client: WikimediaClient, prop: str) -> list[WikidataItem]:
    """Every Wikidata item holding `prop`, with its P18 images and P373 categories."""
    query = (
        f"SELECT ?item ?id ?image ?category WHERE {{ ?item wdt:{prop} ?id . "
        "OPTIONAL { ?item wdt:P18 ?image } OPTIONAL { ?item wdt:P373 ?category } }"
    )
    payload = client.get_json(WIKIDATA_SPARQL_URL, {"query": query, "format": "json"})
    rows = _Sparql.model_validate(payload).results.bindings
    found: defaultdict[str, tuple[set[str], set[str], set[str]]] = defaultdict(
        lambda: (set(), set(), set())
    )
    for row in rows:
        qid = row["item"].value.removeprefix(_ENTITY)
        ids, images, categories = found[qid]
        ids.add(row["id"].value.strip())
        if "image" in row:
            images.add("File:" + unquote(row["image"].value.removeprefix(_FILE_PATH)))
        if "category" in row:
            categories.add("Category:" + row["category"].value.strip())
    items = [
        WikidataItem(qid, tuple(sorted(i)), tuple(sorted(im)), tuple(sorted(c)))
        for qid, (i, im, c) in sorted(found.items())
    ]
    logger.info("Wikidata: %d items with %s", len(items), prop)
    return items


# --- Commons --------------------------------------------------------------------


class _Meta(BaseModel):
    value: JsonValue = None


class _CommonMeta(BaseModel):
    name: str | int
    value: JsonValue = None


class _ImageInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    url: str
    descriptionurl: str
    thumburl: str | None = None
    width: int
    height: int
    mime: str
    extmetadata: dict[str, _Meta] = {}
    commonmetadata: list[_CommonMeta] = []


class _Link(BaseModel):
    url: str


class _Page(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    ns: int
    missing: bool = False
    imageinfo: list[_ImageInfo] = []
    extlinks: list[_Link] = []


class _Query(BaseModel):
    pages: list[_Page] = []


class _Answer(BaseModel):
    query: _Query | None = None


def _imageinfo_params() -> dict[str, str]:
    return {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "maxlag": str(MAXLAG_S),
        "prop": "imageinfo|extlinks",
        "iiprop": "url|size|mime|extmetadata|commonmetadata",
        "iiurlwidth": str(THUMB_WIDTH),
        "iiextmetadatafilter": "|".join(_EXTMETADATA),
        "iiextmetadatalanguage": "en",
        # Links to POP records, which the Mérimée and Joconde templates on file pages make.
        "ellimit": "max",
        "elprotocol": "https",
        "elquery": "www.pop.culture.gouv.fr/notice/",
    }


def category_files(client: WikimediaClient, category: str) -> CategoryListing:
    """The newest files directly in a category, and its subcategories."""
    params = {
        **_imageinfo_params(),
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file|subcat",
        "gcmlimit": str(CATEGORY_FILES),
        "gcmsort": "timestamp",
        "gcmdir": "desc",
    }
    pages = _pages(client.get_json(COMMONS_API_URL, params))
    files = tuple(f for p in pages if (f := _file(p)) is not None)
    subcategories = tuple(sorted(p.title for p in pages if p.ns == 14))
    return CategoryListing(files, subcategories)


def files_by_title(client: WikimediaClient, titles: Iterable[str]) -> dict[str, CommonsFile]:
    """Image info for files named by title, 50 per call. Missing files are left out."""
    found: dict[str, CommonsFile] = {}
    for chunk in batched(sorted(set(titles)), TITLES_PER_CALL):
        params = {**_imageinfo_params(), "titles": "|".join(chunk)}
        for page in _pages(client.get_json(COMMONS_API_URL, params)):
            file = _file(page)
            if file is not None:
                found[file.title] = file
    return found


def _pages(payload: object) -> list[_Page]:
    answer = _Answer.model_validate(payload)
    return [] if answer.query is None else answer.query.pages


def _file(page: _Page) -> CommonsFile | None:
    if page.ns != 6 or page.missing or not page.imageinfo:
        return None
    info = page.imageinfo[0]
    meta = {k: _text(v.value) for k, v in info.extmetadata.items()}
    exif = {str(m.name): m.value for m in info.commonmetadata}
    # Only the camera's own GPS: the file page's location may be the monument's position.
    lat, lon = _exif_gps(exif)
    camera = " ".join(
        dict.fromkeys(_text(exif.get(k)) for k in ("Make", "Model") if _text(exif.get(k)))
    )
    return CommonsFile(
        title=page.title,
        page_url=info.descriptionurl,
        url=_clean_url(info.url),
        thumb_url=_clean_url(info.thumburl or info.url),
        width=info.width,
        height=info.height,
        mime=info.mime,
        licence=meta.get("LicenseShortName", ""),
        licence_url=meta.get("LicenseUrl", ""),
        author=meta.get("Artist", ""),
        credit=meta.get("Credit", ""),
        taken=_text(exif.get("DateTimeOriginal")) or meta.get("DateTimeOriginal", ""),
        latitude=lat,
        longitude=lon,
        camera=camera,
        categories=tuple(c for c in meta.get("Categories", "").split("|") if c),
        description=meta.get("ImageDescription", "")[:500],
        pop_refs=tuple(
            sorted({m[1] for link in page.extlinks if (m := _POP_REF.search(link.url))})
        ),
    )


def _clean_url(url: str) -> str:
    """Drop the tracking query string Commons adds to file URLs."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _text(value: JsonValue) -> str:
    """Plain text of a metadata value: HTML tags and entities gone, spaces collapsed."""
    if value is None or isinstance(value, list | dict):
        return ""
    text = html.unescape(_TAG.sub(" ", str(value)))
    return _SPACES.sub(" ", text).strip()


def _exif_gps(exif: dict[str, JsonValue]) -> tuple[float | None, float | None]:
    """EXIF GPS position in signed decimal degrees, as the API gives it after its own parsing."""
    lat, lon = exif.get("GPSLatitude"), exif.get("GPSLongitude")
    if isinstance(lat, bool) or isinstance(lon, bool):
        return None, None
    if not isinstance(lat, int | float) or not isinstance(lon, int | float):
        return None, None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        return None, None
    return float(lat), float(lon)


def titles(items: Sequence[WikidataItem]) -> list[str]:
    """Every P18 file of these items."""
    return sorted({t for item in items for t in item.images})


class _Coordinate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    lat: float
    lon: float
    primary: bool = False
    type: str = ""


class _CoordinatePage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str
    coordinates: list[_Coordinate] = []


class _CoordinateQuery(BaseModel):
    pages: list[_CoordinatePage] = []


class _CoordinateAnswer(BaseModel):
    query: _CoordinateQuery | None = None


def camera_positions(
    client: WikimediaClient, file_titles: Iterable[str]
) -> dict[str, tuple[float, float]]:
    """(latitude, longitude) where each file was taken, as its page's {{Location}} gives it,
    50 titles per call. Only the camera's position: a file page's "object location" is the
    monument's own point and would give the answer away."""
    found: dict[str, tuple[float, float]] = {}
    for chunk in batched(sorted(set(file_titles)), TITLES_PER_CALL):
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "maxlag": str(MAXLAG_S),
            "prop": "coordinates",
            "coprop": "type|dim|globe",
            "coprimary": "all",
            "colimit": "max",
            "titles": "|".join(chunk),
        }
        answer = _CoordinateAnswer.model_validate(client.get_json(COMMONS_API_URL, params))
        for page in [] if answer.query is None else answer.query.pages:
            camera = [c for c in page.coordinates if c.primary and c.type == "camera"]
            if camera and _valid(camera[0].lat, camera[0].lon):
                found[page.title] = (camera[0].lat, camera[0].lon)
    return found


def _valid(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180 and not (lat == 0 and lon == 0)
