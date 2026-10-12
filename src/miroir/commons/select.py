"""Which Commons files may stand in for a visitor's photo, and which few to keep per entity.

A file passes when it looks like a recent camera photo: a JPEG with the camera's make and
model in its EXIF, taken in `MIN_YEAR` or later, not a scan, map, drawing or old postcard
(file title and categories), and not linked on its page to another POP record.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from miroir.commons.api import CommonsFile
from miroir.commons.model import Source, View

#: Photos taken before this year are left out: visitors upload recent colour photos.
MIN_YEAR = 2000
#: Shorter side, in pixels, below which a file is too small to be a useful query.
MIN_SHORT_SIDE = 480


class Kind(StrEnum):
    """What the entity is: the keywords that mean "not a photo of it" differ."""

    BUILDING = "building"
    ARTWORK = "artwork"


class Reject(StrEnum):
    """Why a Commons file is not used as a query."""

    NOT_JPEG = "not_jpeg"
    """PNG, SVG, TIFF, PDF...: drawings, maps and scans far more often than photos."""
    TOO_SMALL = "too_small"
    NO_CAMERA = "no_camera"
    """No EXIF make or model: a scan, an export or a reproduction, not a camera photo."""
    SCANNER = "scanner"
    """EXIF names a scanner or a copy-stand digital back (Phase One, Leaf, Sinar)."""
    NO_DATE = "no_date"
    OLD = "old"
    """Taken before MIN_YEAR."""
    NOT_A_PHOTO = "not_a_photo"
    """Title or categories say plan, map, drawing, engraving, postcard, scan..."""
    ARCHIVE = "archive"
    """An archive print (Médiathèque du patrimoine, Paris Musées, Gallica uploads, or an author
    credited with dates before MIN_YEAR): perhaps our own image."""
    OTHER_RECORD = "other_record"
    """The file page links to POP records, none of them the answer."""
    NEAR_COPY = "near_copy"
    """T2b: near-identical to the Joconde image (perceptual hash)."""
    NO_THUMBNAIL = "no_thumbnail"
    """T2b: the thumbnail could not be fetched, so the copy check cannot run."""


#: Scanners, and the digital backs museums and archives use on copy stands: a file made with one
#: is a reproduction of a print or a page, whatever the date in its EXIF.
_SCANNER = re.compile(
    r"scan|perfection|expression|coolscan|opticfilm|plustek|reflecta|xerox|bizhub|"
    r"\bfi-\d|\bgt-\d|\bds-\d|\bmfc|imagerunner|workcentre|phase one|\bleaf\b|sinar|cruse"
)
_NOT_A_PHOTO_ANY = (
    r"maps?|cartes? postales?|postcards?|cpa|cadastre|scans?|scanned|numerise|"
    r"gravures?|engravings?|lithograph\w*|estampes?|illustrations?|"
    r"historical images|old photographs?|photographies? anciennes?|vintage|"
    r"glass plates?|plaques? de verre|autochromes?|stereo\w*|negatives?|"
    r"books?|livres?|pages?|journal|newspapers?|diagrams?|schemas?"
)
_NOT_A_PHOTO_BUILDING = (
    r"plans?|drawings?|dessins?|sketch\w*|croquis|aquarelles?|watercolou?rs?|"
    r"paintings?|peintures?|tableaux?|elevations?|releves?|blasons?|coats? of arms|logos?"
)
_NOT_A_PHOTO: dict[Kind, re.Pattern[str]] = {
    Kind.BUILDING: re.compile(rf"\b({_NOT_A_PHOTO_ANY}|{_NOT_A_PHOTO_BUILDING})\b"),
    Kind.ARTWORK: re.compile(rf"\b({_NOT_A_PHOTO_ANY})\b"),
}
_ARCHIVE = re.compile(
    r"mediatheque de l.architecture|mediatheque du patrimoine|archives (departementales|"
    r"nationales|municipales|communales)|base memoire|images from the archives|"
    r"images from paris musees|gallica|btv1b|bibliotheque nationale"
)
_YEAR = re.compile(r"\b(1[0-9]\d\d|20\d\d)\b")


def _old_author(author: str) -> bool:
    """The credit gives the author's dates, all before MIN_YEAR: "Atget, Eugène (1857-1927)"."""
    years = [int(y) for y in _YEAR.findall(author)]
    return bool(years) and max(years) < MIN_YEAR


_INTERIOR = re.compile(
    r"\b(interieurs?|interiors?|inside|nefs?|naves?|choeur|choir|chancel|cryptes?|crypts?|"
    r"transepts?|voutes?|vaults?|retables?|altarpieces?|altars?|autels?|orgues?|organs?|"
    r"tribunes?|stalles|chaires?|pulpits?|staircases?|escaliers?)\b"
)
_DETAIL = re.compile(
    r"\b(details?|portails?|portals?|chapiteaux|chapiteau|capitals?|gargouilles?|"
    r"gargoyles?|vitraux|vitrail|stained glass|tympans?|tympanums?|sculptures?|statues?|"
    r"bas-reliefs?|reliefs?|inscriptions?|plaques?|cadrans?|sundials?|modillons?|corbels?|"
    r"doors?|portes?|windows?|fenetres?|carvings?)\b"
)


def fold(text: str) -> str:
    """Lower case without accents, for keyword rules."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _words(file: CommonsFile) -> str:
    name = file.title.removeprefix("File:").rsplit(".", 1)[0].replace("_", " ")
    return fold(" ".join((name, *file.categories)))


def photo_problem(file: CommonsFile, entity_id: str, kind: Kind) -> Reject | None:
    """Why `file` cannot stand in for a visitor's photo of `entity_id`, or None if it can.

    The first rule that fails is the reason.
    """
    words = _words(file)
    year = file.year
    checks = (
        (Reject.NOT_JPEG, file.mime != "image/jpeg"),
        (Reject.TOO_SMALL, min(file.width, file.height) < MIN_SHORT_SIDE),
        (Reject.OTHER_RECORD, bool(file.pop_refs) and entity_id not in file.pop_refs),
        (Reject.ARCHIVE, bool(_ARCHIVE.search(words)) or _old_author(file.author)),
        (Reject.NOT_A_PHOTO, bool(_NOT_A_PHOTO[kind].search(words))),
        (Reject.NO_CAMERA, not file.camera),
        (Reject.SCANNER, bool(_SCANNER.search(fold(file.camera)))),
        (Reject.NO_DATE, year is None),
        (Reject.OLD, year is not None and year < MIN_YEAR),
    )
    return next((reason for reason, failed in checks if failed), None)


def view_of(file: CommonsFile, subcategory: str = "") -> View:
    """Interior, detail or exterior, from the subcategory name or else the file's words."""
    for words in (fold(subcategory), _words(file)):
        if _INTERIOR.search(words):
            return View.INTERIOR
        if _DETAIL.search(words):
            return View.DETAIL
    return View.EXTERIOR


#: Words too common in category names to tell one place from another.
_COMMON_WORDS = frozenset(
    [
        "category",
        "eglise",
        "chapelle",
        "chateau",
        "maison",
        "saint",
        "sainte",
        "paris",
        "notre",
        "dame",
        "hotel",
        "ancien",
        "ancienne",
        "abbaye",
        "cathedrale",
        "musee",
        "beaux",
        "arts",
        "france",
        "interior",
        "exterior",
        "details",
        "place",
        "pont",
        "porte",
        "tour",
        "fontaine",
        "croix",
        "ville",
        "palais",
        "prieure",
        "couvent",
        "manoir",
        "ferme",
        "moulin",
        "church",
        "castle",
        "house",
        "chapel",
        "cathedral",
        "abbey",
        "museum",
        "with",
        "from",
        "dans",
        "pour",
        "village",
    ]
)
_WORD = re.compile(r"[a-z]{4,}")


def names_place(file: CommonsFile, category: str) -> bool:
    """The file title holds a distinctive word of the category name ("gervais", "benest")."""
    wanted = set(_WORD.findall(fold(category))) - _COMMON_WORDS
    title = set(_WORD.findall(fold(file.title)))
    return bool(wanted & title)


#: Words a file title may hold besides the commune and still say nothing of the building.
_FILLER = frozenset(
    ["de", "la", "le", "les", "du", "des", "d", "l", "en", "sur", "et", "img", "dsc"]
)
_TITLE_WORD = re.compile(r"[a-z]+")


def only_commune(file: CommonsFile, commune: str) -> bool:
    """The title holds only the commune's name and numbers ("Torsac 3", "Mazangé (11)").

    Such a file shows somewhere in the commune, not always the building (label check,
    docs/results.md, filter B).
    """
    name = file.title.removeprefix("File:").rsplit(".", 1)[0]
    words = set(_TITLE_WORD.findall(fold(name))) - _FILLER
    town = set(_TITLE_WORD.findall(fold(commune))) - _FILLER
    return bool(town) and words <= town


def is_view_subcategory(name: str) -> bool:
    """A subcategory worth opening for interior and detail photos."""
    words = fold(name)
    return bool(_INTERIOR.search(words) or _DETAIL.search(words))


@dataclass(frozen=True, slots=True)
class Candidate:
    file: CommonsFile
    source: Source
    category: str
    view: View
    named: bool = False
    """The file title shares a distinctive word with the item's category name."""

    @property
    def confirmed(self) -> bool:
        """The file page links to a POP record (only the answer's get this far), or it is P18."""
        return bool(self.file.pop_refs) or self.source == Source.P18

    @property
    def visit(self) -> tuple[str, str]:
        """Author and day: photos sharing both are likely frames of one visit."""
        return (self.file.author, self.file.taken[:10])


def rank(candidates: Sequence[Candidate], rng: random.Random) -> list[Candidate]:
    """Seeded shuffle, then confirmed files first, then named ones. One file appears once.

    Categories also hold photos of neighbouring buildings (a town hall in a church's
    category); a file whose title names the place is less often one of those.
    """
    by_title: dict[str, Candidate] = {}
    for c in sorted(candidates, key=lambda c: (c.file.title, c.source != Source.P18)):
        by_title.setdefault(c.file.title, c)  # a file that is also P18 counts as P18
    unique = list(by_title.values())
    rng.shuffle(unique)
    return sorted(unique, key=lambda c: (not c.confirmed, not c.named))


def pick(ranked: Sequence[Candidate], cap: int) -> list[Candidate]:
    """Up to `cap` files: one per view first, then other visits, then anything left."""
    chosen: list[Candidate] = []
    visits: set[tuple[str, str]] = set()

    def take(c: Candidate) -> None:
        chosen.append(c)
        visits.add(c.visit)

    for view in (View.EXTERIOR, View.INTERIOR, View.DETAIL, View.ARTWORK):
        match = next((c for c in ranked if c.view == view and c.visit not in visits), None)
        if match is not None and len(chosen) < cap:
            take(match)
    for c in ranked:
        if len(chosen) < cap and c not in chosen and c.visit not in visits:
            take(c)
    for c in ranked:
        if len(chosen) < cap and c not in chosen:
            take(c)
    return chosen
