"""Example photos on the home page: one tap runs a search (screen 1).

Chosen from the tune halves of the test sets (technical spec §7) among photos the stack
finds: Commons photos of buildings and museum works (T2, T2b) and collection photos (T1, T3,
T5). The server fetches each one (`/api/examples/<id>/image`), so the page can crop it like
an uploaded photo. Commons photos carry their author and licence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Example:
    id: str
    label: str
    kind: str
    """"Bâtiment", "Peinture", "Objet"..."""
    url: str
    credit: str
    frame: bool = False
    """Open the photo screen so the person draws a box, instead of searching at once."""


_COMMONS = "https://upload.wikimedia.org/wikipedia/commons/thumb"
_THUMB = "https://thumb.wikimedia.org/wikipedia/commons/thumb"

# Each comment says which test query the photo is and what the stack gives on the
# screening-v2 index (tune half; `report.json` of `miroir web build-index`).
EXAMPLES: tuple[Example, ...] = (
    # T2, PA00094687: right building first; below the building cut-off (0.865 < 0.921).
    Example(
        "moulin",
        "Moulin de Mascarville",
        "Monument",
        f"{_THUMB}/3/3f/Mascarville_-_Le_Moulin.jpg/960px-Mascarville_-_Le_Moulin.jpg",
        "Didier Descouens, CC BY-SA 4.0, Wikimedia Commons",
    ),
    # T2b, Joconde 05630030670: right record first, above the object cut-off.
    Example(
        "masque",
        "Masque de théâtre",
        "Objet de musée",
        f"{_THUMB}/f/fd/MSR-D-69-2-48-ET.JPG/960px-MSR-D-69-2-48-ET.JPG",
        "Emmanuelle Teste, musée Saint-Raymond, Licence Ouverte, Wikimedia Commons",
    ),
    # T3 room photo (Rose Valland JDP-0232): 9 of its 11 tune crops find their record first.
    Example(
        "jeu-de-paume",
        "Une salle du Jeu de Paume, vers 1942",
        "Salle d’exposition",
        "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/mnr/JDP-0232/jdp-04-p.jpg",
        "fonds Rose Valland, ministère de la Culture, POP",
        frame=True,
    ),
    # T2, PA00109757: right building first; below the building cut-off (0.86).
    Example(
        "porte",
        "Porte Saint-Julien, La Ferté-Bernard",
        "Monument",
        f"{_COMMONS}/4/40/La_Fert%C3%A9-Bernard_%2872%29_Porte_Saint-Julien_1.jpg/"
        "960px-La_Fert%C3%A9-Bernard_%2872%29_Porte_Saint-Julien_1.jpg",
        "GO69, CC0, Wikimedia Commons",
    ),
    # T2b, Joconde 02410000201: right record first; below the cut-off (0.525).
    Example(
        "jouy",
        "Vue de la vallée de Jouy-en-Josas",
        "Dessin",
        f"{_THUMB}/4/4f/La_vall%C3%A9e_de_Jouy-en-Josas%2C_Viollet-le-Duc%2C_1874%2C_"
        "mus%C3%A9e_de_Dreux%2C_Eure-et-Loir%2C_France.jpg/960px-La_vall%C3%A9e_de_"
        "Jouy-en-Josas%2C_Viollet-le-Duc%2C_1874%2C_mus%C3%A9e_de_Dreux%2C_Eure-et-Loir%2C_"
        "France.jpg",
        "Le Passant, CC BY-SA 4.0, Wikimedia Commons",
    ),
    # T2b, Joconde 05630029893: right record first, above the object cut-off.
    Example(
        "lampe",
        "Lampe à huile romaine",
        "Objet de musée",
        f"{_THUMB}/0/0f/MSR-26977-MV.jpg/960px-MSR-26977-MV.jpg",
        "Materia Viva, domaine public, Wikimedia Commons",
    ),
)
