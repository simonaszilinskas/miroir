from __future__ import annotations

from miroir.caption import Gazetteer

TWO_NAMED_ALIKE = Gazetteer.build(
    [
        ("Saint-Martin", "Gers", 43.6, 0.3),
        ("Saint-Martin", "Var", 43.5, 6.0),
        ("Beaune", "Côte-d'Or", 47.02, 4.84),
    ]
)


def test_a_bare_name_two_communes_bear_picks_neither() -> None:
    assert TWO_NAMED_ALIKE.find("ST-MARTIN. Eglise, portail") is None


def test_the_departement_settles_a_name_two_communes_bear() -> None:
    found = TWO_NAMED_ALIKE.find("SAINT-MARTIN (Var). Chapelle")
    assert found is not None
    assert found.departement == "Var"


def test_a_bare_name_one_commune_bears_still_counts() -> None:
    found = TWO_NAMED_ALIKE.find("BEAUNE. Hospices")
    assert found is not None
    assert found.commune == "Beaune"
