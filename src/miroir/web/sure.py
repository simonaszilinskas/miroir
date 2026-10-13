"""When an object search counts as sure: the margin rule (`sure.json` in the index bundle).

The cosine "confidence" alone is a poor guide for objects: many records share an image (two
pieces of a pair photographed together, a crop shared by neighbouring works), so the top
answer can score 1.0 and still be the wrong record. What tells a right answer is how far it
stands above the others. An object search is sure when its top entity's confidence beats
every other entity of the top `over` by at least `margin`. Buildings keep their bucket
cut-off (`routing.json`).

`miroir web sure` sets the margin on the tune halves (docs/results.md, "Sure answers").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ConfigDict, TypeAdapter

from miroir.web.routing import PhotoType

SURE_FILE = "sure.json"


@dataclass(frozen=True, slots=True)
class SureRule:
    __pydantic_config__ = ConfigDict(extra="forbid")

    margin: float
    """Least gap between the top entity's confidence and the best of the others."""
    over: int = 30
    """How many top entities the gap is measured over."""
    photo_type: PhotoType = PhotoType.OBJECT
    note: str = ""
    """Where the margin comes from, and what it gives on the tune halves."""


def margin_of(confidences: Sequence[float]) -> float:
    """Top confidence minus the best of the others; the whole confidence when alone."""
    if not confidences:
        msg = "no confidences"
        raise ValueError(msg)
    rest = confidences[1:]
    return confidences[0] - (max(rest) if rest else 0.0)


_SURE = TypeAdapter(SureRule)


def read_sure(path: Path) -> SureRule | None:
    return _SURE.validate_json(path.read_bytes()) if path.exists() else None


def write_sure(path: Path, rule: SureRule) -> None:
    path.write_bytes(_SURE.dump_json(rule, indent=2) + b"\n")
