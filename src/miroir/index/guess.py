"""The G4 building-or-object guess, trained on the full sample and saved for the website.

Same labels as R2 (`miroir.eval.router.photo_type`): Joconde, Rose Valland and Mémoire photos
of a Palissy object (PM) are objects, Mémoire photos of a Mérimée building (PA) are buildings,
and every building or object of a tune query is left out. Only the training images change:
every image of the run instead of the screening gallery.

The website applies it to a query's EmbeddingGemma 2 (A7) vector `x` (unit length):

    p = sigmoid(((x - mean) / scale) @ coef + bias)
    weight of A7 = p * w_building + (1 - p) * w_object;  MegaLoc (A3) gets 1 - that

then sums each model's min-max scaled cosines with those weights (G1). R2 picked the weights
below on T1, T3 and T5 tune (results.md, G4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from miroir.config import Settings
from miroir.eval.router import GuessReport, guess_report, photo_types
from miroir.eval.vectors import read_vectors
from miroir.records.store import read_images

logger = logging.getLogger(__name__)

#: G4 blend (R2): A7 weight for a sure building and for a sure object.
W_BUILDING = 1.0
W_OBJECT = 0.2
#: G4 hard switch (R2): A7 weight when p > 0.5, and otherwise.
SWITCH_BUILDING = 0.9
SWITCH_OBJECT = 0.2


@dataclass(frozen=True, slots=True)
class SavedGuess:
    report: GuessReport
    path: Path


def train_and_save(settings: Settings, run: str, out: Path) -> SavedGuess:
    """Fit the guess on every labelled image of `run` and write it to `out` (.npz)."""
    vectors = read_vectors(settings.vectors_path(run))
    entity_of = {i.image_id: i.entity_id for i in read_images(settings.images_parquet)}
    report = guess_report(settings, [vectors], vectors.ids, entity_of, photo_types(settings))
    guess = report.guess
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        mean=guess.mean.astype(np.float32),
        scale=guess.scale.astype(np.float32),
        coef=guess.coef.astype(np.float32),
        bias=np.float32(guess.bias),
        penalty=np.float32(guess.penalty),
        w_building=np.float32(W_BUILDING),
        w_object=np.float32(W_OBJECT),
        switch_building=np.float32(SWITCH_BUILDING),
        switch_object=np.float32(SWITCH_OBJECT),
        trained_on=np.int64(report.trained_on),
        held_out_accuracy=np.float32(report.held_out_accuracy),
        vectors_run=np.array(run),
    )
    logger.info(
        "guess on %s: %d images, held-out balanced accuracy %.3f, wrote %s",
        run,
        report.trained_on,
        report.held_out_accuracy,
        out,
    )
    return SavedGuess(report, out)
