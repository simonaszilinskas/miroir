"""Query expansion, alpha-QE (experiments.md step E5).

Search once, add the top `k` gallery images to the query vector, each weighted by its
similarity to the power `alpha`, normalise, and search again (Radenović et al., 2018).
`alpha` 0 is a plain average of the query and its hits. Images hidden from a query are never
used to expand it, so expansion cannot leak an excluded answer.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from miroir.eval.scoring import Asked, Floats, Gallery, hide
from miroir.eval.vectors import VectorSet

#: Hits added to each query.
QE_TOP = 3

_CHUNK = 256


def expand_queries(
    query_vectors: VectorSet,
    gallery: Gallery,
    queries: Sequence[Asked],
    *,
    top: int = QE_TOP,
    alpha: float = 0.0,
) -> VectorSet:
    """New query vectors, one per query image, each the query plus its top hits."""
    image_ids = tuple(q.image_id for q in queries)
    if len(set(image_ids)) != len(image_ids):
        msg = "two queries share an image; expansion keys vectors by image"
        raise ValueError(msg)
    if not 0 < top <= len(gallery.image_ids) or alpha < 0:
        msg = f"top must be in 1..gallery size and alpha >= 0, got {top} and {alpha}"
        raise ValueError(msg)
    row_of = query_vectors.index()
    matrix: Floats = np.asarray(gallery.vectors, dtype=np.float32)
    out: Floats = np.empty((len(queries), matrix.shape[1]), dtype=np.float32)
    for start in range(0, len(queries), _CHUNK):
        chunk = queries[start : start + _CHUNK]
        asked: Floats = np.asarray(
            query_vectors.vectors[[row_of[q.image_id] for q in chunk]], dtype=np.float32
        )
        sims = hide(asked @ matrix.T, gallery, chunk)
        best = np.argpartition(-sims, top - 1, axis=1)[:, :top]
        weights = np.clip(np.take_along_axis(sims, best, axis=1), 0.0, None) ** alpha
        expanded = asked + np.einsum("qk,qkd->qd", weights, matrix[best])
        out[start : start + len(chunk)] = expanded / np.linalg.norm(expanded, axis=1, keepdims=True)
    return VectorSet(ids=image_ids, vectors=out.astype(np.float16), info=query_vectors.info)
