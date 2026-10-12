from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from miroir.eval.vectors import Route, RunInfo, read_vectors, write_vectors
from miroir.index.pca import fit_pca, load_pca, project, save_pca, write_projected

INFO = RunInfo(
    option="A3",
    model="gmberton/MegaLoc",
    resolution=322,
    pooling="salad",
    gallery="g",
    created=date(2026, 10, 12),
    cost_usd=0.0,
    wall_time_s=0.0,
    route=Route.HOSTED,
)


def cloud(rows: int = 300, dim: int = 40, rank: int = 4) -> np.ndarray:
    """Unit vectors that live near a `rank`-dimensional subspace."""
    rng = np.random.default_rng(0)
    low = rng.normal(size=(rows, rank)) @ rng.normal(size=(rank, dim))
    x = low + 0.01 * rng.normal(size=(rows, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_pca_keeps_the_subspace_and_the_neighbours(tmp_path: Path) -> None:
    x = cloud()
    run = write_vectors(tmp_path / "run", [f"i{n}" for n in range(len(x))], x, INFO)
    pca = fit_pca(run, 4, "run")
    assert pca.explained.sum() > 0.99
    reduced = project(pca, x)
    assert reduced.shape == (len(x), 4)
    assert np.allclose(np.linalg.norm(reduced, axis=1), 1.0, atol=1e-5)
    centred = x - x.mean(axis=0)
    centred /= np.linalg.norm(centred, axis=1, keepdims=True)
    full = centred @ centred.T
    small = reduced @ reduced.T
    # Each row's nearest neighbour survives the projection.
    np.fill_diagonal(full, -np.inf)
    np.fill_diagonal(small, -np.inf)
    assert (full.argmax(axis=1) == small.argmax(axis=1)).mean() > 0.9


def test_saved_pca_projects_the_same_and_runs_note_it(tmp_path: Path) -> None:
    x = cloud()
    ids = [f"i{n}" for n in range(len(x))]
    run = write_vectors(tmp_path / "run", ids, x, INFO)
    pca = fit_pca(run, 8, "run")
    save_pca(tmp_path / "pca.npz", pca)
    loaded = load_pca(tmp_path / "pca.npz")
    assert loaded.fitted_on == "run"
    assert np.allclose(project(loaded, x), project(pca, x), atol=1e-5)

    write_projected(pca, run, tmp_path / "small")
    small = read_vectors(tmp_path / "small")
    assert small.ids == tuple(ids)
    assert small.dim == 8
    assert "PCA 40 -> 8 dims fitted on run" in small.info.notes
