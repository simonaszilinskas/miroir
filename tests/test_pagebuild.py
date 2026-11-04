"""The site's built page: built from today's sources, or refused at deploy."""

from __future__ import annotations

from pathlib import Path

import pytest

from miroir.jobs import scw_run
from miroir.web.pagebuild import DIGEST_FILE, PageNotBuiltError, check_built, source_digest

#: What Vite's build writes for `sources` (the rule in web-ui/vite.config.ts, run with Node).
NODE_DIGEST = "baecc0587248e193bf56dd5e553bb3bf6cc7fed06d925ac9376087bf69da2e63"


def sources(root: Path) -> Path:
    (root / "src" / "a").mkdir(parents=True)
    (root / "public" / "dsfr").mkdir(parents=True)
    (root / "index.html").write_bytes("<p>é</p>".encode())
    (root / "package.json").write_bytes(b"{}")
    (root / "src" / "a" / "b.ts").write_bytes(b"x")
    (root / "src" / "Z.tsx").write_bytes(b"y")
    (root / "public" / "dsfr" / "dsfr.css").write_bytes(b"skip")
    (root / "public" / "favicon.svg").write_bytes(b"svg")
    return root


def built(folder: Path, digest: str) -> Path:
    folder.mkdir()
    (folder / "index.html").write_text("<p>Miroir</p>")
    (folder / DIGEST_FILE).write_text(digest + "\n")
    return folder


def test_the_digest_matches_the_one_vite_writes(tmp_path: Path) -> None:
    assert source_digest(sources(tmp_path)) == NODE_DIGEST


def test_the_digest_follows_the_sources_but_not_the_copied_dsfr(tmp_path: Path) -> None:
    root = sources(tmp_path)
    (root / "public" / "dsfr" / "dsfr.css").write_bytes(b"other")
    assert source_digest(root) == NODE_DIGEST
    (root / "src" / "Z.tsx").write_bytes(b"changed")
    assert source_digest(root) != NODE_DIGEST


def test_a_current_build_passes(tmp_path: Path) -> None:
    root = sources(tmp_path / "web-ui")
    check_built(root, built(tmp_path / "ui", NODE_DIGEST))


def test_a_missing_or_old_build_is_refused(tmp_path: Path) -> None:
    root = sources(tmp_path / "web-ui")
    with pytest.raises(PageNotBuiltError, match="not built"):
        check_built(root, tmp_path / "ui")
    old = built(tmp_path / "ui", "0" * 64)
    with pytest.raises(PageNotBuiltError, match="older"):
        check_built(root, old)


def test_the_demo_image_refuses_a_page_not_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scw_run, "WEB_UI_SOURCE", sources(tmp_path / "web-ui"))
    monkeypatch.setattr(scw_run, "WEB_UI_BUILD", tmp_path / "ui")
    with pytest.raises(PageNotBuiltError):
        scw_run.build_context(page=True)
    assert scw_run.build_context()  # job images carry no page check
