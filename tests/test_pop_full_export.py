"""scripts/pop_full_export.py, with a fake network (no socket is opened)."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from miroir.config import REPO_ROOT


def _load() -> ModuleType:
    path = REPO_ROOT / "scripts" / "pop_full_export.py"
    spec = importlib.util.spec_from_file_location("pop_full_export", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pfe = _load()
BODY = b"Reference|Lien_vers_l_image\n" + b"x" * 1000
NAME = "memoire_mpp.csv"


class Response:
    def __init__(self, body: bytes, status: int = 200, headers: dict[str, str] | None = None):
        self.body, self.status, self.headers = body, status, headers or {}

    def read(self, n: int = -1) -> bytes:
        n = len(self.body) if n < 0 else n
        out, self.body = self.body[:n], self.body[n:]
        return out

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


Handler = Callable[[urllib.request.Request], Response]


class Opener:
    """Answers each request with `handler`, and keeps the requests."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: int) -> Response:
        assert timeout > 0
        self.requests.append(req)
        return self.handler(req)


def _http(handler: Handler, tries: int = 3) -> tuple[Any, Opener]:
    opener = Opener(handler)
    return pfe.Http(1000.0, opener=opener, sleep=lambda _s: None, tries=tries), opener


def _listing(body: bytes, etag: str = "e1") -> bytes:
    return (
        '<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Contents>'
        f"<Key>POP/{NAME}</Key><Size>{len(body)}</Size><ETag>&quot;{etag}&quot;</ETag>"
        "</Contents></ListBucketResult>"
    ).encode()


def _server(body: bytes, *, ranges: bool = True, cut: int | None = None) -> Handler:
    """The open-data bucket: its listing, and the CSV (honouring Range when `ranges`, and
    cutting the connection after `cut` bytes when set)."""

    def handle(req: urllib.request.Request) -> Response:
        url = req.full_url
        if "?prefix=" in url:
            return Response(_listing(body))
        start = 0
        header = req.get_header("Range") or ""
        if ranges and header:
            start = int(header.removeprefix("bytes=").rstrip("-"))
        part = body[start:] if cut is None else body[start:cut]
        if ranges and header:
            headers = {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"}
            return Response(part, 206, headers)
        return Response(part)

    return handle


def _csv(out: Path) -> Path:
    return out / "_raw" / "csv" / NAME


def _meta(out: Path) -> dict[str, Any]:
    return json.loads((out / "_raw" / "csv" / f"{NAME}.meta.json").read_text())


# --- download ---------------------------------------------------------------------------


def test_an_interrupted_first_download_resumes_instead_of_starting_over(tmp_path: Path) -> None:
    http, _ = _http(_server(BODY, cut=400), tries=1)
    with pytest.raises(OSError, match="expected"):
        pfe.step_download(tmp_path, ["memoire"], http)
    assert _csv(tmp_path).stat().st_size == 400
    assert _meta(tmp_path)["complete"] is False

    http, opener = _http(_server(BODY))
    pfe.step_download(tmp_path, ["memoire"], http)
    assert _csv(tmp_path).read_bytes() == BODY
    assert _meta(tmp_path)["complete"] is True
    assert opener.requests[-1].get_header("Range") == "bytes=400-"


def test_a_cut_connection_is_retried_from_the_bytes_kept(tmp_path: Path) -> None:
    calls: list[int] = []

    def flaky(req: urllib.request.Request) -> Response:
        if "?prefix=" not in req.full_url:
            calls.append(1)
        return _server(BODY, cut=300 if len(calls) == 1 else None)(req)

    http, opener = _http(flaky)
    pfe.step_download(tmp_path, ["memoire"], http)
    assert _csv(tmp_path).read_bytes() == BODY
    assert [r.get_header("Range") for r in opener.requests[1:]] == ["bytes=0-", "bytes=300-"]


def test_a_server_ignoring_range_replaces_the_partial_file(tmp_path: Path) -> None:
    http, _ = _http(_server(BODY, cut=250), tries=1)
    with pytest.raises(OSError, match="expected"):
        pfe.step_download(tmp_path, ["memoire"], http)

    http, _ = _http(_server(BODY, ranges=False))
    pfe.step_download(tmp_path, ["memoire"], http)
    assert _csv(tmp_path).read_bytes() == BODY  # not 250 bytes, then the whole file again


def test_a_206_from_the_wrong_byte_is_refused(tmp_path: Path) -> None:
    _csv(tmp_path).parent.mkdir(parents=True)
    _csv(tmp_path).write_bytes(BODY[:100])

    def wrong(req: urllib.request.Request) -> Response:
        if "?prefix=" in req.full_url:
            return Response(_listing(BODY))
        return Response(BODY, 206, {"Content-Range": f"bytes 0-{len(BODY) - 1}/{len(BODY)}"})

    http, _ = _http(wrong, tries=1)
    (tmp_path / "_raw" / "csv" / f"{NAME}.meta.json").write_text(json.dumps({"etag": "e1"}))
    with pytest.raises(RuntimeError, match="asked for byte 100"):
        pfe.step_download(tmp_path, ["memoire"], http)


def test_http_get_retries_server_errors_but_not_404() -> None:
    answers = [500, 200]

    def handle(req: urllib.request.Request) -> Response:
        code = answers.pop(0)
        if code != 200:
            raise urllib.error.HTTPError(req.full_url, code, "error", {}, None)  # type: ignore[arg-type]
        return Response(b"ok")

    http, _ = _http(handle)
    assert http.get("https://example.org/a") == b"ok"

    def missing(req: urllib.request.Request) -> Response:
        raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)  # type: ignore[arg-type]

    http, opener = _http(missing)
    with pytest.raises(urllib.error.HTTPError):
        http.get("https://example.org/b")
    assert len(opener.requests) == 1


# --- scraped pages --------------------------------------------------------------------------


def _bucket(out: Path, prefix: str, refs: list[str]) -> None:
    raw = out / "_raw" / "images"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{prefix}.tsv").write_text("".join(f"{prefix}/{r}/{r}.jpg\t10\n" for r in refs))
    (raw / f"{prefix}.state.json").write_text(json.dumps({"done": True, "count": len(refs)}))


def _pages(out: Path, base: str) -> Path:
    path = out / "_raw" / "pages" / f"{base}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def test_a_torn_last_page_line_is_dropped_and_scraping_goes_on(tmp_path: Path) -> None:
    _bucket(tmp_path, "mnr", ["MNR1", "MNR2"])
    pages = _pages(tmp_path, "rose-valland")
    good = json.dumps({"REF": "MNR1", "VIDEO": ["mnr/MNR1/MNR1.jpg"]}) + "\n"
    pages.write_text(good + '{"REF": "MNR2", "VID')

    http, opener = _http(lambda _req: Response(b"<html></html>"))
    pfe.step_scrape(tmp_path, ["rose-valland"], http, limit=None)
    assert pages.read_text().startswith(good)
    assert len(opener.requests) == 1  # MNR2 again; MNR1 was done
    pfe.build_scraped_base(tmp_path, "rose-valland")
    built = (tmp_path / "rose-valland" / "rose-valland.jsonl").read_text().splitlines()
    assert [json.loads(line)["REF"] for line in built] == ["MNR1"]


def test_more_scraped_pages_make_the_base_rebuild(tmp_path: Path) -> None:
    _bucket(tmp_path, "mnr", ["MNR1", "MNR2"])
    pages = _pages(tmp_path, "rose-valland")
    pages.write_text(json.dumps({"REF": "MNR1", "VIDEO": ["mnr/MNR1/a.jpg"]}) + "\n")
    pfe.step_build(tmp_path, ["rose-valland"], force=False)
    meta = tmp_path / "rose-valland" / "metadata.json"
    assert not pfe.sources_changed(tmp_path, "rose-valland", meta)
    with pages.open("a") as fh:
        fh.write(json.dumps({"REF": "MNR2", "VIDEO": ["mnr/MNR2/b.jpg"]}) + "\n")
    assert pfe.sources_changed(tmp_path, "rose-valland", meta)
    pfe.step_build(tmp_path, ["rose-valland"], force=False)
    assert json.loads(meta.read_text())["exportedCount"] == 2


def test_a_listing_page_appended_after_the_last_state_is_dropped(tmp_path: Path) -> None:
    tsv = tmp_path / "x.tsv"
    tsv.write_text("a\n" + "b\n")
    pfe.resume_listing(tsv, {"token": "t", "bytes": 2})
    assert tsv.read_text() == "a\n"


# --- field map ------------------------------------------------------------------------------

MEMOIRE_CSV = "Reference|Lien_vers_l_image|Diffusion\nAP1|memoire/AP1/a.jpg|\n"


def _memoire_csv(out: Path) -> None:
    _csv(out).parent.mkdir(parents=True, exist_ok=True)
    for name in pfe.CSV_BASES["memoire"]:
        (out / "_raw" / "csv" / name).write_text(MEMOIRE_CSV)


def _no_pages(req: urllib.request.Request) -> Response:
    raise urllib.error.HTTPError(req.full_url, 404, "missing", {}, None)  # type: ignore[arg-type]


def test_rights_column_falls_back_to_the_field_docs_label(tmp_path: Path) -> None:
    out, docs = tmp_path / "out", tmp_path / "exports"
    _memoire_csv(out)
    (docs / "memoire").mkdir(parents=True)
    fields = [{"field": "DIFF", "label": "Diffusion", "tsType": "string"}]
    (docs / "memoire" / "field-docs.json").write_text(json.dumps({"fields": fields}))

    http, _ = _http(_no_pages)
    pfe.step_fieldmap(out, ["memoire"], http, samples=5, docs_dir=docs)
    mapping = pfe.load_fieldmap(out, NAME)
    assert mapping["Diffusion"]["code"] == "DIFF"
    assert mapping["Reference"]["code"] == "REF"


def test_an_unmapped_rights_column_stops_the_field_map(tmp_path: Path) -> None:
    _memoire_csv(tmp_path)
    http, _ = _http(_no_pages)
    with pytest.raises(SystemExit, match="DIFF"):
        pfe.step_fieldmap(tmp_path, ["memoire"], http, samples=5, docs_dir=tmp_path / "none")


def test_build_refuses_a_field_map_without_rights(tmp_path: Path) -> None:
    _memoire_csv(tmp_path)
    fm = tmp_path / "_raw" / "fieldmap"
    fm.mkdir(parents=True)
    for name in pfe.CSV_BASES["memoire"]:
        ref = {"Reference": {"code": "REF", "list": False}}
        (fm / f"{name}.json").write_text(json.dumps({"map": ref}))
    with pytest.raises(SystemExit, match="DIFF"):
        pfe.step_build(tmp_path, ["memoire"], force=True)


# --- data dir -----------------------------------------------------------------------------


def test_output_follows_miroir_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    assert pfe.default_out(env) == Path("~/miroir-data/exports-full").expanduser()
    env.write_text(f"MIROIR_DATA_DIR={tmp_path / 'from-file'}\nMIROIR_EXPORTS_DIR='/july'\n")
    assert pfe.default_out(env) == tmp_path / "from-file" / "exports-full"
    assert pfe.default_field_docs(env) == Path("/july")
    monkeypatch.setenv("MIROIR_DATA_DIR", str(tmp_path / "from-env"))
    assert pfe.default_out(env) == tmp_path / "from-env" / "exports-full"
