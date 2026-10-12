#!/usr/bin/env python3
"""Full export of every POP record that has at least one image, from public sources only.

Writes one folder per base under OUT (default ~/miroir-data/exports-full):
<base>/<base>.jsonl (one record per line, POP field codes) and <base>/metadata.json.

Sources, all public and needing no account:
- Open-data CSV dumps of the Ministère de la Culture, listed on data.gouv.fr and stored at
  https://ministere-culture.s3.sbg.io.cloud.ovh.net/POP/ (Joconde, Mérimée, Palissy, Mémoire,
  Autor, Muséofile).
- The public POP image bucket listing (https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/),
  read for object names only, to get image paths the CSVs leave out. No image is downloaded.
- Notice pages on https://pop.culture.gouv.fr/notice/<base>/<REF> (robots.txt allows all), which
  embed the raw POP record. Used for Rose-Valland (MNR) and Enluminures, which have no open-data
  dump, and for a few sample pages per base to map CSV column names back to POP field codes.
  api.pop.culture.gouv.fr is never called: its robots.txt disallows all.

Every step is resumable: rerun the same command after an interruption.

    python3 scripts/pop_full_export.py all              # every step, every base
    python3 scripts/pop_full_export.py build --bases joconde merimee
    python3 scripts/pop_full_export.py status

Steps: download, list-images, fieldmap, scrape, build. Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

UA = "miroir-export/0.1 (research; contact zilinskassimonas@gmail.com)"
CSV_HOST = "https://ministere-culture.s3.sbg.io.cloud.ovh.net/POP/"
ASSET_HOST = "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/"
NOTICE_HOST = "https://pop.culture.gouv.fr/notice/"
IIIF = "https://api.pop.culture.gouv.fr/notices/{db}/{ref}/iiif/manifest"
LICENCES = {
    "joconde.csv": "Licence Ouverte v2.0 (data.gouv.fr dataset 5b448216a3a729752eb11d6f)",
    "merimee.csv": "Licence Ouverte v2.0 (data.gouv.fr dataset 5af120e5b595087cfabcde81)",
    "merimee_acr.csv": "Licence Ouverte v2.0 (same bucket as merimee.csv)",
    "palissy.csv": "Licence Ouverte (data.gouv.fr dataset 615bcdd729be9185b3b0681a)",
    "memoire_palissy_merimee.csv": "ODbL (data.gouv.fr dataset 68e8bbfa16a41347e37c821c)",
    "memoire_mpp.csv": "Licence Ouverte v2.0 (data.gouv.fr dataset 68e8b8a4ca89b8efcd6bf154)",
    "autor.csv": "Licence Ouverte v2.0 (data.gouv.fr dataset 63a5287d98eba18a5da80dc8)",
    "museofile.csv": "Licence Ouverte (Ministère de la Culture open data bucket)",
}

# base -> CSV files. Bases missing here have no open-data dump and are scraped.
CSV_BASES: dict[str, list[str]] = {
    "joconde": ["joconde.csv"],
    "merimee": ["merimee.csv", "merimee_acr.csv"],
    "palissy": ["palissy.csv"],
    "memoire": ["memoire_palissy_merimee.csv", "memoire_mpp.csv"],
    "autor": ["autor.csv"],
    "museo": ["museofile.csv"],
}
SCRAPED_BASES = ["rose-valland", "enluminures"]
ALL_BASES = list(CSV_BASES) + SCRAPED_BASES
# POP database name -> folder in the image bucket.
BUCKET_PREFIX = {
    "joconde": "joconde",
    "merimee": "merimee",
    "palissy": "palissy",
    "museo": "museo",
    "rose-valland": "mnr",
    "enluminures": "enluminures",
    "memoire": "memoire",
}
# Prefixes listed only as one entry per REF (too many files to list one by one).
REF_ONLY_PREFIXES = {"memoire"}
BASE_LABEL = {
    "joconde": "Collections des musées de France (Joconde)",
    "merimee": "Patrimoine architectural (Mérimée)",
    "palissy": "Patrimoine mobilier (Palissy)",
    "memoire": "Photographies (Mémoire)",
    "autor": "Ressources biographiques (Autor)",
    "museo": "Répertoire des Musées de France (Muséofile)",
}
REF_COLUMNS = ("Reference", "Reference_de_la_notice", "reference_notice", "Identifiant")
PLACEHOLDER_RE = re.compile(r"(^|/)(sur_demande\.jpg|noimage|placeholder)", re.I)


def log(msg: str) -> None:
    sys.stderr.write(f"[{dt.datetime.now(dt.UTC):%H:%M:%S}] {msg}\n")
    sys.stderr.flush()


class Http:
    """Polite HTTP client: one request at a time, at most `rate` per second, retries on errors."""

    def __init__(self, rate: float) -> None:
        self.gap = 1.0 / rate
        self.last = 0.0

    def get(self, url: str, headers: dict[str, str] | None = None, tries: int = 5) -> bytes:
        for attempt in range(tries):
            wait = self.last + self.gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise
                if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == tries - 1:
                    raise
            time.sleep(min(60, 5 * 2**attempt))
        raise RuntimeError("unreachable")


# ---------------------------------------------------------------- download


def step_download(out: Path, bases: list[str], http: Http) -> None:
    raw = out / "_raw" / "csv"
    raw.mkdir(parents=True, exist_ok=True)
    listing = list_bucket_files(http)
    for base in bases:
        for name in CSV_BASES.get(base, []):
            remote = listing.get(name)
            if remote is None:
                log(f"download: {name} not in the open-data bucket, skipped")
                continue
            dest = raw / name
            meta_path = raw / f"{name}.meta.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            if meta.get("etag") != remote["etag"] and dest.exists():
                log(f"download: {name} changed upstream, starting over")
                dest.unlink()
            have = dest.stat().st_size if dest.exists() else 0
            if have == remote["size"]:
                log(f"download: {name} up to date ({have:,} bytes)")
            else:
                download_resumable(CSV_HOST + name, dest, have, remote["size"])
            meta_path.write_text(json.dumps({**remote, "url": CSV_HOST + name}, indent=2))


def list_bucket_files(http: Http) -> dict[str, dict[str, Any]]:
    body = http.get("https://ministere-culture.s3.sbg.io.cloud.ovh.net/?prefix=POP/")
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    files: dict[str, dict[str, Any]] = {}
    for c in ET.fromstring(body).findall("s:Contents", ns):
        key = c.findtext("s:Key", "", ns)
        files[key.removeprefix("POP/")] = {
            "size": int(c.findtext("s:Size", "0", ns)),
            "etag": c.findtext("s:ETag", "", ns).strip('"'),
            "lastModified": c.findtext("s:LastModified", "", ns),
        }
    return files


def download_resumable(url: str, dest: Path, have: int, size: int) -> None:
    log(f"download: {dest.name} from byte {have:,} of {size:,}")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={have}-"})
    with urllib.request.urlopen(req, timeout=300) as resp, dest.open("ab") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    got = dest.stat().st_size
    if got != size:
        raise RuntimeError(f"{dest.name}: got {got} bytes, expected {size}; rerun to resume")


# ---------------------------------------------------------------- image bucket listing


def step_list_images(out: Path, bases: list[str], http: Http) -> None:
    raw = out / "_raw" / "images"
    raw.mkdir(parents=True, exist_ok=True)
    wanted = {BUCKET_PREFIX[b] for b in bases if b in BUCKET_PREFIX}
    if any(b in bases for b in ("merimee", "palissy", "autor")):
        wanted.add("memoire")  # coverage figures only
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    for prefix in sorted(wanted):
        tsv = raw / f"{prefix}.tsv"
        state_path = raw / f"{prefix}.state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get("done"):
            log(f"list-images: {prefix} already listed ({state.get('count', 0):,} entries)")
            continue
        ref_only = prefix in REF_ONLY_PREFIXES
        token, count = state.get("token"), state.get("count", 0)
        if token is None and tsv.exists():
            tsv.unlink()
        log(f"list-images: {prefix}{' (REFs only)' if ref_only else ''} from entry {count:,}")
        while True:
            q = {"list-type": "2", "prefix": f"{prefix}/", "max-keys": "1000"}
            if ref_only:
                q["delimiter"] = "/"
            if token:
                q["continuation-token"] = token
            root = ET.fromstring(http.get(ASSET_HOST + "?" + urllib.parse.urlencode(q)))
            lines: list[str] = []
            if ref_only:
                for p in root.findall("s:CommonPrefixes", ns):
                    lines.append(p.findtext("s:Prefix", "", ns) + "\n")
            else:
                for c in root.findall("s:Contents", ns):
                    key = c.findtext("s:Key", "", ns)
                    lines.append(f"{key}\t{c.findtext('s:Size', '0', ns)}\n")
            with tsv.open("a", encoding="utf-8") as fh:
                fh.writelines(lines)
            count += len(lines)
            token = root.findtext("s:NextContinuationToken", None, ns)
            done = root.findtext("s:IsTruncated", "false", ns) != "true" or not token
            state = {"token": token, "count": count, "done": done}
            state_path.write_text(json.dumps(state))
            if count and count % 50000 < len(lines):
                log(f"list-images: {prefix} {count:,}")
            if done:
                log(f"list-images: {prefix} done, {count:,} entries")
                break


def read_bucket_images(out: Path, prefix: str) -> dict[str, list[str]]:
    """REF -> image keys, from the bucket listing, in name order."""
    tsv = out / "_raw" / "images" / f"{prefix}.tsv"
    by_ref: dict[str, list[str]] = defaultdict(list)
    if not tsv.exists():
        return by_ref
    with tsv.open(encoding="utf-8") as fh:
        for line in fh:
            key = line.split("\t", 1)[0].rstrip("\n")
            parts = key.split("/")
            if len(parts) >= 3 and parts[2] and not PLACEHOLDER_RE.search(key):
                by_ref[parts[1]].append(key)
    for keys in by_ref.values():
        keys.sort()
    return by_ref


# ---------------------------------------------------------------- notice pages


NEXT_CHUNK_RE = re.compile(r'self\.__next_f\.push\(\[1,"(.*?)"\]\)</script>', re.S)


def fetch_notice(http: Http, db: str, ref: str) -> dict[str, Any] | None:
    """Raw POP record embedded in the notice page, or None when the page has none."""
    url = NOTICE_HOST + db + "/" + urllib.parse.quote(ref, safe="")
    try:
        html = http.get(url).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    payload = "".join(json.loads('"' + c + '"') for c in NEXT_CHUNK_RE.findall(html))
    needle = '"REF":' + json.dumps(ref, ensure_ascii=False)
    dec = json.JSONDecoder()
    pos = payload.find(needle)
    while pos != -1:
        start = payload.rfind('{"_id":', 0, pos)
        if start != -1:
            try:
                obj, _ = dec.raw_decode(payload, start)
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and obj.get("REF") == ref:
                return obj
        pos = payload.find(needle, pos + 1)
    return None


def step_scrape(out: Path, bases: list[str], http: Http, limit: int | None) -> None:
    for base in bases:
        if base not in SCRAPED_BASES:
            continue
        refs = sorted(read_bucket_images(out, BUCKET_PREFIX[base]))
        if not refs:
            log(f"scrape: no image listing for {base}; run list-images first")
            continue
        raw = out / "_raw" / "pages"
        raw.mkdir(parents=True, exist_ok=True)
        dest, missing = raw / f"{base}.jsonl", raw / f"{base}.missing.txt"
        done = set()
        for path in (dest, missing):
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        done.add(json.loads(line)["REF"] if path == dest else line.rstrip("\n"))
        todo = [r for r in refs if r not in done][:limit]
        log(f"scrape: {base} {len(done):,} done, {len(todo):,} to fetch")
        with dest.open("a", encoding="utf-8") as fh, missing.open("a", encoding="utf-8") as mh:
            for i, ref in enumerate(todo, 1):
                rec = fetch_notice(http, base, ref)
                if rec is None:
                    mh.write(ref + "\n")
                    mh.flush()
                else:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                if i % 200 == 0:
                    log(f"scrape: {base} {i:,}/{len(todo):,}")


# ---------------------------------------------------------------- CSV reading and field map


def open_csv(path: Path) -> tuple[Iterator[list[str]], list[str]]:
    head = path.open("rb").read(1 << 20)
    try:
        head.decode("utf-8")
        enc = "utf-8"
    except UnicodeDecodeError as e:
        enc = "utf-8" if e.start > len(head) - 4 else "cp1252"
    first = head.split(b"\n", 1)[0]
    delim = "|" if first.count(b"|") > first.count(b";") else ";"
    csv.field_size_limit(1 << 30)
    fh = path.open(encoding=enc, errors="replace", newline="")
    reader = csv.reader(fh, delimiter=delim)
    header = [h.strip().lstrip("﻿") for h in next(reader)]
    return reader, header


def iter_csv(path: Path) -> Iterator[dict[str, str]]:
    reader, header = open_csv(path)
    for row in reader:
        if len(row) == len(header):
            yield dict(zip(header, row, strict=True))


def ref_of(row: dict[str, str]) -> str:
    for col in REF_COLUMNS:
        if row.get(col):
            return row[col].strip()
    return ""


def norm_value(v: Any) -> str | None:
    if isinstance(v, list):
        if not v or not all(isinstance(x, str) for x in v):
            return None
        return ";".join(x.strip() for x in v)
    if isinstance(v, str):
        return v.strip()
    return None


def squash(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def step_fieldmap(out: Path, bases: list[str], http: Http, samples: int) -> None:
    """Map CSV column names to POP field codes by matching values against sample notice pages."""
    fm_dir = out / "_raw" / "fieldmap"
    fm_dir.mkdir(parents=True, exist_ok=True)
    for base in bases:
        for name in CSV_BASES.get(base, []):
            path = out / "_raw" / "csv" / name
            dest = fm_dir / f"{name}.json"
            if dest.exists() or not path.exists():
                continue
            rows = pick_samples(path, samples)
            votes: dict[str, Counter[str]] = defaultdict(Counter)
            is_list: dict[str, bool] = {}
            used = 0
            for row in rows:
                rec = fetch_notice(http, base, ref_of(row))
                if rec is None:
                    continue
                used += 1
                values: dict[str, list[str]] = defaultdict(list)
                for code, v in rec.items():
                    nv = norm_value(v)
                    if nv:
                        values[nv].append(code)
                        is_list.setdefault(code, isinstance(v, list))
                for col, v in row.items():
                    for code in values.get(v.strip(), []) if v.strip() else []:
                        votes[col][code] += 1
            mapping: dict[str, dict[str, Any]] = {}
            taken: set[str] = set()
            ranked = sorted(votes.items(), key=lambda kv: -kv[1].most_common(1)[0][1])
            for col, counter in ranked:
                for code, n in counter.most_common():
                    if code in taken or code in ("_id", "id"):
                        continue
                    ambiguous = len(counter) > 1 and n == 1 and squash(col)[:4] != squash(code)[:4]
                    if n >= 2 or not ambiguous:
                        mapping[col] = {"code": code, "list": is_list.get(code, False), "votes": n}
                        taken.add(code)
                    break
            ref_col = next((c for c in REF_COLUMNS if c in rows[0]), None) if rows else None
            if ref_col:
                mapping[ref_col] = {"code": "REF", "list": False, "votes": used}
            dest.write_text(
                json.dumps({"file": name, "samples": used, "map": mapping}, indent=2, ensure_ascii=False)
            )
            log(f"fieldmap: {name} {len(mapping)} columns mapped from {used} notice pages")


def pick_samples(path: Path, n: int) -> list[dict[str, str]]:
    """The fullest rows among the first 20,000, spread across the file start."""
    best: list[tuple[int, dict[str, str]]] = []
    for i, row in enumerate(iter_csv(path)):
        if i >= 20000:
            break
        if ref_of(row):
            best.append((sum(1 for v in row.values() if v.strip()), row))
    best.sort(key=lambda t: -t[0])
    return [r for _, r in best[:n]]


def load_fieldmap(out: Path, name: str) -> dict[str, dict[str, Any]]:
    path = out / "_raw" / "fieldmap" / f"{name}.json"
    return json.loads(path.read_text())["map"] if path.exists() else {}


def to_pop(row: dict[str, str], fmap: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for col, v in row.items():
        m = fmap.get(col)
        if m is None:
            rec[col] = v
        elif m["list"]:
            rec[m["code"]] = [x.strip() for x in v.split(";") if x.strip()]
        else:
            rec[m["code"]] = v
    return rec


# ---------------------------------------------------------------- build


def asset_url(key: str) -> str:
    return key if key.startswith("http") else ASSET_HOST + key.lstrip("/")


def split_refs(v: str) -> list[str]:
    return [x.strip() for x in re.split(r"[;,\s]+", v or "") if x.strip()]


class MemoireIndex:
    """Mémoire images by Mémoire REF, and by the Mérimée/Palissy REF they illustrate."""

    def __init__(self, out: Path) -> None:
        self.by_ref: dict[str, str] = {}
        self.by_target: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for name in CSV_BASES["memoire"]:
            path = out / "_raw" / "csv" / name
            if not path.exists():
                continue
            for row in iter_csv(path):
                key = memoire_image(row)
                if not key:
                    continue
                ref = ref_of(row)
                self.by_ref[ref] = key
                rank_s = row.get("Rang_d_affichage_de_l_image_dans_les_bases_Palissy_et_Merimee", "")
                rank = int(rank_s) if rank_s.strip().isdigit() else 10**6
                for target in split_refs(row.get("References_Palissy_Merimee_lien_notice_en_cours", "")):
                    self.by_target[target].append((rank, key))
        for lst in self.by_target.values():
            lst.sort()
        log(f"memoire index: {len(self.by_ref):,} images, {len(self.by_target):,} linked notices")


def memoire_image(row: dict[str, str]) -> str:
    v = row.get("Lien_vers_l_image", "").strip()
    first = v.split(";")[0].strip()
    return "" if not first or PLACEHOLDER_RE.search(first) else first


def step_build(out: Path, bases: list[str], force: bool) -> None:
    mem: MemoireIndex | None = None
    for base in bases:
        meta_path = out / base / "metadata.json"
        if meta_path.exists() and not force and not sources_changed(out, base, meta_path):
            log(f"build: {base} up to date ({json.loads(meta_path.read_text())['exportedCount']:,})")
            continue
        if base in ("merimee", "palissy", "autor") and mem is None:
            mem = MemoireIndex(out)
        if base in CSV_BASES:
            build_csv_base(out, base, mem)
        else:
            build_scraped_base(out, base)


def sources_changed(out: Path, base: str, meta_path: Path) -> bool:
    old = json.loads(meta_path.read_text()).get("sourceEtags", {})
    for name in CSV_BASES.get(base, []):
        p = out / "_raw" / "csv" / f"{name}.meta.json"
        if p.exists() and json.loads(p.read_text())["etag"] != old.get(name):
            return True
    return False


def images_for(base: str, ref: str, row: dict[str, str], bucket: dict[str, list[str]],
               mem: MemoireIndex | None) -> list[str]:
    keys = list(bucket.get(ref, []))
    if base == "memoire":
        k = memoire_image(row)
        keys = [k] if k else keys
    elif base in ("merimee", "palissy") and mem is not None:
        keys += [k for _, k in mem.by_target.get(ref, []) if k not in keys]
    elif base == "autor" and mem is not None:
        for mref in split_refs(row.get("liste_images_memoire_liees", "")):
            k = mem.by_ref.get(mref)
            if k and k not in keys:
                keys.append(k)
    return keys


def finish_record(rec: dict[str, Any], base: str, keys: list[str]) -> dict[str, Any]:
    rec.setdefault("BASE", BASE_LABEL.get(base, base))
    rec["base"] = base
    rec["IMG"] = keys
    rec["imageUrl"] = asset_url(keys[0])
    rec["imageUrls"] = [asset_url(k) for k in keys]
    rec["iiif_manifest"] = IIIF.format(db=base, ref=urllib.parse.quote(rec["REF"], safe=""))
    rec["sourceCollection"] = base
    return rec


def build_csv_base(out: Path, base: str, mem: MemoireIndex | None) -> None:
    bucket = read_bucket_images(out, BUCKET_PREFIX[base]) if base in BUCKET_PREFIX else {}
    if base == "memoire":
        bucket = {}
    folder = out / base
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{base}.jsonl.tmp"
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    sources: dict[str, Any] = {}
    with tmp.open("w", encoding="utf-8") as fh:
        for name in CSV_BASES[base]:
            path = out / "_raw" / "csv" / name
            if not path.exists():
                log(f"build: {name} missing; run download first")
                continue
            fmap = load_fieldmap(out, name)
            meta_p = out / "_raw" / "csv" / f"{name}.meta.json"
            src_meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            sources[name] = {**src_meta, "licence": LICENCES.get(name), "fieldsMapped": len(fmap)}
            log(f"build: {base} reading {name}")
            for row in iter_csv(path):
                ref = ref_of(row)
                if not ref or ref in seen:
                    counts["skippedNoRefOrDuplicate"] += 1
                    continue
                seen.add(ref)
                counts["sourceRecords"] += 1
                flagged = row.get("Presence_image", "").strip().lower() == "oui" or (
                    row.get("notice_illustree", "").strip().lower() == "oui"
                )
                counts["flaggedWithImage"] += flagged
                keys = images_for(base, ref, row, bucket, mem)
                if not keys:
                    counts["flaggedButNoImageUrl"] += flagged
                    continue
                rec = finish_record(to_pop(row, fmap), base, keys)
                rec["REF"] = ref
                rec["sourceFile"] = name
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["exportedCount"] += 1
                counts["imageCount"] += len(keys)
    counts["bucketRefsNotInCsv"] = len(set(bucket) - seen)
    write_meta(out, base, tmp, counts, sources, filter_desc(base))


def build_scraped_base(out: Path, base: str) -> None:
    src = out / "_raw" / "pages" / f"{base}.jsonl"
    if not src.exists():
        log(f"build: {base} has no scraped pages; run scrape first")
        return
    bucket = read_bucket_images(out, BUCKET_PREFIX[base])
    folder = out / base
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{base}.jsonl.tmp"
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    with src.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fh:
        for line in fin:
            rec = json.loads(line)
            ref = rec["REF"]
            if ref in seen:
                continue
            seen.add(ref)
            counts["sourceRecords"] += 1
            keys = [k for k in (rec.get("IMG") or []) if isinstance(k, str) and k]
            if not keys:  # Rose-Valland keeps its images in VIDEO
                video = rec.get("VIDEO") or []
                keys = [k for k in video if isinstance(k, str) and "/" in k and " " not in k[:4]]
            keys = keys or bucket.get(ref, [])
            keys = [k for k in dict.fromkeys(keys) if not PLACEHOLDER_RE.search(k)]
            if not keys:
                continue
            rec = finish_record(rec, base, keys)
            rec["sourceFile"] = "pop.culture.gouv.fr notice pages"
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["exportedCount"] += 1
            counts["imageCount"] += len(keys)
    miss = out / "_raw" / "pages" / f"{base}.missing.txt"
    counts["bucketRefsWithoutNoticePage"] = (
        sum(1 for _ in miss.open()) if miss.exists() else 0
    )
    counts["bucketRefs"] = len(bucket)
    sources = {"pages": {"url": NOTICE_HOST + base + "/<REF>", "refsFrom": ASSET_HOST + BUCKET_PREFIX[base] + "/"}}
    write_meta(out, base, tmp, counts, sources, filter_desc(base))


def filter_desc(base: str) -> str:
    return {
        "joconde": "CSV row whose REF has at least one file under joconde/<REF>/ in the image bucket",
        "merimee": "CSV row with files under merimee/<REF>/ or a Mémoire photo linked to the REF",
        "palissy": "CSV row with files under palissy/<REF>/ or a Mémoire photo linked to the REF",
        "memoire": "CSV row with a non-empty Lien_vers_l_image",
        "autor": "CSV row whose liste_images_memoire_liees points to a Mémoire record with an image",
        "museo": "CSV row whose REF has at least one file under museo/<REF>/ in the image bucket",
    }.get(base, "REF with files in the image bucket, record read from its notice page, IMG or VIDEO non-empty")


def write_meta(out: Path, base: str, tmp: Path, counts: Counter[str], sources: dict[str, Any],
               filt: str) -> None:
    final = out / base / f"{base}.jsonl"
    tmp.replace(final)
    meta = {
        "collection": base,
        "source": sources,
        "filter": filt,
        "generatedFields": ["IMG", "imageUrl", "imageUrls", "iiif_manifest", "sourceCollection", "sourceFile"],
        "fieldNames": "POP field codes where the CSV column could be matched to one (see _raw/fieldmap); "
        "other CSV columns keep their open-data name",
        **dict(counts),
        "sizeBytes": final.stat().st_size,
        "sourceEtags": {k: v.get("etag") for k, v in sources.items() if isinstance(v, dict)},
        "generatedAt": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "script": "miroir/scripts/pop_full_export.py",
    }
    (out / base / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    log(f"build: {base} wrote {counts['exportedCount']:,} records ({meta['sizeBytes']:,} bytes)")


def step_status(out: Path) -> None:
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    log(f"{out}: {total / 1e9:.2f} GB on disk")
    for base in ALL_BASES:
        m = out / base / "metadata.json"
        if m.exists():
            d = json.loads(m.read_text())
            log(f"  {base:13} {d.get('exportedCount', 0):>9,} records  {d.get('imageCount', 0):>9,} images")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", choices=["download", "list-images", "fieldmap", "scrape", "build", "status", "all"])
    ap.add_argument("--out", type=Path, default=Path.home() / "miroir-data" / "exports-full")
    ap.add_argument("--bases", nargs="+", default=ALL_BASES, choices=ALL_BASES)
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second (bucket, CSV)")
    ap.add_argument("--page-rate", type=float, default=1.0, help="notice pages per second")
    ap.add_argument("--samples", type=int, default=12, help="notice pages per CSV for the field map")
    ap.add_argument("--scrape-limit", type=int, default=None)
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--force", action="store_true", help="rebuild even when sources are unchanged")
    a = ap.parse_args()
    steps = ["download", "list-images", "fieldmap", "scrape", "build"] if "all" in a.steps else a.steps
    a.out.mkdir(parents=True, exist_ok=True)
    http, pages = Http(a.rate), Http(a.page_rate)
    for step in steps:
        used = sum(p.stat().st_size for p in a.out.rglob("*") if p.is_file()) / 1e9
        if used > a.max_gb:
            sys.exit(f"stop: {a.out} holds {used:.1f} GB, over --max-gb {a.max_gb}")
        if step == "download":
            step_download(a.out, a.bases, http)
        elif step == "list-images":
            step_list_images(a.out, a.bases, http)
        elif step == "fieldmap":
            step_fieldmap(a.out, a.bases, pages, a.samples)
        elif step == "scrape":
            step_scrape(a.out, a.bases, pages, a.scrape_limit)
        elif step == "build":
            step_build(a.out, a.bases, a.force)
        elif step == "status":
            step_status(a.out)


if __name__ == "__main__":
    main()
