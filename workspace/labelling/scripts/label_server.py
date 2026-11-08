#!/usr/bin/env python
"""Local labelling server: serves labelling/index.html + tiles and a small JSON API.

    python scripts/label_server.py [--port 8765]   ->  http://127.0.0.1:8765/

API (see .plans/lithology-labelling-tool-spec.md, "Contracts"):
  GET /api/pages | /api/vocab | /api/page/<stem>      PUT /api/label/<stem> {"label", "version"}
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shutil
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelslib as L  # noqa: E402
from labelling_setup import require_dependencies

PAGE_DIR = L.ROOT / "labelling"
CACHE_DIR = L.ROOT / "labelling_cache"
HISTORY_DIR = L.LABELS_DIR / ".history"
HISTORY_KEEP = 20
HISTORY_MIN_INTERVAL_S = 120

_lock = threading.Lock()
_last_snapshot: dict[str, float] = {}


def version_of(path: Path) -> str:
    try:
        return str(path.stat().st_mtime_ns)
    except FileNotFoundError:
        return "0"


def read_label(stem: str) -> dict | None:
    p = L.label_path(stem)
    return L.sync_legend_labels(json.loads(p.read_text())) if p.exists() else None


def snapshot(stem: str) -> None:
    """Copy the current file to history, at most once per HISTORY_MIN_INTERVAL_S per stem."""
    src = L.label_path(stem)
    now = time.time()
    if not src.exists() or now - _last_snapshot.get(stem, 0) < HISTORY_MIN_INTERVAL_S:
        return
    d = HISTORY_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, d / time.strftime("%Y%m%d-%H%M%S.json", time.localtime(src.stat().st_mtime)))
    _last_snapshot[stem] = now
    for old in sorted(d.glob("*.json"))[:-HISTORY_KEEP]:
        old.unlink()


def write_label(stem: str, label: dict) -> None:
    p = L.label_path(stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(L.dumps(label))
    os.replace(tmp, p)


def prepped(stem):
    try:
        meta = json.loads((CACHE_DIR / stem / 'meta.json').read_text())
        return 'pyramid' in meta.get('layers', {})
    except (OSError, ValueError, TypeError):
        return False


def allocation_for(stem):
    try:
        allocation = json.loads((L.ROOT / 'labelling_assignments.json').read_text())
    except FileNotFoundError:
        return None, False
    return allocation.get('assignments', {}).get(stem), stem in allocation.get('protected_stems', [])


def page_summary(stem: str, split: str) -> dict:
    label = read_label(stem)
    assignee, protected = allocation_for(stem)
    row = {"stem": stem, "split": split, "image_name": L.page_info(stem)["image_name"],
           "prepped": prepped(stem), "has_label": label is not None, "assignee": assignee, "in_progress": protected}
    if label is None:
        row.update(done=False, n_errors=1, n_warnings=0, n_unreviewed=0)
    else:
        v = L.validate(label, stem)
        row.update(done=not v["errors"], n_errors=len(v["errors"]), n_warnings=len(v["warnings"]),
                   n_unreviewed=v["n_unreviewed"])
    return row


class Handler(SimpleHTTPRequestHandler):
    server_version = "LabelServer/1"

    def log_message(self, fmt, *args):  # quieter: only API calls and errors
        if "/api/" in (self.path or "") or (args and str(args[1]) >= "400"):
            sys.stderr.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    # ---- helpers
    def send_json(self, obj, status=HTTPStatus.OK):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, base: Path):
        try:
            path = path.resolve()
            path.relative_to(base.resolve())  # no traversal
        except (ValueError, OSError):
            return self.send_error(HTTPStatus.FORBIDDEN)
        if not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "text/javascript"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        # tiles never change once written; page code may change during development
        self.send_header("Cache-Control", "max-age=3600" if base == CACHE_DIR else "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def stem_from(self, path: str, prefix: str) -> str | None:
        stem = unquote(path[len(prefix):])
        return stem if stem in L.label_stems() else None

    # ---- routes
    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/index.html"):
            return self.send_file(PAGE_DIR / "index.html", PAGE_DIR)
        if path.startswith("/labelling/"):
            return self.send_file(PAGE_DIR / unquote(path[len("/labelling/"):]), PAGE_DIR)
        if path.startswith("/cache/"):
            return self.send_file(CACHE_DIR / unquote(path[len("/cache/"):]), CACHE_DIR)
        if path == "/api/vocab":
            return self.send_json(list(L.load_vocab()))
        if path == "/api/pages":
            return self.send_json([page_summary(s, sp) for s, sp in L.label_stems().items()])
        if path.startswith("/api/legend-suggestions/"):
            stem = self.stem_from(path, "/api/legend-suggestions/")
            if stem is None:
                return self.send_json({"error": "unknown stem"}, HTTPStatus.NOT_FOUND)
            try:
                from detect_legend import detect
                return self.send_json(detect(CACHE_DIR / stem))
            except ImportError as exc:
                return self.send_json({"error": f"Legend detection dependency missing: {exc}. Activate .venv, run python -m pip install -r requirements.txt, and restart the server."}, HTTPStatus.SERVICE_UNAVAILABLE)
            except (OSError, ValueError, KeyError) as exc:
                return self.send_json({"error": f"Could not detect swatches: {exc}"}, HTTPStatus.CONFLICT)
        if path.startswith("/api/proposals/"):
            stem = self.stem_from(path, "/api/proposals/")
            if stem is None:
                return self.send_json({"error": "unknown stem"}, HTTPStatus.NOT_FOUND)
            p = CACHE_DIR / stem / "proposals.json"
            if not p.exists():
                return self.send_json({"error": "run prep to build proposals"}, HTTPStatus.CONFLICT)
            return self.send_json(json.loads(p.read_text()))
        if path.startswith("/api/page/"):
            stem = self.stem_from(path, "/api/page/")
            if stem is None:
                return self.send_json({"error": "unknown stem"}, HTTPStatus.NOT_FOUND)
            meta_p = CACHE_DIR / stem / "meta.json"
            if not meta_p.exists():
                return self.send_json({"error": "page not prepped: run scripts/prep_labelling.py"},
                                      HTTPStatus.CONFLICT)
            label = read_label(stem) or L.empty_label(stem, L.label_stems()[stem])
            return self.send_json({"meta": json.loads(meta_p.read_text()), "label": label,
                                   "validation": L.validate(label, stem),
                                   "version": version_of(L.label_path(stem))})
        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_PUT(self):
        path = urlsplit(self.path).path
        if not path.startswith("/api/label/"):
            return self.send_error(HTTPStatus.NOT_FOUND)
        stem = self.stem_from(path, "/api/label/")
        if stem is None:
            return self.send_json({"error": "unknown stem"}, HTTPStatus.NOT_FOUND)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))), parse_constant=lambda v: (_ for _ in ()).throw(ValueError("non-finite JSON number")))
            label, version = body["label"], str(body["version"])
            if not isinstance(label, dict):
                raise ValueError("label must be an object")
        except (ValueError, KeyError, TypeError) as e:
            return self.send_json({"error": f"bad request: {e}"}, HTTPStatus.BAD_REQUEST)
        with _lock:
            current = version_of(L.label_path(stem))
            if version != current:
                return self.send_json({"error": "conflict", "version": current}, HTTPStatus.CONFLICT)
            snapshot(stem)
            label = L.recompute_depths(L.sync_legend_labels(label))
            write_label(stem, label)  # saved even when invalid
            new_version = version_of(L.label_path(stem))
        return self.send_json({"validation": L.validate(label, stem), "version": new_version, "label": label})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    require_dependencies()
    if not all(prepped(stem) for stem in L.label_stems()):
        raise SystemExit('Pages are not prepared. Run python scripts/prep_labelling.py --workers 2 successfully before starting the server.')
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"labelling: http://127.0.0.1:{args.port}/  ({len(L.label_stems())} pages)  Ctrl+C to stop")
    print(f"Python: {sys.executable}")
    for initial in ('E', 'B', 'N'):
        print(f"  {initial}: http://127.0.0.1:{args.port}/?assignee={initial}")
    if not shutil.which('tesseract'):
        print('Legend OCR needs Tesseract: brew install tesseract (macOS), or sudo apt-get install tesseract-ocr (Ubuntu). Manual legend names remain available.')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
