"""Audit source access without changing network policy or the held-out dataset.

Run: python -I bakeoff/external/round3_tools/audit_access.py
This is an access audit, not a downloader or a completed triage pipeline.
"""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import requests

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "external_untrusted/round3"
SELECTION = [
    ("west_flank_coso", "West Flank Coso FORGE mud logs", "ZIP and individual PDF; indexed 6 wells."),
    ("stratagraph", "Stratagraph sample log gallery", "Gallery confirms 4 samples; engineering PDF added to route manifest."),
    ("precision", "Precision Well Logging sample mud logs", "Services page confirms 3 sample PDF links."),
    ("nwt_orogo", "Northwest Territories OROGO public well documents", "Registry discovery incomplete; cap 40. Additional Fall Stone F-01 registry route found."),
    ("mississippi", "Mississippi Office of Geology", "Hancock well folder confirms 3 mud-log viewer entries; XML/image route unresolved; cap 30."),
    ("oregon_dogami", "Oregon DOGAMI oil and gas well-log index", "Public index contains mud-labelled links; traversal/download incomplete; cap 40."),
    ("dhi_wister", "DHI Wister", "Single composite log; original bytes unavailable."),
    ("utah_forge", "Utah FORGE well 16A", "GDR landing page returned HTTP 429 in web tool; requests to that host stopped. OEDI mirror found."),
    ("gsq", "Queensland GSQ open-file reports: GEOstrip", "Single report; original bytes unavailable."),
    ("new_brunswick", "New Brunswick legislative library", "Single report potentially containing several logs; unverified."),
    ("quebec", "Qu\u00e9bec geoscience document repository", "Single report; original bytes unavailable."),
    ("bc_coal", "British Columbia coal assessment reports", "Single report; original bytes unavailable."),
]
EXTRAS = {
    "west_flank_coso": ["https://data.openei.org/submissions/7000"],
    "stratagraph": ["https://stratagraph.com/wp-content/uploads/2019/11/Engineering_Log_Example.pdf"],
    "nwt_orogo": ["https://www.orogo.gov.nt.ca/en/well-data/fall-stone-f-01"],
    "mississippi": ["https://geology.deq.ms.gov/Energy/Data02/MudLogs/Hancock/mdl123045000360000a.xml"],
    "utah_forge": ["https://data.openei.org/submissions/9750"],
}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def held_out_digest():
    """Read-only digest includes relative paths and all file contents."""
    digest = hashlib.sha256()
    paths = sorted((ROOT / "external_tests").rglob("*"))
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def main():
    if not sys.flags.isolated:
        raise SystemExit("Run Python with -I")
    data = json.loads((ROOT / "bakeoff/notes/external/round3/merged.json").read_text(encoding="utf-8"))
    before = held_out_digest()
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (research crawler)"
    last_request = {}
    results, routes, attempts = [], [], []
    for slug, match, note in SELECTION:
        source = next(row for row in data if row["name"].startswith(match))
        urls = list(dict.fromkeys(source["example_urls"] + EXTRAS.get(slug, [])))
        folder = OUT / slug
        folder.mkdir(parents=True, exist_ok=True)
        routes.append(dict(slug=slug, example_urls=urls, bulk_route=source["bulk_route"],
                           download_budget_bytes=1_500_000_000, max_file_bytes=300_000_000,
                           requests_per_second_per_host=1))
        # Probe one route per source; do not retry a policy-blocked connection.
        url = urls[0]
        host = urlsplit(url).netloc
        time.sleep(max(0, last_request.get(host, 0) + 1.05 - time.monotonic()))
        last_request[host] = time.monotonic()
        try:
            with session.get(url, stream=True, timeout=(5, 15), allow_redirects=False) as response:
                attempt = dict(url=url, http_status=response.status_code,
                               status="headers_only", body_saved=False)
                if response.status_code in (429, 503):
                    attempt["status"] = "throttled_no_retry"
        except requests.RequestException as error:
            # Keep diagnostics free of credentials, proxy URLs, and page content.
            attempt = dict(url=url, status="blocked", error_type=type(error).__name__,
                           reason="Shell network unavailable; connection through configured proxy failed.",
                           body_saved=False)
        dump(folder / "access_attempt.json", attempt)
        attempts.append(dict(slug=slug, **attempt))
        results.append(dict(slug=slug, downloaded_files=0, bytes=0, status=attempt["status"],
                            raster_or_vector="unknown", has_lith_column="unknown", has_legend="unknown",
                            converted_logs=0, truth_intervals=0, unlabelled_pages=0, notes=note))
        print(slug + ": " + attempt["status"], flush=True)
    after = held_out_digest()
    if before != after:
        raise RuntimeError("Held-out dataset changed during read-only audit")
    dump(OUT / "routes.json", routes)
    dump(OUT / "access_audit.json", dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
         attempts=attempts, external_tests_sha256_before=before, external_tests_sha256_after=after))
    summary = dict(sources=results, external_tests_round3=dict(logs=0, windows=0, intervals=0),
                   external_unlabelled=dict(pages=0), blocked=[
        "Shell network is explicitly disabled; configured proxy refuses connections. No original files saved.",
        "Web inspection cannot transfer original PDF/TIFF/ZIP bytes into this workspace.",
        "GDR Utah landing route returned HTTP 429; no immediate retry made.",
        "Download, visual triage, thumbnails, contact sheets, extraction and packaging remain incomplete."])
    dump(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
