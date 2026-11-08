"""Fetch checksum-pinned private release data, without storing credentials.

Run from the repository root. GITHUB_TOKEN/GH_TOKEN needs contents:read for this
repository. Locally, Git Credential Manager is also supported. Only trusted lock
files committed in this repository should be used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def credential():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n", text=True,
                            capture_output=True, timeout=60)
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not values.get("password"):
        raise RuntimeError("Authenticate GitHub with Git Credential Manager or set GH_TOKEN")
    return values["password"]


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward repository credentials to release asset storage."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            redirected.remove_header("Authorization")
        return redirected


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def safe_extract(archive, destination, max_bytes=3_000_000_000):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if sum(item.file_size for item in members) > max_bytes:
            raise ValueError("Dataset expands beyond the supported size budget")
        seen = set()
        for item in members:
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or not path.parts or ".." in path.parts
                    or "\\" in item.filename or ":" in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError(f"Unsafe archive path: {item.filename}")
            target = destination.joinpath(*path.parts).resolve()
            if not target.is_relative_to(destination) or target == destination:
                raise ValueError("Archive path escapes destination")
            if str(target).casefold() in seen:
                raise ValueError("Duplicate archive member")
            seen.add(str(target).casefold())
        # Validate the entire archive before creating any file.
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def fetch(name, destination, lock_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    entry = lock["bundles"][name]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://api.github.com/repos/{lock['repository']}/releases/assets/{int(entry['asset_id'])}"
    cache = ROOT / "data" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"{entry['sha256']}.zip"
    if not archive.exists() or sha256(archive) != entry["sha256"]:
        opener = urllib.request.build_opener(SafeRedirect())
        temporary = archive.with_suffix(".part")
        try:
            # Bounded range requests also work through proxies which truncate
            # large streamed responses. Verify each range before appending.
            token = credential()
            with temporary.open("wb") as output:
                copied = 0
                while copied < entry["size"]:
                    end = min(copied + 4 * 1024 * 1024, entry["size"]) - 1
                    request = urllib.request.Request(url + f"?download=1&chunk={copied}", headers={
                        "Authorization": "Bearer " + token, "Accept": "application/octet-stream",
                        "Range": f"bytes={copied}-{end}", "X-GitHub-Api-Version": "2022-11-28"})
                    last_error = None
                    for attempt in range(3):
                        try:
                            with opener.open(request, timeout=120) as response:
                                expected = f"bytes {copied}-{end}/{entry['size']}"
                                if response.status != 206 or response.headers.get("Content-Range") != expected:
                                    raise ValueError("Release storage did not honor the bounded byte range")
                                chunk = response.read(end - copied + 2)
                            if len(chunk) != end - copied + 1:
                                raise ValueError("Incomplete dataset byte range")
                            output.write(chunk)
                            copied += len(chunk)
                            break
                        except (OSError, ValueError) as error:
                            last_error = error
                    else:
                        raise RuntimeError("Dataset download failed after three range attempts") from last_error
            if copied != entry["size"] or sha256(temporary) != entry["sha256"]:
                raise ValueError("Dataset checksum or size mismatch")
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    safe_extract(archive, destination)
    print(f"Verified {name}: {entry['sha256']} -> {destination}")


def pack(manifests, output):
    """Collect only manifest-referenced assets, retaining relative paths."""
    manifests = [Path(path).resolve() for path in manifests]
    base = manifests[0].parent
    files = set(manifests)
    # Preserve source attribution, QA notes, evidence tags and the import ledger.
    if (base / "provenance").exists():
        files.update(path.resolve() for path in (base / "provenance").rglob("*") if path.is_file())
    if (base / "source-snapshot.json").is_file():
        files.add(base / "source-snapshot.json")
    for manifest in manifests:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for record in data["records"]:
            for key in ("image", "annotation", "truth"):
                if record.get(key):
                    if Path(record[key]).is_absolute() or PurePosixPath(record[key]).is_absolute():
                        raise ValueError("Bundle manifests must use portable relative asset paths")
                    files.add((manifest.parent / record[key]).resolve())
    if any(not path.is_relative_to(base) for path in files):
        raise ValueError("All bundled files must be under the manifest directory")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
        for path in sorted(files):
            info = zipfile.ZipInfo(path.relative_to(base).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, bundle.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target)
    print(json.dumps({"sha256": sha256(output), "size": output.stat().st_size,
                      "asset_name": output.name, "manifests": [path.name for path in manifests]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("fetch")
    download.add_argument("bundle")
    download.add_argument("--destination", default="data/local")
    download.add_argument("--lock", default="data/datasets.lock.json")
    package = sub.add_parser("pack")
    package.add_argument("manifests", nargs="+")
    package.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.bundle, args.destination, args.lock)
    else:
        pack(args.manifests, args.output)


if __name__ == "__main__":
    main()
