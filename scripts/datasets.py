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
import re
import subprocess
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def credential():
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    if shutil.which('gh'):
        result = subprocess.run(['gh', 'auth', 'token'], capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    result = subprocess.run(["git", "credential", "fill"],
                            input="protocol=https\nhost=github.com\n\n", text=True,
                            capture_output=True, timeout=30,
                            env={**os.environ, 'GIT_TERMINAL_PROMPT':'0', 'GCM_INTERACTIVE':'never'})
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not values.get("password"):
        raise RuntimeError("Run gh auth login, use Git Credential Manager, or set GH_TOKEN with repository read access")
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


def portable_path(name):
    """Reject names whose meaning changes across Windows/POSIX filesystems."""
    path=PurePosixPath(name)
    reserved=re.compile(r'^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?$',re.I)
    if (path.is_absolute() or not path.parts or '..' in path.parts or '\\' in name or ':' in name
            or any(part.endswith(('.', ' ')) or reserved.fullmatch(part) or any(ord(c)<32 or c in '<>"|?*' for c in part) for part in path.parts)):
        raise ValueError(f'Unsafe or nonportable archive path: {name}')
    return path


def safe_extract(archive, destination, max_bytes=3_000_000_000):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if sum(item.file_size for item in members) > max_bytes:
            raise ValueError("Dataset expands beyond the supported size budget")
        seen = set()
        for item in members:
            path = portable_path(item.filename)
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
        files={PurePosixPath(item.filename) for item in members if not item.is_dir()}
        file_names={path.as_posix().casefold() for path in files}
        if any(parent.as_posix().casefold() in file_names
               for item in members for parent in PurePosixPath(item.filename).parents):
            raise ValueError('Archive file/directory paths conflict')
        # Stage and CRC-check every member before touching an existing dataset.
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='viridian-data-',dir=destination.parent) as temporary:
            staging=Path(temporary)
            for item in members:
                if item.is_dir():continue
                relative=PurePosixPath(item.filename)
                staged=staging.joinpath(*relative.parts)
                staged.parent.mkdir(parents=True,exist_ok=True)
                with bundle.open(item) as source,staged.open('wb') as output:
                    shutil.copyfileobj(source,output)
                target=destination.joinpath(*relative.parts)
                if target.exists() and (not target.is_file() or sha256(target)!=sha256(staged)):
                    raise ValueError(f'Existing dataset file differs; preserve your work and extract into a fresh directory: {relative}')
                for parent in target.parents:
                    if parent==destination.parent:break
                    if parent.exists() and not parent.is_dir():
                        raise ValueError('Existing dataset file conflicts with an archive directory')
            # Revalidate after staging: a destination ancestor may have been
            # replaced since initial path validation. This closes that stable
            # swap window, not every hostile concurrent filesystem race.
            for relative in files:
                target=destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError('Archive destination path changed or escapes through a symlink')
            for relative in sorted(files):
                target=destination.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(destination):
                    raise ValueError('Archive destination path changed or escapes through a symlink')
                if not target.exists():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    staging.joinpath(*relative.parts).replace(target)


def fetch(name, destination, lock_path, *, extract=True, cache_dir=None, use_gh=False):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    entry = lock["bundles"][name]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://api.github.com/repos/{lock['repository']}/releases/assets/{int(entry['asset_id'])}"
    cache = Path(cache_dir) if cache_dir is not None else ROOT / "data" / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (entry['sha256'] + Path(entry['asset_name']).suffix)
    if use_gh and shutil.which('gh') and (not archive.exists() or archive.stat().st_size!=entry['size'] or sha256(archive)!=entry['sha256']):
        if Path(entry['asset_name']).name!=entry['asset_name']:
            raise ValueError('Release asset name must be a plain filename')
        print(f"Downloading {entry['asset_name']} ({entry['size']/1024**2:.1f} MiB)",flush=True)
        with tempfile.TemporaryDirectory(prefix='release-download-',dir=cache) as temporary:
            downloaded=Path(temporary)/entry['asset_name']
            for attempt in range(3):
                with downloaded.open('wb') as output:
                    result=subprocess.run(['gh','api',url,'--header','Accept: application/octet-stream'],
                        stdout=output,stderr=subprocess.PIPE,timeout=1800)
                if result.returncode==0:break
                if attempt<2:time.sleep(attempt+1)
            else:
                raise RuntimeError('Private download failed. Check gh auth login and repository access: '+
                                   result.stderr.decode('utf-8',errors='replace').strip())
            if downloaded.stat().st_size!=entry['size'] or sha256(downloaded)!=entry['sha256']:
                raise ValueError('Dataset checksum or size mismatch')
            downloaded.replace(archive)
    if not archive.exists() or archive.stat().st_size!=entry['size'] or sha256(archive) != entry["sha256"]:
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
    if extract:
        safe_extract(archive, destination)
    print(f"Verified {name}: {entry['sha256']} -> {destination}")
    return archive


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
                    portable_path(record[key])
                    if ('\\' in record[key] or ':' in record[key] or Path(record[key]).is_absolute() or PurePosixPath(record[key]).is_absolute()):
                        raise ValueError("Bundle manifests must use portable relative asset paths")
                    lexical=manifest.parent/record[key]
                    if lexical.absolute()!=lexical.resolve():
                        raise ValueError('Portable bundles cannot contain symlinked asset references; copy the asset bytes first')
                    files.add((manifest.parent / record[key]).resolve())
    if any(not path.is_relative_to(base) for path in files):
        raise ValueError("All bundled files must be under the manifest directory")
    output = Path(output)
    if output.resolve() in files:
        raise ValueError("Bundle output must not overwrite a source manifest or asset")
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
