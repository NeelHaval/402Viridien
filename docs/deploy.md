# Strata deployment

## Source receipt

The first sync comes from live release `combined-3d7af6dd61d7`, image
`sha256:b240c89e43f50a403131c624ecdb9edbf1ed5539444adfa80e9de4f874aec779`.
Backend app and tests were copied with `docker cp` from `stratum-app`.
Frontend source and public assets came from the same release directory because
its final container contains only compiled frontend assets. The release Dockerfile
is included, with its private source-manifest COPY removed so a public checkout builds.

## Local use

Use Python 3.12, Node 22, GNU Make, and Tesseract on PATH. From `demo`, copy
`.env.example` to `.env`, then run `make setup` and `make run`.
The Makefile reads `.env`. Credentials are optional for local use. Without Gemini
credentials, the app uses its deterministic report fallback. Without a local source
catalog, the source library is empty. Native upload processing is enabled by default.
Run `make test` for backend tests, frontend typecheck, and frontend build.

For a container, copy `.env.example` to `.env` and run `docker compose up --build`
from `demo`. The Docker build context is the repository root. State uses the
`app-state` named volume. The image creates its mount point for UID 1001.

## Production settings

The live app listens on VM loopback port 8010. Caddy terminates HTTPS and forwards
all requests to that port. The old `stratum-frontend-cd9b26a83bf4` container is kept
for rollback and does not serve the current public route.

The VM uses an external environment file at `/home/viridien/secrets/gemini.env`.
Keep this file outside Git. The deployed model is `gemini-2.5-flash`.
Production sets `APP_RELEASE_ID` separately from `APP_CODE_RELEASE_ID` to preserve
saved processing identities across code releases.

S19 production mode requires `S19_ADAPTER_ENABLED=1`, a pinned runtime directory
in `S19_ADAPTER_ROOT`, and its OCR dependencies in `S19_SITE_PACKAGES`.
Set `S19_BATCH_ROOT`, `APP_SOURCE_CATALOG_ROOT`, and `S19_COMPUTE_LOCK` to the
mounted batch, approved source catalog, and compute lock. Local defaults are under
`demo/backend/data`; do not leave these unset on the VM. The adapter worker uses
Linux file locks and resource limits. Enable it only on Linux with the pinned assets.

Production also mounts persistent `/state`, OCR model directories, and S19 Python
package directories. These runtime assets, source scans, staged results, databases,
secrets, caches, and installed dependencies are deliberately excluded from this sync.
Large model and scan assets are managed separately. Frontend fonts and icons remain
in source because the live UI needs them.

The public compose file is a local deployment base. Keep production mount paths,
resource limits, proxy setup, and the external credential-file reference in the VM
release procedure. Do not copy live environment contents into this repository.

## Second live sync

The final sync includes release `combined-7a76cd1b718c`, image
`sha256:839775785c8fb476f20e30a61692e2cbca494fbdb268bdae44011f67c9c4ceb0`.
Its backend was copied again from the live app. Its frontend is unchanged.
This round adds checked bbox-sidecar binding for older saved results and avoids
reading incomplete result JSON while processing is in progress.

Validation: the deployer ran the exact combined image suite: 344 passed, 2 skipped,
2 warnings in 17.73s. The first public sync, including path overrides, ran on the
production image with public app/tests mounted: 339 passed, 2 skipped, 3 warnings
in 9.80s. Frontend `npm run typecheck` and `npm run build` passed; Vite built in
6.58s with a large MapLibre chunk warning. Windows backend tests cannot run the
Linux-only adapter checks (`fcntl` is absent). The literal `make setup && make run`
command was not run end to end on this Windows host. Use Linux or WSL for S19.

The final public portability changes also passed a focused regression check on the
new live image: `python -m pytest -q tests/test_catalog_result_binding.py
tests/test_result_publication_review.py tests/test_staged_result_contract.py`:
19 passed, 3 warnings in 0.24s. Warnings cover deprecated HTTP test dependencies
and the read-only container test cache.
