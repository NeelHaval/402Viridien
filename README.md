# Strata

Turn a scanned mud log into named lithology intervals, image coordinates, and a CSV that can be checked.
Built by **402_Payment_Required** for Viridien's *Decoding the Subsurface* challenge at HackTheBurgh XIII.

![All three released pages: source pixels beside saved standalone S19 predictions](docs/showcase/01-released-pages-s19.png)

**Saved predictions, no reference interval labels.** All three released pages use the same vertical pixel scale.
These are fixed middle excerpts, not selected successes. The sparse Woody output is retained.
The axis is full-image row in pixels. S19 depths in these saved outputs are uncalibrated.
This is **standalone S19 evidence**, not proof that the Strata app uses S19. Open the image to read the labels.

## What this version establishes

| Evidence | State | Exact scope |
|---|---|---|
| Saved app upload → ready → CSV → report | Yes | One generated four-band fixture; historical image `fcb8b3196042` |
| Standalone S19 saved predictions | Yes | Three released pages; no reference interval truth or accuracy estimate |
| Official depth export for those saved S19 files | No | Calibration was absent; blanks must not become zeros |
| Final Outfit/Strata/provider build | Pending | Deployment owner's final source, image digest, and browser receipt |
| Core candidate `559c99cb4e24` accepted | No | Preserve its source archive; it is not the final release |
| Judge source access and submission route | Pending | Team write access does not prove judge access |

Current verified private repository: [NeelHaval/viridian402](https://github.com/NeelHaval/viridian402).
The requested name is `NeelHaval/402_viridien`; an admin must complete and verify the rename.
Python package names remain `viridian402`. Repository naming does not change the model or app version.

## How it works

1. Read the original scan and supplied boxes for the lithology column, legend, and depth column.
2. Read legend swatches and captions. Keep the page's names, cluster IDs, and unknown labels.
3. Detect changes in the column and assign interval labels. S19 combines learned boundary and matching components; the app has a separate OCR pipeline.
4. Map image rows to depth with that scan's calibration. Record printed units and datum; missing calibration blocks official depth export.
5. Write CSV rows and an annotated column. Check boundaries, label mapping, hashes, units, and document coverage before submission.

![Generated fixture legend swatches, saved OCR captions, and cluster IDs](docs/showcase/02-fixture-legend.png)

The left side is an exact crop of the generated fixture. The right side uses labels and IDs from its saved API CSV.
No unknown occurs here. On other scans, an unknown remains explicit; colour alone does not identify a rock.
This receipt came from historical app image `sha256:5b938b2669b04dcf86220b6c3bff712904d45cf4fcc29848b9df8ec8bbd65451`.

## Outputs and units

The official example/notebook header is:

```csv
top_depth,bottom_depth,lithology,top_y,bottom_y,cluster_id
```

The challenge slide header is:

```csv
top_y,bottom_y,lithology,cluster_id,top_depth,base_depth
```

`base_depth` is an alias for the exact value of `bottom_depth`. It does not convert units.
Coordinates are full-image pixels, top-left origin; intervals are sorted, nonoverlapping, and bottom-exclusive.
Keep each scan's printed unit and datum with its calibration receipt. There is no universal feet assumption.
Missing or nonfinite depths block the pinned exporter. Raw incomplete predictions remain diagnostics.
An export check does not prove calibration quality, correct geology, or complete annotated documents.

The saved app endpoint uses another column order: `top_y,bottom_y,cluster_id,lithology,top_depth,base_depth`.
Use an explicit adapter for the required schema; the app download is not a format guarantee.
See [the output contract and pinned export command](docs/output-contract.md).

## A generated example

![Four-band generated source with its saved app intervals and depth values](docs/showcase/03-fixture-source-prediction.png)

**Regression check, not an accuracy estimate.** This is a generated fixture, not a challenge scan or training result.
Source commit: `fcb8b319604200b999902b44753afb55b7da133d`; image digest is stated above.
Three internal drawn edges: `700,1000,1300`; saved predictions: `696,1000,1304` (maximum offset **4 px**).
The same saved CSV contains these four rows. Depth values are in **fixture feet only**.

```csv
top_y,bottom_y,cluster_id,lithology,top_depth,base_depth
400,696,1,Shale grey,909.9,1206.0
696,1000,0,Sandstone,1206.0,1510.1
1000,1304,2,Halite,1510.1,1814.2
1304,1600,1,Shale grey,1814.2,2110.3
```

The TIFF, boxes, and HTTP receipt were hash-checked before these figures were made.
[Figure provenance, crops, versions, and full hashes](docs/showcase/README.md) explain the derivation.
No final Strata screenshot is included until the deployment owner supplies an accepted build receipt.

## Try it

Use Python 3.12. Clone only source paths; no dataset download is needed for the manufactured example:

```sh
gh repo clone NeelHaval/viridian402 -- --filter=blob:none --sparse
cd viridian402
git sparse-checkout set src scripts configs docs tests examples stress_testing .github notebooks data/provenance demo
python scripts/local_test.py
```

That command checks a manufactured example and runs no learned model.
For the checked-in app source, use Python 3.12, Node/npm, and GNU Make:

```sh
cd demo
make setup
make run
```

Open `http://localhost:8000`; inspect `/api/health` before upload. Use one backend worker.
The map starts empty. Uploads create local state; keep scans and generated CSVs private.
The app needs its pinned RapidOCR package or documented Tesseract fallback. Map/font services need network.
Setup installs dependencies; these instructions do not claim reproduction of the final deployed image.
See [app setup and API](demo/README.md) and [the experiment guide](docs/experiment-guide.md).

## Code and evidence map

| Path | Purpose |
|---|---|
| `demo/` | FastAPI upload/OCR pipeline and React/MapLibre app |
| `src/viridian402/` | Experiment CLI, metrics, reports, and adapters |
| `scripts/challenge_export.py` | Pinned saved-row validator and two-schema exporter; no inference |
| `docs/showcase/` | Three figures, generated-fixture CSVs, saved receipt, and provenance |
| `docs/local-testing.md` | Bounded setup, dataset notices, and permitted development suites |
| `docs/experiment-guide.md` | Earlier experiment workflow and evidence scopes |
| `requirements*.lock`, `demo/frontend/package-lock.json` | Dependency records; not a full deployed-image lock |

S19's separate source/config/weight handoff is checksum-pinned by its owner. These docs do not wire it into the app.
Historical PR5/import/release material is a source handoff, not final app acceptance. PR6 is not promoted here.
The lab owner reported a cached S19 composite replay of `0.25237305749927796` on **five DEV windows from five wells**.
Those windows share one programmatic reference family. This is a development score, not independent or full-document accuracy.
Depth, broader DEV, synthetic gates, sparse-human, and full challenge validation were not run in that lab closure.
The three released pages above have no reference interval truth, so their accuracy is unavailable.

## Limits, data, and disclosure

Thin beds, texture changes, unreadable legends, missing boxes, and bad depth ticks can change or prevent a result.
Legend captions are page-specific. Local Clay/Shale scoring equivalence must not relabel exported captions.
Any app scores or area summaries need their own validation; they are not geological or commercial truth.
The optional Gemini provider patch is separate and is not asserted to be active in the accepted app.

These figures are shared only in this private repository under the owner's scope. No public scan licence is implied.
No protected truth gallery, held-out TEST labels, dataset archive, weight bytes, `.env`, or credential is added here.
Source ownership does not grant redistribution rights to challenge data or third-party models. Check original terms.
No root project licence was found at the pinned core commit; dependency licences remain with their owners.

AI tools assisted implementation, review, testing, coordination, and writing: OpenAI Codex/GPT models, with Claude Opus 5.5 advice on evidence selection.
Existing OCR models, NumPy/SciPy/OpenCV/Pillow/PyTorch, FastAPI, React, MapLibre, Vite, and other dependencies are external contributions.
The supplied challenge notebook, example, and page annotations are organiser inputs. The generated fixture is labelled.
This is a partial tool/component record; the team must confirm exact pre-event code, provider use, and event-built contributions before entry.
The [event rules](https://hacktheburgh.com/code-of-conduct) require AI and pre-existing contribution disclosure. Disclosure does not establish prize eligibility.

The private submission bundle uses `python tools/verify_package.py --strict` as its final gate once its owner installs and checks it.
`--integrity-only` checks copy/hash integrity; it does not mean submission-ready. This README change does not submit the entry.
