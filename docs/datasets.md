# Add, label and version data

Dataset manifests are JSON, with paths relative to the manifest. Every evaluation case is one page window. One image may have several non-overlapping windows. Use stable IDs, well-level groups and explicit splits.

```json
{
  "schema_version": 1,
  "name": "new-human-page-v1",
  "records": [{
    "id": "my-well-page-1-y1000-7000",
    "image": "images/page-1.tif",
    "annotation": "bbox_annotation/page-1.json",
    "truth": "truth/page-1.csv",
    "source": "human",
    "group": "my-well",
    "split": "dev",
    "window": [1000, 7000],
    "truth_window": [1000, 7000],
    "tags": ["same_colour_hatch"],
    "label_provenance": {"kind": "human", "annotator": "YOUR_NAME", "review": "second-person review pending"},
    "depth_unit": "ft",
    "depth_datum": "MD"
  }]
}
```

`source` accepts `human`, `organizer`, `synthetic`, `external_labelled`, `external_unlabelled`, `challenge_unlabelled`, or `silver`. `split` accepts `train`, `dev`, `test`, or `verify`. The loader rejects duplicate IDs, cross-split groups or byte-identical images, invalid boxes, overlapping/unsorted rows, non-finite coordinates and labelled windows without truth-filled pixels. Keep a combined catalog for the entire experiment's data so cross-file group conflicts can be checked together.

Annotations contain image `width`, `height`, and `bounding_boxes` with `interpreted_lithology_column`, plus `depth_column` and `lithology_legend` when available. Boxes are `[x_min,y_min,x_max,y_max]`. Coordinates use the original image and exclusive right/bottom edges.

Truth CSV columns are `top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth`. Pixel endpoints, lithology and integer cluster IDs are required. Assign the same cluster ID to repeated visual patterns within a page; a unique ID per interval would change the clustering task. Depths can both be absent. The `base_depth` alias is accepted on input. Unknown/ambiguous regions should have no truth row. State `truth_window` even when a page is only partly labelled. Gaps inside that coverage are ignored in accuracy, and the report exposes the labelled pixel count.

## Bring in human labels

1. Use the source repo's `labelling` tool to box the legend, calibrate depths, draw boundaries and assign page-local rock entries. It runs locally; this repo does not start a second annotation service.
2. Save only your assigned JSON files. Review uncertain beds, equal-width mixed rocks, show symbols and thin layers. A structurally valid file is not proof the page is fully labelled.
3. Export CSV with the source tool's `labelling/scripts/export_gt_csv.py`, or write this CSV contract directly. Keep the original labelling JSON, annotation/reviewer identity, timestamp, depth unit/datum and uncertainty notes alongside the data.
4. Create/update the manifest. Put a new well in one split; keep all windows, rescans, resolutions and synthetic descendants in that same group. Add source URLs and licence/research-use notes for outside data.
5. Run `python -m viridian402 validate --manifest YOUR_MANIFEST.json`, inspect overlays against the original scan, and evaluate before publishing a new data bundle. Do not silently edit a previously published release.

Unlabelled outside/organizer holdout data use `truth: null` and their appropriate unlabelled source. They can immediately exercise crashes, timeouts and output validity. Accuracy becomes available only after a reviewed truth file and provenance are supplied in a new dataset version.

## Import the existing team snapshot

```sh
python -m viridian402 import-source --source-root ../source-hacktheburgh26 --dataset-root ../hack27/project/dataset --output data/local
python -m viridian402 validate --manifest data/local/all.json
```

The source checkout must contain `viridien/human_labels`, `external_shared`, `external_shared_verify`, organizer gold CSV and metadata; external images must be checked out too. The challenge dataset comes from the organizer distribution. Inputs are copied, never edited. The importer retains source metadata/provenance and writes per-suite manifests plus a combined catalog. Available small human CSV exports are not automatically treated as adjudicated truth: this initial import uses the explicitly documented reviewed human snapshot and organizer example.

To add future labels, author a new manifest or update an imported copy and create a new version. No network scraping or automatic relabelling happens during evaluation.

## Synthetic and stress data

```sh
python -m viridian402 synthetic --manifest data/local/human.json --output data/local/synthetic --seed 2026 --n 3 --height 1600
python -m viridian402 synthetic --manifest data/local/human.json --output data/local/stress --seed 2026 --n 3 --height 1600 --stress noise
```

Generation reuses labelled source interval patches and records the parent group/split and seed. Built-in demo patterns have exact manufactured truth and are useful for interface tests. Report synthetic scores separately. The source manifest must supply usable labelled bed crops for inherited rock names; missing candidates fail visibly instead of inventing labels. Add a new generator/degradation behind `synthetic.py`, give it a deterministic seed and test its label/coordinate preservation. Hard fixtures should cover empty/one-bed columns, thin alternating beds, same-colour distinct hatches, absent legends, non-rock symbols, bad depth OCR and resource limits.

## Publish a private dataset version

`scripts/datasets.py pack` creates deterministic portable ZIPs containing only manifest-referenced assets. Keep source attribution and QA metadata in the bundle too. Upload a new private GitHub Release asset, record its immutable asset ID, byte size and SHA-256 in `data/datasets.lock.json`, and review the lock change in a pull request. The download verifies size and hash before extraction, rejects traversal, symlinks and oversized archives, and never sends the GitHub token to release storage redirects. Extract into a fresh data directory to avoid mixing versions.

The default evaluation bundle includes only the images needed by the 36 supervised windows. The optional challenge bundle contains the remaining supplied scans; it is larger and isn't fetched for normal CI. No sealed tests, credentials or model weights are in these assets.
