#!/usr/bin/env python3
"""Export validated free layers to challenge CSV and <stem>.composition.json.

    python scripts/export_gt_csv.py [stem ...] [--out labels_csv]

One row per layer; dominant entry supplies its same-symbol cluster and names.
Gaps produce no rows. Invalid pages are skipped and exit status is nonzero.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelslib as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*")
    ap.add_argument("--out", default=str(L.ROOT / "labels_csv"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = L.label_stems()
    todo = args.stems or list(stems)
    skipped = []
    for stem in todo:
        p = L.label_path(stem)
        if stem not in stems or not p.exists():
            skipped.append((stem, "no label file" if stem in stems else "not a labelled page")); continue
        label = L.recompute_depths(L.sync_legend_labels(json.loads(p.read_text())))
        v = L.validate(label, stem)
        if v["errors"]:
            skipped.append((stem, f"{len(v['errors'])} validation errors, first: {v['errors'][0]['msg']}")); continue
        for w in v["warnings"]:
            print(f"warning {stem}: {w['msg']}")
        rows = L.to_csv_rows(label)
        with open(out / f"{stem}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=L.CSV_HEADER)
            w.writeheader()
            w.writerows(rows)
        (out / f"{stem}.composition.json").write_text(json.dumps(L.recompute_depths(label)["layers"], indent=2) + "\n")
        print(f"wrote {out.name}/{stem}.csv  ({len(rows)} rows)")
    for stem, why in skipped:
        print(f"SKIPPED {stem}: {why}")
    print(f"\nexported {len(todo) - len(skipped)}/{len(todo)}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
