#!/usr/bin/env python
"""Pick a well-grouped ~20% TEST split of the lithology dataset for hand-labelling.

    python scripts/make_split.py --seed 42 --test-frac 0.2
    python scripts/make_split.py --force-test <stem> --exclude-test <stem>
    python scripts/make_split.py --groups-only        # inventory + groups table, then stop

Writes splits.json, splits/page_features.csv, splits/report.md and
splits/{test,train}_contact_sheet.png. Never modifies ../viridien/dataset.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from splitlib import inventory, search, report  # noqa: E402
from labelling_setup import DATASET

ROOT = Path(__file__).resolve().parent.parent

# The only B&W well is a separate holdout; log01 was used for baseline development.
BW_HOLDOUT_GROUPS = ("0000000009_d-582",)

CSV_COLUMNS = [
    "stem", "group_id", "image_mode", "is_colour", "mean_sat", "log_type", "column_height_px",
    "depth_range_ft", "px_per_ft", "legend_aspect", "legend_area_frac", "blank_frac",
    "texture_diversity", "feature_warnings", "col_thumb", "legend_thumb",
]


def print_groups(groups, header_api):
    print(f"\n{'group_id':32s} {'n':>2s}  stems")
    for g, stems in groups.items():
        print(f"{g:32s} {len(stems):2d}  " + " | ".join(stems))
    print(f"\n{len(groups)} groups, {sum(map(len, groups.values()))} pages")



def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--out-dir", default=str(ROOT / "splits"))
    ap.add_argument("--splits-json", default=str(ROOT / "splits.json"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-iter", type=int, default=search.DEFAULT_N_ITER)
    ap.add_argument("--force-test", action="append", default=[], metavar="STEM")
    ap.add_argument("--exclude-test", action="append", default=[], metavar="STEM")
    ap.add_argument("--group-overrides", default=None,
                    help="extra JSON {stem_or_group_id: group_id} on top of the built-in field merges")
    ap.add_argument("--jobs", type=int, default=0, help="feature workers (0 = auto)")
    ap.add_argument("--groups-only", action="store_true", help="stop after printing the groups table")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. inventory + sanity checks
    pages, issues = inventory.load_inventory(args.dataset)
    print(f"{len(pages)} paired pages")
    for msg in issues:
        print("  !", msg)

    # 2. groups
    groups = inventory.build_groups(pages, inventory.load_overrides(args.group_overrides))
    print_groups(groups, inventory.HEADER_API)
    gwarn = inventory.grouping_warnings(inventory.build_groups(pages))  # on the raw (un-merged) groups
    if args.groups_only:
        for w in gwarn:
            print("  ?", w)
        return 0

    # 3. features
    from splitlib import features  # imported lazily: heavy deps only needed past --groups-only
    rows = features.compute_all(pages, out / "cache", n_jobs=args.jobs)
    gid_of = {s: g for g, ss in groups.items() for s in ss}
    for r in rows:
        r["group_id"] = gid_of[r["stem"]]
    with open(out / "page_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # 4. split search
    bw_stems = [s for g in BW_HOLDOUT_GROUPS for s in groups.get(g, [])]
    excluded = [s for s in bw_stems if '_log01_' in s]
    test_bw = [s for s in bw_stems if s not in excluded]
    if set(args.force_test) & set(bw_stems):
        ap.error('B&W holdout pages cannot be forced into the colour test split')
    exclude = list(args.exclude_test) + bw_stems
    split = search.search_split(rows, groups, test_frac=args.test_frac, seed=args.seed, n_iter=args.n_iter,
                                force_test=args.force_test, exclude_test=exclude)
    split['train'] = [s for s in split['train'] if s not in bw_stems]

    # 5. outputs
    method = (f"group-level random search ({split['n_iter']} candidates, seed {args.seed}) minimising "
              "size deviation + strata TV + numeric SMD + empty-stratum / dominant-group penalties; "
              f"groups = API-10 well id with field-level merges; {', '.join(BW_HOLDOUT_GROUPS)} held out as test_bw; log01 excluded")
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    Path(args.splits_json).write_text(json.dumps({
        "seed": args.seed,
        "created": created,
        "method": method,
        "test_frac": args.test_frac,
        "test": split["test"],
        "train": split["train"],
        "test_bw": test_bw,
        "excluded": excluded,
        "test_bw_note": "Entire B&W well held out; log01 excluded because used for baseline development.",
        "groups": groups,
        "score": split["score"],
    }, indent=2) + "\n")

    meta = {"seed": args.seed, "created": created, "method": method, "test_frac": args.test_frac,
            "n_iter": args.n_iter, "argv": " ".join(sys.argv[1:]) or "(defaults)",
            "force_test": args.force_test, "exclude_test": exclude}
    rp = out / "report.md"
    colour_groups = {g:ss for g,ss in groups.items() if g not in BW_HOLDOUT_GROUPS}
    colour_rows = [r for r in rows if r['stem'] not in bw_stems]
    report.write_report(rp, colour_rows, colour_groups, split, meta)
    with open(rp, "a") as f:
        f.write("\n## Grouping decisions\n\n")
        f.write("Group id = first 10 digits of the US API number; otherwise prefix up to the 2nd `_`. "
                "Field-level merges applied on top (same block/pad/lease shares legend + geology):\n\n")
        for k, v in inventory.DEFAULT_GROUP_OVERRIDES.items():
            f.write(f"- `{k}` -> `{v}`\n")
        f.write("\nAPI numbers read from headers of pages without one in the filename "
                "(none matched an API-named group, so no cross-merges):\n\n")
        for k, v in inventory.HEADER_API.items():
            f.write(f"- `{k}`: {v}\n")
        f.write("\nAutomatic grouping heuristics flagged (before merges):\n\n")
        f.write("".join(f"- {w}\n" for w in gwarn))
        f.write("\n## B&W holdout (`test_bw`)\n\nEntire well held out of training. "
                "log01 excluded because it was used for baseline development. "
                "Tables above describe colour train/test pages only. The split search score "
                "was computed before removing the held-out well from training.\n\n")
        f.write("".join(f"- `{s}`\n" for s in test_bw))
        f.write("\n## Inventory issues\n\n" + ("".join(f"- {m}\n" for m in issues) or "- none\n"))

    report.contact_sheet(rows, split["test"], out / "test_contact_sheet.png",
                         f"TEST ({len(split['test'])} pages, seed {args.seed})", groups=groups)
    report.contact_sheet(rows, split["train"], out / "train_contact_sheet.png",
                         f"TRAIN ({len(split['train'])} pages, seed {args.seed})", groups=groups)
    if test_bw:
        report.contact_sheet(rows, test_bw, out / "test_bw_contact_sheet.png",
                             "test_bw (held out, reported separately)", groups=groups)

    s = split["score"]
    print(f"\nTEST: {s['n_test_pages']} pages / {len(split['test_groups'])} groups "
          f"(target {s['target']}), score {s['total']:.3f}")
    for st in split["test"]:
        print("  ", st)
    print("test_bw:", test_bw)
    print(f"wrote {args.splits_json}, {out}/page_features.csv, {rp}, contact sheets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
