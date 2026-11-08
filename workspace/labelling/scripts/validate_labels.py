#!/usr/bin/env python
"""Validate labels/<stem>.json for all test/test_bw pages (or the given stems).

    python scripts/validate_labels.py [stem ...]

Exit code 1 if any page has errors (a missing label file counts as an error).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labelslib as L  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*")
    args = ap.parse_args(argv)
    stems = L.label_stems()
    todo = args.stems or list(stems)
    bad = 0
    for stem in todo:
        p = L.label_path(stem)
        if stem not in stems:
            print(f"ERROR {stem}: not a test/test_bw page"); bad += 1; continue
        if not p.exists():
            print(f"ERROR {stem}: no label file ({p.relative_to(L.ROOT)})"); bad += 1; continue
        try:
            label = L.recompute_depths(L.sync_legend_labels(json.loads(p.read_text())))
        except json.JSONDecodeError as e:
            print(f"ERROR {stem}: invalid JSON: {e}"); bad += 1; continue
        v = L.validate(label, stem)
        status = "OK   " if not v["errors"] else "ERROR"
        bad += bool(v["errors"])
        print(f"{status} {stem}  ({len(v['errors'])} errors, {len(v['warnings'])} warnings, "
              f"{len(label.get('layers', []))} layers)")
        for e in v["errors"]:
            print(f"    error   {e['msg']}" + (f"  [y={e['y']}]" if e["y"] is not None else ""))
        for w in v["warnings"]:
            print(f"    warning {w['msg']}" + (f"  [y={w['y']}]" if w["y"] is not None else ""))
    print(f"\n{len(todo) - bad}/{len(todo)} pages valid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
