"""Build BSEE request tables from the saved public inventory. Use Python -I.

This script reads file metadata and local annotation file names only.
It does not read images, labels, model files, or benchmark data.
"""

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import zipfile


REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "external_untrusted" / "bsee"
NOTES = REPO / "bakeoff" / "notes" / "bsee_list"
MEMBER = "FRSWellDataRawData/mv_frs_file_inventory.txt"
QUERY_URL = "https://www.data.bsee.gov/Other/FileRequestSystem/WellData.aspx"
MUD = re.compile(r"\bmud[\s_-]*log\b", re.I)
CORE = ("GC", "MC", "AT", "GB")
NEAR = ("WR", "KC", "AC", "EB")
UNCERTAINTY = (
    "Metadata match only. Three-panel layout, lithology column, printed legend, "
    "colour, scan quality, and exact truth are not checked."
)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def date_value(text):
    if not text:
        return None
    return dt.datetime.strptime(text.split()[0], "%m/%d/%Y").date()


def number(text):
    return float(text) if text else 0.0


def doc_features(row):
    title = row["DOC_TYPE"].lower()
    area = row["AREA"].strip()
    area_score = 30 if area in CORE else 20 if area in NEAR else 10
    content_score = 0
    content = "Mud log title; lithology is not confirmed"
    if re.search(r"physical\s*formation|master\s*log|litholog", title):
        content_score, content = 25, "Title names physical formation, master log, or lithology"
    elif "formation evaluation" in title:
        content_score, content = 20, "Title names formation evaluation"
    elif re.search(r"engineering|drilling data|gas analysis|gas ratio|show", title):
        content_score, content = 10, "Title names an added drilling, gas, or show track"
    # Do not read 1in/1000ft as the target 1in/100ft scale.
    scale_match = re.search(r"\b([125])\s*in\b(?:\s*/\s*(\d+)\s*ft)?", title)
    scale = scale_match.group(1) if scale_match else "unknown"
    denominator = scale_match.group(2) if scale_match else None
    scale_score = {"1": 15, "2": 10, "5": 5}.get(scale, 0)
    if denominator and denominator != "100":
        scale_score = 0
    coordinate = "TVD" if re.search(r"\btvd\b", title) else "MD" if re.search(r"\bmd\b", title) else "not stated"
    coordinate_score = 5 if coordinate != "not stated" else 0
    span = max(0.0, number(row["BLI"]) - number(row["TLI"]))
    span_score = 10 if span >= 5000 else 5 if span >= 2000 else 0
    final_score = 5 if re.search(r"\bfinal\b", title) else 0
    score = area_score + content_score + scale_score + coordinate_score + span_score + final_score
    evidence = (
        f"Area {area}: {area_score}; {content}: {content_score}; "
        f"scale token {scale}in" + (f"/{denominator}ft" if denominator else "") +
        f": {scale_score}; depth basis {coordinate}: {coordinate_score}; "
        f"BLI minus TLI {span:g}: {span_score}; final title: {final_score}"
    )
    return {
        "metadata_score": score,
        "score_evidence": evidence,
        "scale_token_inches": scale,
        "scale_denominator_ft": denominator or "not stated",
        "depth_basis_in_title": coordinate,
        "logged_span_from_index": f"{span:g}",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("LITHOLOGY_DATASET", REPO.parent / "dataset")))
    args = parser.parse_args()
    annotation_dir = args.dataset / "bbox_annotation"
    if not annotation_dir.is_dir():
        raise SystemExit("Set LITHOLOGY_DATASET or pass --dataset. Annotation file names are needed to check overlap.")
    names = sorted(p.stem for p in annotation_dir.glob("*.json"))
    roots = sorted({m.group(1) for n in names if (m := re.match(r"(\d{10})\d{2,4}", n)) and not n.startswith("0000000009")})
    # Store only API roots, not unrelated well names.
    (NOTES / "local_api_roots.json").write_text(json.dumps({"annotation_file_count": len(names), "api_roots": roots}, indent=2) + "\n")
    snapshot = dt.date.fromisoformat(json.loads((DATA / "sources" / "FRSWellDataRawData.download.json").read_text())["retrieved_utc"][:10])
    counts = collections.Counter()
    mud_rows, recent_rows = [], []
    archive = DATA / "sources" / "FRSWellDataRawData.zip"
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo(MEMBER)
        if info.file_size > 1024**3:
            raise ValueError("Inventory member exceeds the 1 GB limit")
        with zipped.open(info) as stream:
            reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
            source_fields = reader.fieldnames
            for record, row in enumerate(reader, 2):
                counts["all_inventory_rows"] += 1
                if row["FILE_EXT"].lower() not in (".tif", ".tiff") or not MUD.search(row["DOC_TYPE"]):
                    continue
                counts["all_region_mud_tiffs"] += 1
                if row["REGION"] != "G":
                    continue
                row = {"source_record_number": record, **row}
                mud_rows.append(row)
                counts["gulf_mud_tiffs"] += 1
                run_date = date_value(row["RUN_DATE"])
                if run_date and run_date.year >= 2005 and number(row["WATER_DEPTH"]) > 1000:
                    recent_rows.append(row)
                    counts["gulf_recent_deep_mud_tiffs"] += 1
    raw_fields = ["source_record_number"] + source_fields
    write_csv(DATA / "inventory_mud_tiff.csv", mud_rows, raw_fields)
    write_csv(DATA / "inventory_recent_deep_mud_tiff.csv", recent_rows, raw_fields)
    exclusions, eligible = [], []
    for row in recent_rows:
        reason = ""
        if row["API"][:10] in roots:
            reason = "API root is present in local annotation file names"
        release_date = date_value(row["RELEASABLE_DATE"])
        if not release_date or release_date > snapshot:
            reason = "Public release date is absent or after the snapshot date"
        if reason:
            exclusions.append({**row, "exclusion_reason": reason})
        else:
            eligible.append(row)
    write_csv(NOTES / "excluded_documents.csv", exclusions, raw_fields + ["exclusion_reason"])
    grouped = collections.defaultdict(list)
    for row in eligible:
        grouped[row["API"][:10]].append({**row, **doc_features(row)})
    for docs in grouped.values():
        docs.sort(key=lambda d: (-d["metadata_score"], -number(d["logged_span_from_index"]), d["DOC_ID"]))
    ordered_roots = sorted(grouped, key=lambda root: (-grouped[root][0]["metadata_score"], -number(grouped[root][0]["logged_span_from_index"]), root))
    rank_by_root = {root: rank for rank, root in enumerate(ordered_roots, 1)}
    # Select the best three wells in each main area. Add the best one in each
    # nearby area. Then fill to 20 by rank, with at most six wells per area.
    pilot_roots = []
    for area, limit in [(a, 3) for a in CORE] + [(a, 1) for a in NEAR]:
        pilot_roots.extend([root for root in ordered_roots if grouped[root][0]["AREA"].strip() == area][:limit])
    area_counts = collections.Counter(grouped[root][0]["AREA"].strip() for root in pilot_roots)
    for root in ordered_roots:
        area = grouped[root][0]["AREA"].strip()
        if len(pilot_roots) >= 20:
            break
        if root not in pilot_roots and area_counts[area] < 6:
            pilot_roots.append(root)
            area_counts[area] += 1
    pilot_roots.sort(key=rank_by_root.get)
    selected = {}
    for root in pilot_roots:
        docs = grouped[root]
        chosen = [docs[0]]
        seen = {(docs[0]["scale_token_inches"], docs[0]["depth_basis_in_title"])}
        for doc in docs[1:]:
            kind = (doc["scale_token_inches"], doc["depth_basis_in_title"])
            if kind not in seen and len(chosen) < 3:
                chosen.append(doc)
                seen.add(kind)
        selected[root] = chosen
    pilot_ids = {d["DOC_ID"] for docs in selected.values() for d in docs}
    wells, documents = [], []
    feature_fields = list(doc_features(eligible[0])) if eligible else []
    doc_fields = ["well_rank", "api_root_10", "pilot_selected"] + raw_fields + feature_fields + ["request_page", "filename_id_query", "uncertainty"]
    for root in ordered_roots:
        docs = grouped[root]
        best = docs[0]
        pilot_docs = selected.get(root, [])
        wells.append({
            "well_rank": rank_by_root[root], "api_root_10": root,
            "best_metadata_score": best["metadata_score"], "AREA": best["AREA"].strip(),
            "BLOCK": best["BLOCK"].strip(), "LEASE": best["LEASE"].strip(),
            "WELL_NAME": best["WELL_NAME"].strip(),
            "all_api_values": " | ".join(sorted({d["API"] for d in docs})),
            "all_well_suffixes": " | ".join(sorted({d["WELL_NAME_SUFFIX"] for d in docs})),
            "best_DOC_ID": best["DOC_ID"], "best_DOC_TYPE": best["DOC_TYPE"],
            "best_RUN_DATE": best["RUN_DATE"], "best_WATER_DEPTH": best["WATER_DEPTH"],
            "eligible_documents": len(docs), "eligible_bytes": sum(int(d["FILE_SIZE"] or 0) for d in docs),
            "pilot_selected": "yes" if root in selected else "no",
            "pilot_documents": len(pilot_docs), "pilot_bytes": sum(int(d["FILE_SIZE"] or 0) for d in pilot_docs),
            "pilot_DOC_IDs": " | ".join(d["DOC_ID"] for d in pilot_docs),
            "best_score_evidence": best["score_evidence"], "uncertainty": UNCERTAINTY,
        })
        for doc in docs:
            documents.append({"well_rank": rank_by_root[root], "api_root_10": root, "pilot_selected": "yes" if doc["DOC_ID"] in pilot_ids else "no", **doc, "request_page": QUERY_URL, "filename_id_query": doc["DOC_ID"], "uncertainty": UNCERTAINTY})
    write_csv(NOTES / "ranked_wells.csv", wells, list(wells[0]))
    write_csv(NOTES / "request_documents.csv", documents, doc_fields)
    pilot = [d for d in documents if d["DOC_ID"] in pilot_ids]
    write_csv(NOTES / "pilot_request.csv", pilot, doc_fields)
    (NOTES / "pilot_filename_ids.txt").write_text("\n".join(d["DOC_ID"] for d in pilot) + "\n")
    # Check the actual source-to-output mapping, not just the row counts.
    source_by_id = {r["DOC_ID"]: r for r in recent_rows}
    assert len(source_by_id) == len(recent_rows), "Duplicate document IDs"
    assert len(documents) + len(exclusions) == len(recent_rows)
    assert len({d["DOC_ID"] for d in documents}) == len(documents)
    assert len(pilot_ids) == len(pilot)
    assert len(pilot_roots) == len(set(pilot_roots))
    for d in documents:
        assert all(d[k] == source_by_id[d["DOC_ID"]][k] for k in raw_fields)
        assert d["api_root_10"] not in roots
        assert d["filename_id_query"] == d["DOC_ID"]
    for root, docs in selected.items():
        assert 1 <= len(docs) <= 3
    total_pilot_bytes = sum(int(d["FILE_SIZE"] or 0) for d in pilot)
    assert total_pilot_bytes < 25_000_000_000
    summary = {
        "snapshot_date_utc": str(snapshot), "counts": dict(counts),
        "reference_counts_match": counts["gulf_mud_tiffs"] == 15923 and counts["gulf_recent_deep_mud_tiffs"] == 959,
        "recent_deep_areas": dict(collections.Counter(r["AREA"].strip() for r in recent_rows)),
        "eligible_documents": len(documents), "excluded_documents": len(exclusions),
        "ranked_api_roots": len(wells), "eligible_bytes": sum(int(d["FILE_SIZE"] or 0) for d in documents),
        "pilot_wells": len(pilot_roots), "pilot_documents": len(pilot), "pilot_bytes": total_pilot_bytes,
        "pilot_areas": dict(collections.Counter(grouped[root][0]["AREA"].strip() for root in pilot_roots)),
        "local_annotation_names_checked": len(names), "local_api_roots_checked": len(roots),
        "inventory_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_to_output_check": "PASS: all raw fields match the source; IDs are unique; sizes are within the request limit",
        "evaluations_run": 0, "images_read": 0, "model_weights_used": 0,
    }
    (NOTES / "checks.json").write_text(json.dumps(summary, indent=2) + "\n")
    (DATA / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("Top wells:")
    for w in wells[:12]:
        print(w["well_rank"], w["api_root_10"], w["AREA"], w["BLOCK"], w["WELL_NAME"], w["best_metadata_score"], w["best_DOC_ID"], w["best_DOC_TYPE"])


if __name__ == "__main__":
    main()
