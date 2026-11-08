import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "gt" / "silver" / "B"
OUT.mkdir(parents=True, exist_ok=True)
eval_rows = json.loads((BASE / "eval_set.json").read_text(encoding="utf8"))
for row in eval_rows:
    name = row["log"]
    if not (name.startswith("0000000009") or name.startswith("60818400650000")):
        continue
    with (OUT / f"{name}.csv").open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["top_y", "bottom_y", "lithology"])
        if name.startswith("0000000009"):
            w.writerow([row["y0"], row["y1"], "Unknown"])
