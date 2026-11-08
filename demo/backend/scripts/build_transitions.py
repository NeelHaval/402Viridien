"""Build data/reference/transitions.json from labelling-tool JSON files: python scripts/build_transitions.py LABELS_DIR."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validation.config import COARSE_CLASSES, SEQ_LAPLACE_ALPHA, coarse_class  # noqa: E402

OUT_FILE = ROOT / "data" / "reference" / "transitions.json"


def layer_classes(label: dict) -> list[str]:
    """Coarse class of each layer (top to bottom) using the legend's printed names."""
    names = {entry["id"]: entry.get("label_printed", "") for entry in label.get("legend", [])}
    layers = sorted(label.get("layers", []), key=lambda ly: ly.get("top_px", 0))
    out = []
    for layer in layers:
        comp = layer.get("composition") or []
        if comp and comp[0].get("legend_id") in names:
            out.append(coarse_class(names[comp[0]["legend_id"]]))
    return out


def count_transitions(folder: Path) -> tuple[dict[str, dict[str, int]], int, int]:
    """Count consecutive-bed class transitions across every label file under a folder."""
    counts = {a: {b: 0 for b in COARSE_CLASSES} for a in COARSE_CLASSES}
    files = used = 0
    for path in sorted(folder.rglob("*.json")):
        try:
            label = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(label, dict) or "layers" not in label:
            continue
        files += 1
        seq = layer_classes(label)
        used += max(len(seq) - 1, 0)
        for a, b in zip(seq, seq[1:]):
            counts[a][b] += 1
    return counts, files, used


def smooth(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    """Laplace-smoothed row-normalised probabilities."""
    k = len(COARSE_CLASSES)
    probs = {}
    for a, row in counts.items():
        total = sum(row.values()) + SEQ_LAPLACE_ALPHA * k
        probs[a] = {b: (n + SEQ_LAPLACE_ALPHA) / total for b, n in row.items()}
    return probs


def main(argv: list[str]) -> int:
    """Write transitions.json for the given labels folder."""
    if len(argv) != 2 or not Path(argv[1]).is_dir():
        print(__doc__)
        return 2
    counts, files, used = count_transitions(Path(argv[1]))
    if used == 0:
        print("no transitions found; is this the labelling-tool JSON folder?")
        return 1
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"classes": list(COARSE_CLASSES), "counts": counts, "probs": smooth(counts),
                                    "n_files": files, "n_transitions": used}, indent=1))
    print(f"wrote {OUT_FILE} from {files} files, {used} transitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
