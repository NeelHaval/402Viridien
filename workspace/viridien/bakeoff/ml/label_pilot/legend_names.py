"""Read names with the shared parser. Keep its cache inside this lane."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "3"
from prepare import HOME, BAKEOFF, LOGS, require_non_eval, sha256

cli = argparse.ArgumentParser(description=__doc__)
cli.add_argument("--refresh", action="store_true", help="Read the two legends again")
args = cli.parse_args()

sys.path.insert(0, str(BAKEOFF))
from harness import io, legend
import cv2

cv2.setNumThreads(3)
io.ROOT = HOME / "parser_cache"
result = {}
for key, name in LOGS.items():
    require_non_eval(name)
    if args.refresh:
        (io.ROOT / "out" / "_cache" / "legend" / (name + ".json")).unlink(missing_ok=True)
    log = io.open_log(name)
    try:
        entries = legend.parse(log)
        result[key] = [dict(id=e.id, label=e.label, is_lithology=e.is_lithology,
                            swatch_box=e.swatch_box) for e in entries]
        print(key, [(e.id, e.label) for e in entries if e.is_lithology], flush=True)
    finally:
        log.close()
(HOME / "legend_names.json").write_text(json.dumps(result, indent=2) + "\n")
(HOME / "parser_provenance.json").write_text(json.dumps(dict(
    module="harness.legend", version=legend.VERSION,
    module_sha256=sha256(Path(legend.__file__)),
    purpose="Read printed names only. No rock class was assigned by OCR.",
    cache="parser_cache/out/_cache/legend",
    threads=dict(omp=3, mkl=3, openblas=3, opencv=3, rapidocr_intra=2, rapidocr_inter=2),
    inherited_lithology_weights=False,
    ocr_pretraining_overlap="Not audited"), indent=2) + "\n")
