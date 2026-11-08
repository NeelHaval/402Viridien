"""EXPERIMENT ONLY (not adopted; it changes results): time features + boundaries + naming on the 188k scan with the lithology
column at its native resolution vs resized to a fixed 4 px per foot.

    python scripts/experiment_px_per_ft.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2, numpy as np, pandas as pd
from app.config import Settings
from app.pipeline.stages.boundaries import find_beds
from app.pipeline.stages.features import compute_features
from app.pipeline.stages.legend import read_legend
from app.pipeline.stages.load_column import load_column
from app.pipeline.stages.naming import name_layers
from app.pipeline.model import load_model
from app.tools.preprocess import preprocess, finish_calibration
from app.tools.read_upload import read_upload
from scripts.bench_e2e import SCANS, DATASET

load_model()
cfg = Settings()
name = SCANS["mdn1_188k"]
bbox = json.loads((DATASET / "bbox_annotation" / (Path(name).stem + ".json")).read_text())
base = pd.read_csv(ROOT.parents[1] / "perf" / "baseline" / "mdn1_188k" / "corrected.csv")
ft_per_px = (base.base_depth.max() - base.top_depth.min()) / (base.bottom_y.max() - base.top_y.min())
scan = read_upload(str(DATASET / "images" / name))
layer1 = preprocess(scan, bbox, with_location=False, calibrate=False)
legend = read_legend(scan, layer1.legend_box)
column = load_column(layer1)
print(f"native: {column.rgb.shape[0]} rows, {1 / ft_per_px:.1f} px/ft")
for label, scale in (("native", 1.0), ("4 px/ft", 4 * ft_per_px)):
    rgb = column.rgb if scale == 1.0 else cv2.resize(column.rgb, None, fx=1.0, fy=scale, interpolation=cv2.INTER_AREA)
    col = type(column)(np.ascontiguousarray(rgb), 0)
    t = time.perf_counter(); feats = compute_features(col, cfg.stripe_height); t_f = time.perf_counter() - t
    t = time.perf_counter(); beds = find_beds(col, feats, window=cfg.boundary_window, min_gap=cfg.boundary_min_gap, ink_k=cfg.boundary_ink_k); t_b = time.perf_counter() - t
    t = time.perf_counter(); layers = name_layers(beds, col, legend, split_window=cfg.split_window, split_min_run=cfg.split_min_run); t_n = time.perf_counter() - t
    print(f"{label:8s} rows={rgb.shape[0]:7d} features={t_f:.2f}s boundaries={t_b:.2f}s naming={t_n:.2f}s total={t_f + t_b + t_n:.2f}s layers={len(layers)}")
