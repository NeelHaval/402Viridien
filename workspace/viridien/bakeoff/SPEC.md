# Lithology bake-off: shared spec (read fully before writing code)

Goal: compare 10 strategies that turn a scanned mud-log into lithology intervals, on equal terms:
same loader, same depth step, same scorer, same visualiser. Accuracy AND efficiency are judged.

## Paths
- Dataset (READ ONLY, never write there):
  `C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset`
  - `images/<name>.tif` or `.tiff` (51 annotated; ignore `...md(1).tif` and `Thumbs.db`)
  - `bbox_annotation/<name>.json`: `bounding_boxes` with `depth_column`, `interpreted_lithology_column`,
    `lithology_legend`, each `[x0, y0, x1, y1]` pixels, x1/y1 exclusive.
  - Example notebook (has a reference visualiser): `.../hack27/project/example_notebook.ipynb`
- Bake-off root (all code and outputs): `C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work/bakeoff`
  - `harness/` shared package (python package, has `__init__.py`)
  - `strategies/sNN_<name>.py` one file per strategy
  - `gt/gold/<log>.csv` hand-checked truth (log06 is the supplied example), `gt/silver/<labeler>/<log>.csv`, `gt/synthetic/`
  - `eval_set.json` evaluation windows; `out/<strategy>/...` outputs; `report/` dashboard
- Python: `python` (3.12, has numpy, Pillow, opencv `cv2`, scikit-learn, scikit-image, ruptures, hmmlearn,
  torch 2.12 + CUDA (RTX 4080 Laptop 12 GB), transformers, timm, open_clip, rapidocr_onnxruntime, easyocr, psutil, pandas, matplotlib).
  HF weights cached: `facebook/dinov2-small`, `facebook/dinov2-base`, `google/siglip-base-patch16-224`.
- Run everything from the bake-off root as modules: `python -m harness.run ...`.
- Images are huge (up to 1700 x 313,800 px). `Image.MAX_IMAGE_PIXELS = None`. Never convert a whole page to RGB:
  crop the box first, read in strips of <= 4000 rows. 1-bit ('1') and palette ('P') images must be converted per crop.

## Interval rows (the one output format)
CSV columns, in order: `top_y,bottom_y,lithology,cluster_id,top_depth,bottom_depth`
- `top_y`/`bottom_y`: full-image pixel rows, int, bottom exclusive. Rows sorted, non-overlapping.
- Blank (unfilled) stretches of the column produce NO row. Adjacent rows with the same lithology should be merged
  unless a boundary line is really drawn between them.
- `lithology`: the legend label text exactly as the harness legend parser returns it (`entry.label`), or
  `"Unknown"`. `cluster_id`: int, same id = same visual pattern within this log.
- `top_depth`/`bottom_depth`: from the harness depth calibration, rounded to 0.1.

## Harness API (stage A builds this; strategies only call it)
```python
from harness import io, legend, depth, score, viz, synth
io.list_logs() -> list[str]                       # 51 annotated log names (stem without extension)
io.open_log(name) -> Log                         # Log.image (lazy PIL), Log.boxes dict, Log.name, Log.mode, Log.size
io.read_rgb(log, box, y0=None, y1=None, scale=1.0) -> np.ndarray HxWx3 uint8   # strip-safe crop of box rows y0..y1
legend.parse(log) -> list[LegendEntry]            # cached to out/_cache/legend/<log>.json + swatch PNGs
#   LegendEntry: id:int, label:str, swatch_box:[x0,y0,x1,y1] (full-image px), swatch:np.ndarray RGB (inner area,
#   border removed), mean_rgb, is_lithology:bool (False for modifiers / show & drilling / fossil / accessory symbols)
depth.calibrate(log, engine="rapidocr") -> Calib  # cached; Calib.to_depth(y)->float, .a, .b, .n_points, .rmse_px, .method
score.score(pred_rows, gt_rows, window=(y0,y1)) -> dict   # metrics below
viz.render(log, rows, out_png, gt_rows=None, window=None, stages=None, legend_entries=None, title="")
synth.make(log, n=1, height=4000, seed=0) -> list[(rgb_array, gt_rows)]   # columns stacked from this log's own swatches
```
Strategy file contract (`strategies/sNN_<name>.py`):
```python
NAME = "S1 colour runs"; NEEDS_GPU = False; PAID_API = None   # or "jev" / "codex-terra"
def run(log, window, ctx) -> list[dict] | (list[dict], dict_of_stage_images)
#   log: io.Log; window: (y0, y1) rows to label (full-image coords, inside the litho column box)
#   ctx: dict with "legend" (list[LegendEntry]), "calib" (Calib), "device" ("cuda"/"cpu"), "cache_dir"
#   return rows (dicts with the CSV keys; depths may be omitted, the runner fills them from calib)
#   optional stage images (np RGB arrays) for the pipeline figure, keys in order, e.g.
#   {"1 column": ..., "2 features": ..., "3 boundaries": ..., "4 clusters": ..., "5 legend match": ...}
```
Rules for strategies: never read `gt/`; no per-log hand-tuned constants or log-name special cases; may cache models
under `out/_cache/`; budget <= 60 s per evaluation window on this machine (GPU allowed). Paid API calls must be
counted in the returned stage dict key `"_api_calls"` (int).

## Evaluation windows (`eval_set.json`, already written; do not change)
A list of `{log, y0, y1, tier}`. For log06 (gold) the whole lithology box. For other logs: a 6000-row window (or
the rest of the box if shorter) starting at the first row where the lithology column is mostly filled.
Logs in the eval set (13): the gold log06 plus
`0000000009_d-582_0000020435-0001_log02_fe_5in_md`, `0000000009_d-582_0000020435-0001_log04_fe_5in_md`,
`60807402860000_gb506#2_md_mudlog`, `60811405120000_1inch tvd mudlog`,
`60811407300100_bp_gc743_15607_dc222_st00bp01_mudlog_1md_10700-22872`,
`608114075800 - Allegheny Attic GC297 #3 OCS-G 08876 003 ST00 BP00  Mud Log 3346ft-13925ft MD`,
`60817411400100_mc761_1_st1_mudlog_1md_17133_22886`, `60818400650000_at0574_4_neptune_1inch_tvdmudlog_6300-17378`,
`Critter Creek 201-1807H_Mud_PDF_Vertical.pdf`, `MC765_P011_ST00BP00_1inchMD_Depth_MudLog_3686-20707ft`,
`Rio_Nueces_3744_Unit_1_212_MUD_5IN`, `Shell_08852_KE001_ST01BP00_1inTVD_MudLog_17564-20124ft`.

## Metrics (`score.score`)
Inside the window only, ignoring rows that are blank in GT:
- `row_acc`: fraction of GT-filled pixel rows whose predicted lithology equals GT (case/space-insensitive).
- `boundary_f1`: boundaries = interval tops/bottoms strictly inside the window; match tolerance
  `max(4, min(20, 0.01 * window height))` px, one-to-one greedy.
- `cluster_ari`: adjusted Rand index between predicted cluster_id and GT lithology, per pixel row (sampled every 2 rows).
- `depth_mae_ft`: mean |pred depth - GT depth| over matched boundaries when GT has depths.
- `score = row_acc * boundary_f1`.
Runner adds: `seconds`, `peak_rss_gb`, `peak_gpu_gb`, `api_calls`.

## Runner (`python -m harness.run --strategy s01 [--set eval|full|synthetic] [--gt gold|silver/<labeler>|consensus]`)
- eval: every window in eval_set.json; writes `out/<strategy>/<log>.csv`, `<log>.png` (viz.render with GT and stages),
  `metrics.json` (per log + mean). full: all 51 logs, window = whole litho box, timing only + CSV + PNG.
  synthetic: 3 synthetic columns per eval log, scored against their exact GT.
- Each log runs in a try/except: a crash is recorded as `error` with score 0, never stops the run.
- Timing: wall time of `run()` only (legend parse and depth calibration are shared and timed separately once).

## Visualiser (`viz.render`) must show, in one PNG
left: page thumbnail with the three boxes; middle: the window of the litho column at readable scale with
predicted intervals painted on the left half (colour per lithology, label text), GT on a parallel strip, and an
error strip (green = right row, red = wrong row, grey = blank in GT); boundary ticks with depths; right: the legend
crop with each swatch outlined, the ones used in the prediction highlighted. If `stages` is given, a row of the stage
images underneath ("pipeline view"), each titled. Flat style, white background, no gradients.
