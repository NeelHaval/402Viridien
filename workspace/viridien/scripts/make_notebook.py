"""Build and execute viridien_work/lithology_explainer.ipynb (all figures embedded at full resolution)."""
import sys, pathlib, nbformat as nbf
from nbclient import NotebookClient
W = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(W / "scripts"))
import build_artifact as B  # reuses the strategy cards (S) and measured notes (MEAS)

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))
def imgs(*paths, width=None):
    w = f", width={width}" if width else ""
    code("\n".join(f'display(Image(filename=r"{p}"{w}))' for p in paths))

A, F, H = "artifact/img", "figs", "figs/hard"
md("""
# Decoding the Subsurface: lithology segmentation and clustering

**Viridien challenge, HackTheBurgh 2026.** One notebook that explains the challenge, the dataset, what makes it hard,
how we score answers, eleven strategies with code and real demo images, their trade-offs, the research behind the
next upgrades, and the live bake-off results.

All images are embedded at full resolution, so the notebook reads without running anything. Run it from
`viridien_work/` to refresh the results section.
""")
code("""
import json, glob, os, pandas as pd
from IPython.display import Image, Markdown, display
pd.set_option("display.max_colwidth", 80)
os.getcwd()
""")

md("""
## 1. The task

For every scanned mud log, output one CSV of **contiguous intervals** down the interpreted lithology column:

`top_depth, bottom_depth, lithology, top_y, bottom_y, cluster_id`

plus an annotated image. `lithology` is a name from the log's own legend, and `cluster_id` groups intervals with the same
visual pattern. Judged on **correctness** and **efficiency**. Bonus: find the depth column, the lithology column and the
legend without the supplied boxes.

Every log goes through four steps:

| Step | What it does |
|---|---|
| 1. Split | find each interval's top and bottom pixel row |
| 2. Group | same pattern, same `cluster_id` |
| 3. Name | match each group to a legend swatch |
| 4. Depth | read the depth column and fit pixel row to feet |

The only true answer supplied is for log06 (17 intervals):
""")
code("""
gold = pd.read_csv("bakeoff/gt/gold/0000000009_d-582_0000020435-0001_log06_fe_5in_md.csv")
gold.head(10)
""")
md("What it looks like drawn with the organisers' `visualize_results` (full page at 0.1 scale, then a zoom at 0.5):")
imgs(f"{A}/f01_notebook_full.png", f"{A}/f02_notebook_zoom.png")

md("""
## 2. The dataset, and how varied it is

51 annotated TIFF logs, 7.3 gigapixels in total. Every size varies by 10x to 100x:
""")
code("""
d = pd.read_csv("figs/dataset_stats.csv")
d["aspect"] = (d.height / d.width).round(1)
d["col_pct_of_page"] = (100 * d.col_w / d.width).round(1)
print("image modes:", d["mode"].value_counts().to_dict(), "(1 = 1-bit black and white, P = palette)")
d[["width", "height", "col_w", "col_h", "legend_w", "legend_h", "aspect", "col_pct_of_page"]].describe().loc[["min", "50%", "max"]].round(1)
""")
md("""
| Quantity | Range |
|---|---|
| page height | 2,600 to 313,800 px (median 44,550): 120x |
| lithology column width | 29 to 297 px, only 2 to 6 % of the page |
| depth scale | 0.05 to 0.5 ft per px across the 13 test logs: 10x |
| rock entries per legend | 2 to 30, plus up to 20 non-rock symbols |
| image modes | 36 RGB, 9 1-bit, 6 palette |

**About eight legend families, not 51 random logs** (counts marked ~ come from file names and the legend sheet):

| Family | Logs | Rock column | What breaks |
|---|---|---|---|
| D-582 well, "LITHOLOGY SYMBOLS" grid | 9 | 1-bit hatches only | colour methods see nothing; Shale, Clay and Silty Sandstone are all dash patterns |
| Gulf of Mexico 3-panel (Lithology / Modifiers / Show & drilling) | ~20 | flat colour, often one colour for thousands of rows | parser must drop the modifier and drilling panels; tiny text |
| Geoservices "Lithology Legend" grid | ~4 | colour plus pattern | several sandstones share yellow and differ only in dot density |
| Mineral / Lithology split (MC761) | 3 | colour fills | a mineral list sits beside the rock list |
| MC765 colour-only grid | 4 | colour fills | same well at 1 inch and 5 inch, MD and TVD |
| Texas onshore "Rock Types" box | ~7 | coloured stripes with symbols | labels a few pixels high |
| One-offs (Allegheny, Trail Mountain, Freeman, Shell) | ~4 | bold colour plus hatch | layouts seen once |

**The same word means different pictures.** Sandstone is a dotted white box in D-582, plain yellow in the Gulf logs,
yellow with dots in Geoservices and a striped band in Texas. So every log must be matched against *its own* legend.

**Near-duplicates.** D-582 appears 9 times, MC765 4 times, MC761 3 times. Anything tuned on log06 will look better on
log02 and log04 than it really is, so results are reported for D-582 and the rest separately.
""")
md("### All 51 legends")
imgs(f"{F}/01_all_legends.png")
md("### The depth and lithology columns, 3,000 px from 40 % down each log")
imgs(f"{F}/02_columns_1.png", f"{F}/02_columns_2.png")
md("### Whole pages (contact sheets)")
imgs(*[f"{F}/03_pages_{i}.png" for i in range(1, 6)])

md("""
## 3. The inputs: three boxes per page

Each annotation gives `[x0, y0, x1, y1]` (max exclusive) for **1 depth_column (red)**,
**2 interpreted_lithology_column (blue)** and **3 lithology_legend (orange)**, drawn here like the notebook's `draw_boxes`.
""")
code("""
m = json.load(open("artifact/manifest.json", encoding="utf8"))
pd.DataFrame([{"figure": k, "page": v["page"], **v["boxes"]} for k, v in m.items() if k.startswith("f03")])
""")
for i in range(1, 7):
    imgs(f"{A}/f03_boxes_{i}.png", width=700)

md("## 4. What makes it hard")
for n, c in [("h1_pattern_only", "Pattern-only rocks: 1-bit logs have no colour at all."),
             ("h2_mixed_legend", "Legends mix rocks with modifier and drilling symbols."),
             ("h3_same_colour", "Same colour, different rock."),
             ("h4_tiny_text", "Legend text a few pixels tall."),
             ("h5_blank_runs", "Blank stretches and very long single intervals."),
             ("h6_overprints", "Overprinted lines and symbols break runs.")]:
    md(f"**{c}**")
    imgs(f"{H}/{n}.jpg")

md("""
## 5. Ground truth and metrics

Three tiers of truth:

| Tier | What | Use |
|---|---|---|
| Gold | log06, supplied | final scoring |
| Silver | two AI labellers (A, B) on 13 test windows | aggregate scores only, never per-log tuning |
| Synthetic | columns stacked from each log's own legend swatches | tuning, exact labels |

Metrics (inside the window, ignoring rows blank in the truth):

* **row_acc**: share of filled rows whose predicted lithology matches.
* **boundary_f1**: interval edges matched one-to-one within `max(4, min(20, 1 % of window))` px.
* **cluster_ari**: adjusted Rand index of `cluster_id` against true lithology.
* **depth_mae_ft**: depth error at matched boundaries.
* **score = row_acc x boundary_f1**, plus seconds, peak RAM, peak GPU memory and API calls.

### How good are AI agents at making datasets?

Both labellers also labelled log06, where we know the truth:
""")
code("""
import runpy, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    cwd = os.getcwd(); runpy.run_path(os.path.abspath("scripts/lab_vs_gold.py"), run_name="__main__"); os.chdir(cwd)
print(buf.getvalue())
""")
md("""
They fail in opposite ways. **A** names rocks well (74 % of rows right) but misplaces edges. **B** finds edges well
(boundary F1 0.94) but left 67 % of rows as Unknown. Agents find boundaries well and name black-and-white hatches badly,
so silver labels need a person's check.
""")
imgs(f"{A}/f04_labeller_vs_gold.png")

md("""
## 6. Shared building blocks

### Legend parser (step zero for every strategy)

Swatches are found as rectangles, paired with OCR labels, and entries under "Modifiers" or "Show and drilling" are marked
as not rock. It found every swatch on the 7 pages checked by eye; remaining slips are OCR spellings such as "Anhydrlte",
"Coarsesandstone" and "SHALEGRAY", so names are matched fuzzily.
""")
imgs(*[f"{A}/f05_legend_parse_{i}.png" for i in range(1, 4)])
md("""
### Depth calibration

RapidOCR reads the numbers in the depth column, then RANSAC fits `depth = a * y + b`.

| Log | ft per px | anchor points | fit error (px) | seconds |
|---|---:|---:|---:|---:|
| log06 | 0.0499 | 12 | 0.68 | 9.2 |
| GB506#2 | 0.4993 | 35 | 0.50 | 15.8 |
| BP GC743 | 0.4999 | 62 | 0.47 | 33.3 |
| Allegheny | 0.4995 | 31 | 2.17 | 18.7 |
| Neptune | 0.3329 | 39 | 1.33 | 21.8 |
| Critter Creek | 0.1333 | 161 | 0.60 | 48.4 |
| Rio Nueces | 0.1000 | 119 | 1.11 | 41.6 |

On log06 row 4409 maps to 25,400.27 ft against the supplied 25,400 ft.
""")

md("""
## 7. The strategies, and why each is in the race

All strategies share the loader, legend parser, depth step and scorer, so a score difference comes from the strategy alone.
Each must implement `run(log, window, ctx) -> rows (+ stage images)`. The sketches below show the idea; the full code is in
`bakeoff/strategies/`. Demo images are real runs on real crops.
""")
for sid, title, cls, hw, img, what, cd, why, pro, con in B.S:
    md(f"""
### {sid}: {title}  ({hw})

{what}

**Why it is here:** {why}

| Strength | Weakness |
|---|---|
| {pro} | {con} |

```python
{cd}
```
{("**Measured:** " + B.MEAS[img].split(":", 1)[1].strip()) if img in B.MEAS else ""}
""")
    if img and (W / A / img).exists():
        imgs(f"{A}/{img}")
md("""
### S11: FFT hatch + Lab swatch matching, with an Unknown option  (CPU)

Added from the research (rank 2). Plain template matching scored only 0.04 to 0.08 correlation on log06 because the
column's hatch is shifted and rescaled relative to the swatch. The 2-D FFT magnitude ignores where the pattern starts,
so comparing pooled spectra (plus ink fraction and colour) is phase-invariant; a small top-2 margin returns "Unknown".

```python
def spectrum(grey):
    f = np.abs(np.fft.fftshift(np.fft.fft2((grey - grey.mean()) * hann2d(grey.shape))))
    return pool_radial_angular(f, r_bins=12, a_bins=8, scale=dominant_period(f))
dist = [w_fft * l1(spectrum(win), spectrum(sw)) + w_ink * abs(ink(win) - ink(sw))
        + w_lab * lab_hist_dist(win, sw) for sw in rock_swatches]
name = label[argmin(median(dist over windows))] if margin > tau else "Unknown"
```

### B1 (bonus): find the boxes without the annotation

```python
ans = jev(assets={"page": thumb_png},
          questions={"legend": {"type": "box", "instructions": "the lithology legend panel"}})
coarse = to_pixels(ans["legend"]["box"], page_size)
x0, x1 = snap_to_ruled_lines(coarse, vertical_lines(page))
```
A first test on 3 pages: JEV alone found the legend with overlap 0.47 to 0.73 but mostly missed the columns, which are a
few pixels wide on a 1,024 px thumbnail. Snapping to ruled lines at native width does the rest.

**JEV details learned the hard way:** set the header `TypeSafe-Danger-Zone-Images: true`; images go in `assets` and are
named from `state`; each side must be 512 to 1024 px; sent as base64 text inside `state`, the image is read as text.
""")

md("""
## 8. Trade-offs at a glance

| Strategy | Hardware | Speed | Cost per log | Best on | Weak on |
|---|---|---|---|---|---|
| S1 colour runs | CPU | 0.27 s measured | 0 | flat colour logs | 1-bit logs, same-colour rocks |
| S2 features + PELT | CPU | 3.6 s measured | 0 | boundaries everywhere | naming near-identical patterns |
| S3 template matching | CPU | 0.7 s measured | 0 | exact-scale swatches | scale or phase mismatch |
| S4 k-means + Viterbi | CPU | 3.9 s measured | 0 | noisy scans | picking k, thin beds |
| S5 DINOv2 | GPU | 3.1 s measured | 0 | texture changes | tiny swatches |
| S6 SigLIP | GPU | 2 to 10 s est. | 0 | clear rock names | hatch-only logs |
| S7 synthetic CNN | GPU | about 30 s est. | 0 | names limited to this legend | slowest local option |
| S8 JEV | API | 0.33 s a call measured | one call per cluster | naming from a picture | confident misses |
| S9 Terra | API | tens of s a call est. | ChatGPT plan | reads like a person | slow, varies |
| S10 hybrid | GPU + API | 5 to 15 s est. | calls on ties only | accuracy per second | most moving parts |
| S11 FFT matching | CPU | seconds est. | 0 | shifted hatches | very small swatches |

Measured times are from demo windows of 2,600 to 6,000 rows; section 10 has the bake-off's own numbers.
""")

syn = (W / "research" / "SYNTHESIS.md").read_text(encoding="utf8")
syn = syn.replace("# Mud-log pipeline synthesis", "").replace("\n## ", "\n### ")
md("""
## 9. Research: how to do each step better, cheaper and faster

Eight research agents (one per pipeline step) searched papers, GitHub and model cards; their notes are in `research/r*.md`.
A ninth merged them, spot-checked the key claims against their sources, and ranked the upgrades. Notes: the depth RANSAC
fit (its rank 1) is already in the shared harness, and none of the sources tested anything on mud logs, so the bake-off
decides. The merged report follows verbatim.
""" + "\n" + syn)

md("""
## 10. Live bake-off results

This cell reads whatever the strategy runs have written so far (`bakeoff/out/<strategy>/metrics.json`). Rerun it to refresh.
""")
code("""
rows = []
for p in sorted(glob.glob("bakeoff/out/*/metrics.json")):
    if "_dummy" in p: continue
    m = json.load(open(p, encoding="utf8"))
    logs = m.get("logs", {})
    items = logs.items() if isinstance(logs, dict) else [(x.get("log"), x) for x in logs]
    for log, r in items:
        if not isinstance(r, dict): continue
        rows.append({"strategy": m.get("strategy"), "set": m.get("set"), "gt": m.get("gt"), "log": str(log)[:40],
                     **{k: r.get(k) for k in ("score", "row_acc", "boundary_f1", "cluster_ari", "seconds", "api_calls", "error")}})
res = pd.DataFrame(rows)
if res.empty:
    print("No strategy metrics yet.")
else:
    display(res[res["gt"] == "gold"].sort_values("score", ascending=False).round(3) if "gt" in res else res)
    display(res.groupby(["strategy", "set", "gt"])[["score", "row_acc", "boundary_f1", "cluster_ari", "seconds"]].mean().round(3))
""")
md("Each strategy's pipeline figure on log06 (page with boxes, prediction vs truth vs error strip, legend, and its stages):")
code("""
for p in sorted(glob.glob("bakeoff/out/s*/0000000009_d-582_0000020435-0001_log06_fe_5in_md.png")):
    display(Markdown(f"**{p.split(os.sep)[-2] if os.sep in p else p.split('/')[-2]}**"))
    display(Image(filename=p))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(W)}}).execute()
out = W / "lithology_explainer.ipynb"
nbf.write(nb, out)
errs = [c for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
n_img = sum(1 for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
print(out, round(out.stat().st_size / 1e6, 1), "MB;", len(nb.cells), "cells;", n_img, "embedded images;", len(errs), "errors")
