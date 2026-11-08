# Review brief: lithology-column labelling pipeline (read-only review)

You are a critical reviewer. Be very critical. Find real gaps, not polish. Every claim must cite a file and line,
a data file, or a number you computed yourself. Say plainly when you could not check something.

## The task being solved
Viridien HackTheBurgh "Decoding the Subsurface". 51 scanned mud logs (TIFF), each with bounding boxes for
`depth_column`, `interpreted_lithology_column`, `lithology_legend`. Goal: segment the lithology column into
depth intervals and label each with an entry of that log's own legend. Output: CSV of intervals + annotated image.
Judged on correctness and efficiency. Spec: `bakeoff/SPEC.md`.

Dataset (READ ONLY, never write there):
`../viridien_challenge/Viridien_HTB_challenge/hack27/project/dataset` (`images/`, `bbox_annotation/`).

## Current pipeline (v4)
- `bakeoff/labeltool/legend_classify.py`: parses the legend into swatches per label, filters non-rock entries
  (`NOT_ROCK` regex), merges identical-picture "twins" (e.g. "Clay / Shale"), and classifies a column window by
  nearest legend swatch. Distance = colour dE/10 + texture + orient term + gamma*(1-NCC). Hand-set parameter dict `P`.
  `window_state` rejects windows as mixed/blank/text by hand thresholds.
- `bakeoff/labeltool/beds.py`: v4. Finds bed boundaries from row profiles (CIELAB fill, paper share, ink share,
  band comparison, hand thresholds in `B`), tiles each bed into ~48-row chunks, classifies chunks, majority vote,
  merges thin beds. Output `bakeoff/analysis/categories/beds/<log>.json`.
- `bakeoff/notes/conf_data_v4.py`: confidence = logistic regression on [bed margin, agree fraction, family one-hot]
  fitted on blind-LLM windows the LLM was "very sure" about (from the v3 review), times agree fraction.
  Writes `conf.json`, `beds_csv/<log>.csv`, `classified_v4/`, `compare_v3_v4.json`.
- `bakeoff/notes/llm_review.py`: blind LLM review. One sheet per call = one log's numbered legend (2x) + up to 8
  column crops; classifier answer hidden; LLM returns pick / second / sure (or mixed/none). 10% duplicate crops
  on other sheets for self-agreement. Bands by classifier confidence. `notes/llm_summary.py` summarises.
- `bakeoff/notes/eval_blind.py`: scores classifier variants against the blind LLM picks (this is the hill-climb).
- Review UI: `bakeoff/labeltool/famreview.py` + `static/confreview.html`, `static/mixreview.html`.
- Older work: `bakeoff/strategies/`, `bakeoff/ml/` (boundary, legend_v4, synth_gulf, synth_v2, ...),
  `bakeoff/synth/` (synthetic generator + `CRITIQUE.md`), `bakeoff/harness/`, `bakeoff/eval_set.json`.

## Ground truth that exists
- `bakeoff/gt/gold/` 1 log (log06, the organiser's example). `bakeoff/gt/human/` 1 log (D-582 log02) + `labels.jsonl`.
- `bakeoff/gt/silver/A`, `B`: model-labelled. `neel_labels/`, `neel_dataset/` (teammate human labels / adjudication).
- Family list per log: `analysis/label_shares_per_log.csv` (columns log, family, label, pct).
- External: `external_data/kgs`, `research/` (earlier literature notes, `research/SYNTHESIS.md`).

## Measured so far ("LLM agrees" = the blind LLM picked the classifier's label)
- Baseline progression on LLM-judged windows: v1 35%, v2 55%, v3 65%.
- v3 blind review (`analysis/categories/llmrev/`, 130 sheets, 770 crops): LLM self-agreement 57/70 (81%).
  Agreement by classifier confidence: <50% 45%, 50-80% 62%, 95%+ 85%; when LLM is very sure 61/77/99%.
  By family: D-582 8%, One-off 62%, Geoservices 63%, Texas 74%, Gulf 81%, MC765 98%, MC761 100%.
- v4 on the same 583 blind windows: 66% -> 71%. D-582 7->19%, Geoservices 65->72%, Gulf 85->86%,
  Texas 85->91%, One-off 66->81%, MC765 98->100%, MC761 93->100%.
- Note: the v4 confidence model is fitted on the same v3 blind windows that this comparison reports.
- D-582 (9 logs, black-and-white 1-bit): classifier says "Silty Sandstone" on long two-row dashes, LLM says
  Clay/Shale; in that legend Clay and Shale are identical pictures.
- A fresh blind check of v4 on new windows is running (`analysis/categories/llmrev4/`, may be incomplete).

## The owner's plans (critique and improve these too)
1. Synthetic datasets (generated columns from legend swatches; see `bakeoff/synth/`).
2. Outside datasets similar to this one.
3. Some human-labelled data (small: a few logs).
4. A hill-climbing loop: change classifier -> score on blind LLM picks -> keep if better.

## Rules
- Read only. Do not write anywhere. Do not run anything that writes into the dataset folder.
- No network calls, no API keys. Do not open any `.env` file.
- Python available: `python` 3.12 with numpy, opencv, scikit-learn, scikit-image. Running small read-only
  analysis scripts on the JSON/CSV outputs is fine and encouraged (print to stdout only).
