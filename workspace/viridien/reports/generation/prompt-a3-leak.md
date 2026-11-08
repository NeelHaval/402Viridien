You are one of 11 independent critics of a fake-pattern generator and synthetic well-log data pipeline. Your job: find what is wrong or weak in YOUR LENS (below) and propose concrete fixes. You are a reviewer, not an implementer.

Working folder: C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_work\pattern_lab (Windows, `python -X utf8`, NumPy, Pillow).

## Rules
- **Write exactly one file:** your report at `data/genlab/critique/a3-leak.md`. Scratch images and scripts go in `data/genlab/critique/a3-leak/` only. Edit nothing else; other agents are building code in genlab/ right now.
- **Keep RAM low:** under 2 GB. ~20 other jobs share this machine. Never load a whole TIFF page; use `learn.Crops(model.Images())` or windowed reads.
- **Evidence over opinion:**
  - Run code and measure.
  - Render and LOOK at images: real crop vs generated, at the same scale.
  - Cite file:line.
  - Each finding needs a concrete failure scenario or a measured number. Drop anything you cannot support.
- **Test wells are held out:** see data/normalized/split-policy.json. Flag any place where test-well data could leak into generation or training.

## Context (read what your lens needs)
- **Goal:** realistic fake lithology patterns and end-to-end synthetic well logs (columns and full pages with exact ground truth), to train and test lithology-column detectors/classifiers for 51 scanned well-log TIFFs. Each log has a legend of fill patterns and an interpreted lithology column.
- **Pattern generator lab:**
  - code in genlab/ (README.md, learn.py, swatch.py, candidates.py, build.py, render.py, score.py, server.py);
  - live at http://127.0.0.1:8781 (state.json has `cards` with tile, samples, real boxes, tile_scores, realism, source, verdict);
  - generated images in data/genlab/generated/<doc>/.
- **Agent-made patterns:**
  - Codex makers and reviewers produced data/genlab/proposals/<doc>/ (legend-<id>.json, tile.png, sheets, legend-review-r<k>.json).
  - The live board at http://127.0.0.1:8782/board.json shows every entry's state and last note. 142 training entries are still Failed, mostly because "the swatch shows one course, so the vertical period can't be measured".
- **Owner labels:** data/normalized/current.json (1,028 boxes, growing). Machine guesses: ../bakeoff/labeltool/data/prefill.jsonl (~71% right on the fully labelled Hadrian log 5709fe9e86ce942b).
- **Fake dataset:** genlab/dataset.py, outputs in data/genlab/datasets/ (v1/eval.md: fake→real recognition is 4–11% top-1 open-set; the augmented real-vs-fake discriminator is 63%). Job gen1108-dseval2 is diagnosing that right now and adding a closed-set within-document evaluation; see its progress in data/genlab/datasets/diag/ if present.
- **Synthetic assembler** (jobs spec: data/genlab/jobs/asm-contract.md, asm-stats.md, asm-compose.md, asm-bench.md):
  - genlab/assemble/stats.py → data/genlab/assemble/stats.json and stats-report.md (done);
  - compose.py and noise.py → data/genlab/assemble/out/ (still being built);
  - bench.py → suites/ and report.html (still being built).
  - Review what exists, and note what is unfinished rather than calling it wrong.
- **Owner's analysis tool (read-only):** http://127.0.0.1:8771/api/mix, /api/cat, /api/legend?v=2.

## Report format (data/genlab/critique/a3-leak.md)
1. **Verdict:** 3 lines. Is the work in your lens fit for its purpose? What is the single biggest problem?
2. **Findings,** ranked by impact. Each has:
   - a title;
   - severity: blocker, major or minor;
   - evidence (numbers, file:line, image path you looked at);
   - the fix: what to change, where, and how to verify it;
   - effort: S, M or L;
   - expected gain.
3. **Quick wins:** at most 5 things under an hour each.
4. **What is already good:** 1–3 lines, so nobody breaks it.
5. **Doubts:** what you could not check.

Reply with the same content as the file.

## YOUR LENS (a3-leak)
Leakage and data hygiene across every module: test wells, validation wells, owner labels arriving over time, machine guesses, the 8771 analysis (which covers all 51 logs including test), augmentation pools, cross-log identical symbols. Find every path by which held-out data could shape generation, statistics or model selection, and every place where results would silently go stale as current.json grows. Run checks.
