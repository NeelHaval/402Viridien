# Challenge requirements and failure modes

Source: all eight slides in `hack27/HTB_Challenge_Slides.pptx`, the team bake-off SPEC, and human/external dataset documentation at source commit `663721e72e3fa2ba496987c13b66610e52d1c8f3` in `silmoon04/hacktheburgh26`. The original spelling is Viridien; the repository name follows the requested `viridian402`.

The required stages are interval boundaries, recurring visual-pattern groups, top/base depth estimates, and names matched to each document's legend. Automatically locating the columns and legend is a bonus. Supplied bounding boxes are valid normal inputs. The brief requires automatic final predictions, a CSV and an annotated lithology image, and judges correctness and efficiency on unseen images.

| Requirement | Pipeline support |
|---|---|
| Individual intervals | Sorted, disjoint, positive-height full-image pixel intervals |
| Pattern clustering | Per-document cluster IDs and permutation-invariant ARI |
| Depth | Explicit feet/datum metadata, missing-depth rate and conditional matched-boundary MAE |
| Legend names | End-to-end model input versus explicitly supplied-legend diagnostic mode |
| CSV | Internal `bottom_depth`; submission `base_depth` matches the slides |
| Annotated image | Bounded column preview with boundaries and labels; inspect full-resolution crops for thin beds |
| Efficiency | Per-case subprocess wall time, sampled process-tree peak RSS, hard timeouts |
| Unseen/generalisation | Group separation, provenance, per-source/per-well results and strict paired comparisons |

Important cases to keep visible:

- **Huge TIFFs.** Scans reach hundreds of thousands of rows. A Pillow crop can decode the whole TIFF. The bounded reader checks native decode memory and crop size, uses optional strip decoding when supported, and fails explicitly rather than promising every codec streams. Keep evaluation windows modest. The legacy S11 adapter has a conservative whole-page byte guard.
- **Full-image coordinates.** Bounding-box and interval maxima are exclusive. Add crop offsets once. Resizing a crop must not change the reported original pixel coordinates. Test `.tif` and `.tiff`, bilevel, palette, grayscale and RGB sources.
- **Very thin beds.** Minimum-gap rules and smoothing can erase real 6–25 pixel layers. Broad colour changes, printed rules, symbols and hatching can create false boundaries. Inspect boundary precision and recall separately.
- **Blank regions and partial labels.** Blank or unlabelled truth gaps do not become invented “blank rock” labels. Scoring windows must be inside declared truth coverage. Coverage and empty-prediction counts remain visible even when boundary F1 is 1 on a one-bed window.
- **Legend mapping.** A legend entry ID is local to a page. Equal colours may represent different hatch patterns, and the same rock can have different symbols on other pages. Multi-panel Gulf layouts, tiny OCR text, merged labels, grey fills, modifiers, drilling symbols, fossils, gas and oil shows need distinct tests.
- **Depth.** MD and TVD cannot be silently compared. Metric printed depths need explicit conversion. External Flat Bay pages print metres, but the per-page source manifest says truth was already converted to feet. We retain that truth unchanged and record the conflicting older README. Sparse OCR, broken scales and non-linear depth tracks must be rejected or declared unknown.
- **Leakage.** Several pages and resolutions belong to one well. Synthetic children inherit their parent's group/split. Labels, pseudo-labels, organizer examples and prior model tuning all count as exposure. Visible shared verification is diagnostic only.
- **Synthetic optimism.** Tiled clean swatches do not reproduce scan noise, overprints, skew or difficult legends by themselves. External born-digital logs have clean rendering and different family coverage; they do not establish Gulf Coast scan accuracy.

The bundled S11 is a frozen comparison model, with known failures on empty legends and Shenzi. The small default baseline verifies the interface and provides boundary/clustering signals; it does not claim a competitive automatic lithology recognizer. Improvements belong behind the same model interface and must be measured on the same manifests.
