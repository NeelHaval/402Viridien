# Legend visual audit

All seven required debug PNGs were opened with the image viewer after the final
detector pass. Debug images and numbered swatch crops are under
`out/_cache/legend/`; JSON files contain the returned labels, flags, and OCR boxes.
Green outlines denote lithology entries; red outlines denote excluded entries.

| Log (abbreviated) | Found / visible swatches | Lithology entries | Visual label audit |
|---|---:|---:|---|
| D-582 log06 | 16 / 16 | 10 | Six empty cells excluded. OCR returns `Anhydrlte` for Anhydrite and `SIllty Sandstone` for Silty Sandstone. `LImestone` has harmless case variation. |
| 60811407300100_bp_gc743... | 30 / 30 | 30 | `Coarsesandstone` loses its space. Other labels agree visually; source itself prints `Oolic limestone`. |
| 60807402860000_gb506#2_md_mudlog | 25 / 25 | 12 | No substantive label errors seen. Thirteen technical, show, fossil, and bedding symbols excluded. Plastic Clay and Diatomite retained. |
| Critter Creek 201... | 23 / 23 | 23 | `SHALEGRAY` loses its space. Compact coal and unbordered chert, marlstone, shale and breccia cells recovered. The separate question-mark UNKNOWN key is not a lithology swatch. |
| Rio_Nueces_3744_Unit_1_212_MUD_5IN | 23 / 23 | 23 | `SIDERITEorLIMONITE` loses spaces. Separate question-mark UNKNOWN key omitted. No substantive lithology label errors seen. |
| 608114075800 - Allegheny... | 33 / 33 | 30 | OCR `Gypsum, Anhyd.` uses comma instead of slash; `Claystone Sililicif.` has a spelling error. Pyrite, Glauconite and Foraminifera symbols excluded. `Methamorphics` is the source's printed spelling. |
| 60817411400100_mc761... | 26 / 26 | 11 | No substantive label errors seen. All 13 entries under Mineral excluded, plus Not Available and No Sample. |

Counts include empty framed cells for log06 and technical framed cells for the
other logs. They exclude title frames and the separate UNKNOWN question mark in
the Rock Types panels. Some inferred unbordered cells have a small amount of
surrounding white margin. OCR spellings are preserved as the strategy API labels;
no per-log corrections or ground-truth-derived labels were applied.
