Date: 8 November 2026. Lane: r5_hydrate_science.

**Best result: seven public JIP Leg II LWD interpretation logs, similarity 2.**
All seven files downloaded without sign-in. Each file has one long PDF page.
The total size is 31,832,080 bytes. Each checked crop shows a mineral-volume
column and a mineral legend. No score-3 match or true cuttings mud/master
log was confirmed.

These are Schlumberger EcoView Solutions plots for Chevron science wells.
They show clay, Q-F-M and other mineral volumes beside curves and borehole
images. The legend lists possible minerals. It does not show which minerals
are present in each well. These columns need a separate class review before
any later conversion. They are not cuttings descriptions.

All paths below start at `external_untrusted/round5/r5_hydrate_science/`.
All seven files have score **2** and **1 PDF page**.

| Hole | Local file | Exact source URL |
| --- | --- | --- |
| AC21-A | [documents/jip2_ac21a_ecoscope.pdf](documents/jip2_ac21a_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/AC21-A/field_log_plots/AC21-A_Ecoview_Solutions.pdf) |
| AC21-B | [documents/jip2_ac21b_ecoscope.pdf](documents/jip2_ac21b_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/AC21-B/field_log_plots/AC21-B_Ecoview_Solutions.pdf) |
| GC955-H | [documents/jip2_gc955h_ecoscope.pdf](documents/jip2_gc955h_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/GC955-H/field_log_plots/GC955-H_Ecoview_Solutions.pdf) |
| GC955-I | [documents/jip2_gc955i_ecoscope.pdf](documents/jip2_gc955i_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/GC955-I/field_log_plots/GC955-I_Ecoview_Solutions.pdf) |
| GC955-Q | [documents/jip2_gc955q_ecoscope.pdf](documents/jip2_gc955q_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/GC955-Q/field_log_plots/GC955-Q_Ecoview_Solutions.pdf) |
| WR313-G | [documents/jip2_wr313g_ecoscope.pdf](documents/jip2_wr313g_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/WR313-G/field_log_plots/WR313-G_Ecoview_Solution.pdf) |
| WR313-H | [documents/jip2_wr313h_ecoscope.pdf](documents/jip2_wr313h_ecoscope.pdf) | [PDF](https://mlp.ldeo.columbia.edu/data/ghp/JIP2/WR313-H/field_log_plots/WR313-H_Ecoview_Solutions.pdf) |

Use the [labelled contact sheet](previews/ecoscope_contact_sheet.jpg) for a
quick check. [triage.json](triage.json) has source links, file hashes, page
sizes, document types and score inputs. [preview_manifest.json](preview_manifest.json)
has source URLs, PDF page numbers and crop boxes. Coordinates use PDF points
from the top left. Each EcoView page is 612 points wide and 5,328 to 14,112
points tall. Do not treat a long page as one ordinary paper sheet.

The rank uses similarity × confirmed matching holes × public access.
Access is 1 only when a file download passed. Unknown inputs stay unknown.

| Route | Similarity | Confirmed matching holes | Product |
| --- | ---: | ---: | ---: |
| JIP Leg II EcoView | 2 | 7 | 14 |
| DSDP Leg 96 | 1 | 6 | 6 |
| UT-GOM2-2 core summaries | 1 | 2 | 2 |
| UT-GOM2-1 core figure | 1 | 1 | 1 |
| JIP Leg I matching log route | unknown | unknown | unknown |

Secondary material is kept separate. UT-GOM2-2 Appendix D has 47 total
pages. Pages 3–46 are 44 core sheets for WR313 Holes H003 and H002. Page 38
states no recovery. Thus 43 sheets describe recovered material. These are
core summaries, score 1. They are from **WR313**, not GC955. Appendix F has
19 physical-property log sheets in a 21-page PDF; it adds no confirmed
lithology-column pages. The Methods chapter identifies Strater software.
Its use of “master log” refers to sediment-strength spreadsheets.

The UT-GOM2-1 GC955 report has one confirmed facies figure on PDF page 43
of 138. It covers core H005-4FB. It is score 1. The separate full
lithostratigraphy PDF timed out. DSDP Leg 96 adds six gamma/core comparison
pages for six holes, plus 11 core-sheet pages for Hole 621. Hole 621 occurs
in both files. The explanatory notes have a lithology key on PDF page 7.
These are score 1 sources, not mud logs.

IODP 308 LogDB lists seven logged holes. Three are at Ursa: U1322A, U1323A
and U1324A. The saved U1324A LWD summary has curves and borehole images,
with no drawn lithology column in the checked header and data crop. It is
score 0. NETL JIP reports are useful context, but add no confirmed matching
pages. EDX returned exactly one resource for its JIP project record: a
62-page final report. That record is not a bulk log archive.

Access limits are in [access.jsonl](access.jsonl), with UTC times and errors.
The old UT UDC host timed out twice at 20 seconds. IODP publication pages
and the checked Zenodo pages/API returned 403. Brave returned 429.
The first Appendix D/E transfers exceeded 20 MB. Appendix D was then saved
with a 28 MB bound after a size check. Appendix E was not saved. Its HEAD
size is 29,788,733 bytes. No archive of 1.4 GB was downloaded.

The BSEE/CGG source hypothesis remains **unconfirmed**. The copied BSEE
report supports a submission library, but gives no exact challenge-copy
proof here. Only the 51 names in the supplied CSV were used for exclusions.
The seven EcoView header API roots have no overlap with those names.
The headers contain errors or reused IDs: AC21-A/B share one printed API,
and GC955-I has an unusual printed API. The saved header sheet and
[duplicate_gate.json](duplicate_gate.json) keep this evidence. No ID was
repaired. Name/API checks cannot rule out every renamed copy. Keep these
files outside conversion and training until the later duplicate check.

No explicit training or reuse license was established for the LogDB plots.
EDX states “Other (Open)”; no specific grant was verified. Public download
alone does not establish third-party reuse rights.

All downloads remain untrusted data. Python scripts ran with `-I`.
Rendering used one process and one CPU thread limit. No OCR ran.
No downloaded code was executed. No challenge image, truth or sealed well
source was opened.
No conversion, training, tuning, benchmark, sign-in, message, submission,
commit, service or child agent was used. All new files are in this folder.
