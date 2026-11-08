# Literature lane result

Date: 8 November 2026. Lane: r5_literature.

**No independent full Gulf mud-log page was verified.** No source has a
confirmed similarity of 2 or 3. No file was converted or used for training.

The useful result is a direct test of the BSEE source hypothesis.
All **26 API groups** found in the supplied challenge names have public
BSEE inventory records with type `Mudlog`. There are **29 matching records**.
All 29 are **ZIP packages**, not TIFF files. The earlier TIFF-only list did
not cover this package route.

These are metadata matches. They do not prove an exact file copy or show
how Viridien/CGG obtained the files. No challenge image was opened.
Some catalogue well names differ from the names in the challenge files.
For example, Hadrian's record has well name `001`; the filename says
`hadrian 2`. Keep that limit when you use the match.

| Name in the challenge file | BSEE API | Lease | Well and suffix | Document ID |
| --- | --- | --- | --- | --- |
| Mad Dog, MDN-1 | 608114058000 | G16786 | 001 ST00BP00 | sam-7017131 |
| Ardennes | 608114060501 | G31765 | 001 ST00BP01 | sam-6710441 |
| Shenzi | 608114053501 | G20084 | SB203 ST00BP01 | sam-6539882; sam-6540189 |
| GC743, DC141 | 608114041000 | G15607 | DC141 ST00BP00 | sam-6883820 |
| Allegheny Attic | 608114075800 | G08876 | 003 ST00BP00 | sam-7080051 |
| Neptune | 608184006500 | G08035 | SC003 ST00BP00 | sam-6458717 |

The API join uses the first 12 digits in a challenge name. The inventory
uses 12 digits. The normal challenge form adds two final zeroes.
See [all exact API matches](source/provenance_matches.json) and the
[source rows](source/challenge_metadata_matches.csv).

The [public BSEE inventory](https://www.data.bsee.gov/Other/Files/FRSWellDataRawData.zip)
downloaded without login. Its SHA-256 is
`ce1f8d5f4c7e3514a0f02cb3588a4cd9a83ca3a378b81cb1effd6c0d9dcef20a`.
Its size is 49,438,793 bytes. The local copy is
`source/FRSWellDataRawData.zip`.

There are **3,319 released Gulf mud-log ZIP packages**, with a listed total
of **278,390,510,531 bytes**. Excluding the **20 known challenge API roots**
leaves **3,276 packages**, **1,950 roots**, and **273,551,320,708 bytes**.
These are package counts. Page count, layout, overlap and useful volume
are unknown. Do not treat the remainder as cleared training data.
The same strict title rule reproduces the prior **15,923 TIFF** count.
See [count rules and totals](source/inventory_counts.json).

The [Well Data query page](https://www.data.bsee.gov/Other/FileRequestSystem/WellData.aspx)
is public. The supplied earlier report says image requests need Login.gov.
No image request or sign-in was made. No public ZIP download link was found.
All challenge-associated records and their API roots stay excluded.

Six valid PDFs downloaded without login. They contain **671 pages**.
They are search evidence, not accepted mud-log candidates.

| Downloaded file | PDF pages | Similarity | Result |
| --- | ---: | ---: | --- |
| [CSB Macondo volume 1](documents/csb_macondo_vol1.pdf) | 37 | 0 | Real court-exhibit references; no full formation log confirmed |
| [CSB Macondo volume 2](documents/csb_macondo_vol2.pdf) | 112 | 0 | BOP and barrier report; no full formation log confirmed |
| [CSB Macondo volume 3](documents/csb_macondo_vol3.pdf) | 247 | 0 | Human factors report; no full formation log confirmed |
| [BSEE TAP 777](documents/bsee_tap777.pdf) | 22 | 0 | No mud-log or lithology text hit |
| [BSEE TAP 765](documents/bsee_tap765.pdf) | 201 | 0 | No mud-log or lithology text hit |
| [Ohio State AC810 thesis](documents/osu_hydrate_thesis.pdf) | 52 | 0 | Page 42 has gamma ray and resistivity, without a lithology column |

The [labelled rejection preview](previews/osu_page42_labelled.png) shows
the Ohio State figure. Its [evidence record](previews/osu_page42_evidence.json)
has the source URL, page and figure box. No positive lithology or legend
region was found. Other PDFs were searched as text; they were not checked
page by page by sight. Their unconfirmed column and legend fields are null.

OpenAlex gave real Gulf thesis records and exact repository links. The UT
Mississippi Canyon mass-transport thesis, DOI `10.26153/tsw/10314`, is a
useful lead, but its record returned HTTP 403. WKU, USM, Texas Tech and UNO
PDF links also returned 403. LSU search and AAPG Datapages search returned
403. Rice returned an empty 202 response; Texas A&M returned 403; Tulane
had a DNS error. BEG describes paid TIFF orders at USD 10 per log. SEG
returned a script challenge, OnePetro returned 403, and GCAGS returned 406.
No challenge code was run.

For Macondo, CSB reports cite exact MDL 2179 exhibit URLs. The current court
host had a certificate error on HTTPS and a script gate on HTTP. The cited
USCG route returned 403. The cited BP PDF returned 403; the Transocean PDF
returned 404; the archived BSEE report returned 429. NOAA had a DNS error.
Seven BSEE MC252 packages were found in metadata. They cover several APIs;
block 252 alone does not identify all seven as the accident well.

Rank uses similarity × useful volume × access without login. The checked
PDF routes have score 0. The other routes remain unranked because one or
more inputs are unknown. No positive rank can be justified from labels.

[triage.json](triage.json) gives source URLs, access times, document hashes,
page counts and ranking inputs. [access.jsonl](access.jsonl) keeps each
request and each error. HTML error responses with a `.pdf` filename are
marked as invalid in [document_checks.json](document_checks.json).

No held-out truth, evaluation image or sealed well was inspected or rebuilt.
No benchmark, model, training, tuning, sign-in, contact, submission, service,
child agent or commit was used. All writes stayed in this lane. Download
and parsing scripts ran with Python `-I`. Rendering used one CPU thread.
No downloaded file was executed. Reuse rights remain unknown.
