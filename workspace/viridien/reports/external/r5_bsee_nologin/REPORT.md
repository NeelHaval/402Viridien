# Public Gulf log-image routes

Checked: 8 November 2026. Lane: r5_bsee_nologin.

The strongest public image route is **USGS Data Series 760**. Four TIFFs
from three Gulf wells were downloaded without login. All four have a
lithology column. Two have clear lithology legends. Each TIFF has **one
long page**. They are old black-and-white scans, with **similarity 2/3**.
No score of 3 was confirmed.

| Candidate | Local file, relative to this folder | Exact source URL |
| --- | --- | --- |
| OCG-S-2490 mud log; clear four-symbol legend | [sources/gulf_2490_mud.tif](sources/gulf_2490_mud.tif) | [TIFF](https://pubs.usgs.gov/ds/760/Downloads/OS/OFFSHORE/OCG-S-2490_MUDLOG_200_17500.tif) |
| OCG-S-2523 geologist log; clear ten-symbol legend | [sources/gulf_2523_geo.tif](sources/gulf_2523_geo.tif) | [TIFF](https://pubs.usgs.gov/ds/760/Downloads/OS/OFFSHORE/OCG-S-2523_GEOLOGIST_1240_17388.tif) |
| OCG-S-2523 mud log; damaged header legend | [sources/gulf_2523_mud.tif](sources/gulf_2523_mud.tif) | [TIFF](https://pubs.usgs.gov/ds/760/Downloads/OS/OFFSHORE/OCG-S-2523_MUDLOG_321_17388.tif) |
| OCG-S-2486 mud log; no legend confirmed | [sources/gulf_2486_mud.tif](sources/gulf_2486_mud.tif) | [TIFF](https://pubs.usgs.gov/ds/760/Downloads/OS/OFFSHORE/OCG-S-2486_MUDLOG_500_17800.tif) |

See the [labelled contact sheet](previews/labelled_contact_sheet.png).
[image_evidence.json](image_evidence.json) gives each source URL, hash,
page, and lithology/legend box. Boxes use original TIFF pixels, with zero
at the top left. Header and body previews are in `previews/`.
The two 2523 files belong to one well. Do not count them as independent wells.

The [USGS catalog](https://pubs.usgs.gov/publication/ds760) has 40 files
from 15 eastern Gulf wells. A join of the published file and well lists
gives five MUDLOG and eight GEOLOGIST records from seven wells. These
13 labels are leads, not 13 checked images. The other files include
wireline logs and LAS data. The well list names **BOEM** as the source
for the two 2523 files. It names USGS for the sampled 2486 and 2490 files.
Keep the catalog spelling `OCG-S`; several printed headers say `OCS-G`.
The other 36 Gulf file downloads were not tested.

The route rank uses similarity × checked distinct-well count × public
image access. USGS scores **2 × 3 × 1 = 6**. Unknown image volume or
access stays unknown. Catalog counts do not raise an image-match score.

| Route | Verified result |
| --- | --- |
| [BSEE scanned well files](https://www.data.bsee.gov/Other/FileRequestSystem/ScanWellFiles.aspx) | Actual public raster PDFs. The raw index has 58,844 Gulf well-file records. One [37-page PDF](sources/bsee_pdf_sample.pdf) was downloaded. It is a drilling/procedure packet; no lithology log was seen in its page thumbnails. Useful image-log volume is unknown. |
| [BSEE FRS](https://www.data.bsee.gov/Other/FileRequestSystem/Default.aspx) and BOEM Well Data | Public inventory, not public image bytes. Custom Set Builder redirected to the login page. The current inventory has 15,923 Gulf mud-log TIFF records and 3,319 standard ZIP records, plus four records in other extensions. Layout is unknown. |
| BOEM G&G / paleo / sand atlas / NAMSS | Public summary, permit, reservoir and seismic routes. Paleo sends original operator reports to FRS. No separate mud-log image archive was confirmed. A linked `biochart.pdf` returned HTTP 200 error HTML, not a PDF. |
| NETL EDX Gulf collections | The checked 646-row petrophysical ZIP contains interpretations and GIS files, not source raster logs. Legacy-well resources are XLSX files. Two JIP research PDFs downloaded, with 29 and 196 pages; no target lithology column was confirmed. |
| Old BSEE DiscMediaStore / MMS FTP / DVD / Wayback | BOEM still links Well Logs to DiscMediaStore. That link redirects to an error page. CDX lists 2017–2019 HTML captures. A 2019 replay returned 429. No working anonymous FTP or DVD image-download route was established. |
| Other public mirrors | EDX's old MMS record describes digital curves and links to BOEM home. Archive.org search returned zero items. The tested Data.gov catalog API returned 404. No further bulk image mirror was verified. |

The route details, exact checked URLs, counts, and limits are in
[triage.json](triage.json). [access.jsonl](access.jsonl) records UTC request
times, status codes, errors, sizes, and SHA-256 hashes. Broad archive checks
had a 503 response and two timeouts. Main USGS search and agency search
returned 403. General search engines gave challenges, errors, or unrelated
results. Thus, this work does not prove that no other historical mirror exists.

The filename source hypothesis has stronger **metadata support**, but no
exact challenge copy was obtained. All 20 numeric Gulf API roots in the
supplied names match Mudlog ZIP records. There are 43 records at those
roots; 29 match the supplied full API after 14-to-12-digit normalization.
Examples are `sam-6710441` for API `608114060501`, lease `G31765`, and
`sam-7017131` for API `608114058000`, lease `G16786`.
MC 765 P011 ST00BP00/BP01 match `sam-7059461` and `sam-7059462`.
AC 336, G36102, well 001 ST00BP01 matches `sam-7072629`.
The inventory does not expose the original filenames inside these ZIPs.
It does not prove an exact copy or a transfer to CGG/Viridien.

[challenge_overlap_exclusions.json](challenge_overlap_exclusions.json)
holds the 49 source IDs found by these broad API and named-well checks.
No related package was downloaded. Exact challenge copies must stay out
of conversion and training. The new USGS candidates have different old
lease/well identities. No comparison with evaluation images or sealed
wells was made. Public release alone does not establish all reuse rights
for industry artwork; several scans retain old property stamps.

Use the [ten-minute owner action guide](OWNER_ACTION_GUIDE.md) to start
the existing pilot. Its 60 IDs total 1,490,473,041 bytes. The copied CSV
and new text-ID file preserve the earlier lane's selection. FRS confirms
text-list upload after sign-in, a 25 GB request limit, preparation of up
to 24 hours, and a five-day link after activation. Ten minutes is an
owner work plan, not a promised processing time. Count limits and exact
authenticated profile fields remain unknown. No sign-in or submission
was made. The completed lane's ranked list was not changed.

All new files stay in this lane folder. Download and parsing scripts used
Python `-I`. The downloader used a generic User-Agent and at least one
second between requests per host, including redirects. Rendering used
one CPU thread. No downloaded code was run. No OCR, child agent, service,
email transmission, training, tuning, benchmark, commit, or push was used.
No evaluation truth, challenge image, model weight, or sealed well was opened.
The raw files and previews remain untrusted source evidence only.
