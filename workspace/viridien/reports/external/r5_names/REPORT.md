# Filename source check

Date: 8 November 2026. Lane: r5_names.

The strongest source is the BSEE well-log release system, or a library made
from the same operator submissions. The evidence is strong for a shared
source. It does not prove that Viridien or CGG took the files directly from BSEE.
No exact challenge image was downloaded or checked by hash.

The main new fact is the **Mudlog ZIP route**. The earlier report counted
TIFF images. The same public inventory also has **3,319 released Gulf Mudlog
ZIP records**. The named modern Gulf files map to 34 ZIP records at 30 API12
values. API, lease, area, block and well suffix supply the match. The ZIP
member names and contents remain unchecked.

| API12 in BSEE | Lease | Area/block, well and suffix | Mudlog ZIP DOC_ID |
| --- | --- | --- | --- |
| 608114058000 | G16786 | GC 738 / 001 ST00BP00 | sam-7017131 |
| 608114060501 | G31765 | GC 896 / 001 ST00BP01 | sam-6710441 |
| 608184006500 | G08035 | AT 574 / SC003 ST00BP00 | sam-6458717 |
| 608114053501 | G20084 | GC 653 / SB203 ST00BP01 | sam-6539882 |
| 608084001400 | G21451 | KC 964 / 001 ST00BP00 | sam-6520392 |
| 608174114200 | G26265 | MC 540 / 001 ST00BP00 | sam-6571541 |
| 608174143300 | G12166 | MC 765 / P011 ST00BP00 | sam-7059461 |
| 608054009701 | G36102 | AC 336 / 001 ST00BP01 | sam-7072629 |

Full rows are in [challenge_api_mud_metadata.csv](challenge_api_mud_metadata.csv).
This file has 45 released mud-log records at 31 named or identified APIs,
including D-582. The inventory uses API12. Compare it with the first 12 digits
of the named API14. Keep the well suffix. Do not discard sidetracks.

**D-582 is a legacy document code, not a separate geographic log family.**
The exact public DOC_ID is `0000000009{D-582}0000020435-0001`.
It is a 1-inch Mud Log / Formation Evaluation Log, API12 `608114024601`,
lease **G11075**, **GC block 562**, well **001 ST01BP00**. Its run date is
22 September 1999. Its release date is 20 November 2001. The listed file has
5,367,890 bytes, barcode `gc005960`, source `QC_DATA`, and CD volume `QCCD0122`.

BSEE defines DOC_ID as a unique sequential electronic-document ID.
Thus `0000020435-0001` is part of that ID. It is not an API number.
Sixteen D-582 catalog records, numbered 20424 through 20439, share this well
and CD volume. D-582 likely marks a legacy scan/index group. This last point
is an inference: no decoder for each ID part was found. The leading number,
the `-0001` suffix, and the added challenge `logNN` suffixes are not decoded.
See [d582_inventory.json](d582_inventory.json). No D-582 image was opened.

The filenames without APIs also give useful exclusions. MC765 P011 maps to
API10 root **6081741433**, lease G12166. Shell AC0336 G36102 maps to root
**6080540097**. Shell G08852 KE001 maps to root **6081741395**. These roots
are in [challenge_name_exclusions.json](challenge_name_exclusions.json),
along with D-582 and all named API roots. The at0574 Neptune filename has
the right API and block, but BSEE names the well SC003; its `4` is unresolved.

After these 24 known well-root exclusions, the ZIP catalog has **3,265 records**,
**2,905 API12 values**, and **1,947 API10 groups**. Listed size is
**271,887,467,812 bytes**. These are catalog counts, not confirmed matching
images or independent logs. Unknown aliases and other unlisted overlaps can
remain. [non_challenge_zip_metadata.csv](non_challenge_zip_metadata.csv) keeps
the raw fields. The TIFF count agrees with the copied report: **15,923 released
Gulf mud-log TIFF records**. The 959 recent deep-water TIFF candidates remain
metadata candidates with unknown image fit.

The [public inventory](https://www.data.bsee.gov/Other/Files/FRSWellDataRawData.zip)
download worked. SHA-256:
`ce1f8d5f4c7e3514a0f02cb3588a4cd9a83ca3a378b81cb1effd6c0d9dcef20a`.
The [Custom Set Builder](https://www.data.bsee.gov/Other/FileRequestSystem/Authenticated/CustomSetBuilder.aspx)
redirected to Login.aspx. BSEE states that requests are free, limited to 25 GB,
can take 24 hours, and have a five-day download window. No sign-in or request
was made. Anonymous image access and matching image volume remain unknown.

Three independent EDX PDFs are available without login. Each has a visible
lithology pattern column and a printed legend on page 1. Each scores **2/3**
for shared log features. None is a proven offshore Gulf layout match or part
of the challenge release. They cover three wells and **74 PDF pages**.

| Candidate | Pages | Score | Local PDF |
| --- | ---: | ---: | --- |
| MPC 10-4, 5 inch | 15 | 2 | [evidence/mpc10_4_5in.pdf](evidence/mpc10_4_5in.pdf) |
| MPC 34-1, 1 inch | 4 | 2 | [evidence/mpc34_1_1in.pdf](evidence/mpc34_1_1in.pdf) |
| MRCSP-FEGENCO 1 | 55 | 2 | [evidence/fegenco1.pdf](evidence/fegenco1.pdf) |

See the [labelled contact sheet](previews/contact_sheet.png).
[preview_manifest.json](preview_manifest.json) gives the source URL, page,
PDF-point box and crop path for each column and legend. [triage.json](triage.json)
gives source URLs, verified hashes, PDF producers and access limits. The MPC
files are vector logs from Well Data Services. FEGENCO is a Stratagraph NE log
with vector content and a raster logo. Only page 1 was checked by sight.

Route rank uses similarity × verified well count × access without login.
EDX scores **2 × 3 × 1 = 6**. The BSEE route has unknown similarity and matching
volume, so its numeric score is unknown. Login blocked the tested image
request route. Texas and the other routes also have unknown product scores.
Do not use catalog record counts as image match counts.

Exact names, APIs and leases were sent to Google, Bing, DuckDuckGo and Yahoo.
Brave, Mojeek, Ecosia and Yandex were also checked. Google gave a JavaScript
shell; Bing gave unrelated hits; other engines gave CAPTCHA, temporary-error,
403 or 429 responses. GitHub code search needed authentication; one repository
query gave 504. Kaggle and Hugging Face catalog queries gave no exact image
match. Zenodo gave 403. The current data.gov exact API query gave zero records;
its old CKAN API gave 404. NETL exact API, lease, MC765, Rio, Critter Creek and
Callaway queries gave zero results. Its broader mudlog query found the EDX PDFs.
Archive.org had no relevant log hit. Wayback API and D-582 URL queries were
empty; the Rio query gave 503. These are search limits, not proof of absence.

No source was confirmed for the Texas files. The GLO tract-ID claim for 3744,
1748 and 2829 remains unverified. The copied RRC report already records a
403 image-query error. This lane read the GLO home and Energy Resource
Library; it found no exact log record. Callaway and Critter Creek have no
confirmed state or archive source from filename evidence alone.

All actual response times, status codes, redirects, sizes and download hashes
are in [access.jsonl](access.jsonl). Unknowns are explicit in triage.json.
External files were read as untrusted data. Python ran with `-I`; requests used
a generic User-Agent and at least 1.05 seconds between starts per host,
including redirects. Rendering used one thread. No downloaded code ran.
No eval image, eval truth or sealed well was opened. No training, tuning,
benchmark, conversion admission, commit, service, sign-in or child agent was
used. Exact source matches are provenance evidence and are excluded.
