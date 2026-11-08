# Public vendor mud-log samples

Date: 8 November 2026. Lane: r5_vendor_samples.

Selman has the best verified full-page example. No public vendor route yielded
a verified full deepwater Gulf mud log. No source received similarity 3.
All files stay in this folder as untrusted source data.

| Order | Source and score | Verified volume | Type and limit |
| --- | --- | --- | --- |
| 1 | Selman, 2/3 | One 4-page PDF | Fictional Texas log. Header and legend, two full body pages, one short body continuation. Clear cuttings patterns, curves and rock text. |
| 2, visual reference | WellSight, 2/3 | One 919 x 1033 image | Ellerslie software screenshot. Patterned lithology and rock text. No legend or full log page. Gulf location is not shown. |
| 3, visual reference | Geoservices / SLB, 1/3 | One 12-page PDF | Log figures on pages 6 and 11. Native figures are only 722 x 428 and 446 x 272 pixels. Page 11 is Malaysia. |
| 4, visual reference | Sperry / Halliburton, 1/3 | One 2-page PDF | Small lithology-classification figure and legend on page 1. No full mud log. |
| 5, Gulf context | Baker Hughes, 1/3 | One 1-page PDF | Text confirms an ultra-deepwater Gulf case at over 32,000 ft. Figures show analysis, not a cuttings log. |

The route score is similarity x verified full-log example count x access without
login. Selman has 2 x 1 x 1 = 2. The other downloaded routes have zero verified
full logs. This is a confirmed-yield score. Total volume at each vendor is unknown.
The visual-reference order above does not imply available full log volume.
All listed downloads returned HTTP 200 without login.

## Downloads

| Source | Local file | Exact source URL |
| --- | --- | --- |
| Selman | [documents/Selman_Example_Mudlog_B.pdf](documents/Selman_Example_Mudlog_B.pdf) | https://mud-log.com/wp-content/uploads/2026/07/Selman_Example_Mudlog_B.pdf |
| WellSight | [documents/Ellerslie-ML.png](documents/Ellerslie-ML.png) | https://wellsight.com/wp-content/uploads/Ellerslie-ML.png |
| Geoservices | [documents/flairflex-br.pdf](documents/flairflex-br.pdf) | https://www.slb.com/-/media/files/drilling/brochure/flairflex-br.pdf |
| Halliburton | [documents/LithoFact-Service-Datasheet.pdf](documents/LithoFact-Service-Datasheet.pdf) | https://cdn.brandfolder.io/VUJJLY3X/as/bjmnn54c3jbv6rph26557mqh/LithoFact-Service-Datasheet.pdf |
| Baker Hughes | [documents/HC-Vision-identified-best-production-zones-GoM-cs.pdf](documents/HC-Vision-identified-best-production-zones-GoM-cs.pdf) | https://www.bakerhughes.com/sites/bakerhughes/files/2020-07/HC-Vision-identified-best-production-zones-GoM-cs.pdf |

See [the contact sheet](previews/contact_sheet.jpg). In [triage.json](triage.json),
each confirmed lithology column or legend has its source URL, page, box and crop.
PDF boxes use points from the top left. Image boxes use original pixels.
The boxes support later visual review. They are not interval truth.
SHA-256 hashes, bytes, PDF producer and page counts are in triage.json.
The other four saved documents are a guide, service text, or a photograph.

## BSEE source hypothesis

The public BSEE inventory has **30 matching records for all 26 supplied API
prefixes**. For a 14-digit challenge API, use the first 12 digits to join the
catalog. Keep the ST/BP suffix. Do not remove the leading 60. One supplied
Allegheny name already has a 12-digit API.

| Challenge metadata | BSEE record | Exact catalog agreement |
| --- | --- | --- |
| 60811405800000, G16786 | sam-7017131 | API 608114058000, lease G16786, well 001, ST00BP00, Mudlog ZIP |
| Ardennes 60811406050100, G31765 | sam-6710441 | API 608114060501, lease G31765, well 001, ST00BP01, Mudlog ZIP |
| Shenzi 60811405350100, SB203 | sam-6539882; sam-6540189 | API 608114053501, well SB203, ST00BP01, Mudlog ZIPs |
| 60812400480000, G26419 | US60812400480000_FORMEVAL_45821 | API 608124004800, lease G26419, well 001, ST00BP00, 5in Formation Evaluation (Baker Hughes Atlas), TIFF |

There are 29 ZIP records and one TIFF record in this match set. No matching log
or submitted ZIP was downloaded. These are catalog records, not 30 distinct logs.
The inventory has document IDs, not the original internal filenames. This result
supports BSEE as a possible source. It does not prove exact file identity, or that
Viridien acquired its copy from BSEE. The vendor library pages do not establish
that supply chain. [The full match table](metadata/challenge_metadata_matches.json)
keeps the original metadata fields and the inventory hash.

The inventory download is 49,438,793 bytes. Its hash matches the copied BSEE
report: ce1f8d5f4c7e3514a0f02cb3588a4cd9a83ca3a378b81cb1effd6c0d9dcef20a.
The FRS start page states that Login.gov is used for file requests. No request
was made. Matching challenge records are provenance only and must be excluded
from conversion and training.

## Other routes and access limits

Viridien GeoWells and GeoVerse pages were public. They linked no full mud-log
sample. The linked GeoStore failed TLS hostname validation. Library file access
and Gulf mud-log volume remain unknown. Weatherford returned HTTP 403. Diamond
and Petrolog searches yielded no verified document. Their volume and access
remain unknown. No vendor URL or sample was inferred from a search label.

Search access was limited. DuckDuckGo gave useful initial results, then HTTP 202
challenge pages. Google required JavaScript. Bing gave irrelevant or empty
results. Yahoo reported temporary search errors. Brave returned 429; Mojeek and
Ecosia returned 403; Startpage was blocked. No CAPTCHA was solved. The two
unverified hostname attempts listed in triage.json failed DNS. The missing local
BSEE folder in the copied report was not used. All request URLs, errors, UTC
times and durations are in [access.jsonl](access.jsonl).

## Use limits and checks

No open reuse license was found for the vendor images or PDFs. Copyright terms
remain with each source. Public download access does not grant data reuse rights.
The third-party Sperry manual listing states confidential/proprietary terms;
no manual PDF was downloaded. The 66 MB Geoservices manual listing was not used.

All nine saved vendor documents passed format, byte, hash and page/image checks.
Previews were rendered with one CPU thread. All download and parsing scripts used
Python -I. No installer or demo program was downloaded or run. No form, sign-in,
message, child agent, commit or service was used.

Only the log-name field of the requested CSV was used for the exclusion check.
No evaluation image, evaluation truth file, sealed well, model or weight was
opened. No training, tuning, conversion or benchmark was run. The known names and
APIs do not match any saved vendor sample. Exact byte/pixel deduplication against
the challenge was not done. Keep all files quarantined. Exclude any later exact
challenge copy; it adds provenance, not independent data.
