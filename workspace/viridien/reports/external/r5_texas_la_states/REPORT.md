# State log route check

Date: 8 November 2026. Lane: `r5_texas_la_states`.

Two Mississippi logs are ready for source review. Both score **2 of 3**.
They show depth, lithology patterns, gas tracks, rock text, and printed legends.
They are old black-and-white scans from coastal land wells. They are not
confirmed state-waters logs. No score 3 sample was found.

| Candidate | Pages saved | Local review packet | Source page list |
| --- | ---: | --- | --- |
| AMOCO Cumbest et al. #1, API 23059200030000 | 8 | [PDF](documents/ms_cumbest_23059200030000_review.pdf) | [DEQ XML](https://geology.deq.ms.gov/Energy/Data02/MudLogs/Jackson/mdl123059200030000a.xml) |
| Mobil USA 1-7 #1, header also says ST-1, API 23047200120000 | 2 | [PDF](documents/ms_usa_1_7_23047200120000_review.pdf) | [DEQ XML](https://geology.deq.ms.gov/Energy/Data02/MudLogs/Harrison/mdl523047200120000a.xml) |

These PDFs are local review packets made from the downloaded PNG pages.
The source did not supply these PDFs. All original PNG files remain in `sources/`.
The two Harrison pages are section a, not the full well log.

See the [labelled Jackson preview](previews/ms_jackson_a_001.png),
[labelled Harrison preview](previews/ms_harrison_a_001.png), and
[eight-page contact sheet](previews/jackson_8page_contact.png).
[confirmed_regions.json](confirmed_regions.json) gives the source URL,
page, and pixel box for each confirmed lithology column and legend.
These are review boxes, not interval labels.

The Mississippi route ranks first: similarity 2 × two confirmed well groups
× public access 1 = **4**. Other routes have unknown log volume or unknown
similarity. Their products are unknown. The coastal mud-log search lists
11 well rows. Do not treat all 11 as confirmed matches or count page slices
as separate wells. A third sampled header has a catalog/header identity
conflict. It stays on hold. The Harrison full-GIF link returned HTTP 404;
its XML-listed PNG pages work.

The GLO result is a **near-name provenance lead**, not an exact match.
[Unit 11493](https://cgis.glo.texas.gov/glomap/Units.cfm?PALEASE=11493)
is **Rio Nueces 37-44 Unit 2**, Midland County, Spraberry field.
Its [mineral-file scan](https://cdn.glo.texas.gov/ncu/SCANDOCS/archives_webfiles/arcmaps/webfiles/landgrants/PDFs/9/9/9/999455.pdf)
downloaded without login: 148 pages, 50,838,041 bytes.
Local file: `sources/glo_MF101494.pdf`.
Pages 135-139 name Endeavor Energy Resources and survey Sections 37 and 44.
Thus, 37-44 is a section pair in this record, not GLO unit number 3744.
The challenge names say Unit 1. No exact Rio Unit 1, Lavaca, or San Gabriel
record was found. The GLO well-inventory queries returned service errors.
The scan is legal provenance evidence, not a confirmed mud-log candidate.

Other access results:

- **RRC:** Energy-Depot links to the current Public Search host. That host
  returned HTTP 403. The older API host failed DNS. No new image route was found.
- **Louisiana:** SONRIS and GIS metadata now work. GIS returns exact document
  links for state-lease wells. The old `ucmsearch` route redirects to current
  Document Access, then CAPTCHA. The linked Well Logs page also reaches CAPTCHA.
  No scan was obtained.
- **Mississippi state records:** Five coastal state-named well folders were
  checked. API **23045000070000** has the source note **“Locality offshore.”**
  Its folder lists scout cards and samples, with no mud-log link.
- **Alabama state waters:** One new public hearing packet identifies
  **MB Aloe Bay 75-2 SL 701 No 2**, permit **12604-OS-90-B**, Mobile Offshore.
  [Source PDF](https://www.gsa.state.al.us/Home/DownloadFTSDocument/?path=Exhibit%5C2016%5C2016-24&fileName=2016-24.pdf),
  15 pages; local file `sources/al_mobilebay_2016_24.pdf`.
  Page 12 was checked by sight. No mud log was confirmed in this packet.

The copied Texas and Alabama inventories were not repeated. No exact challenge
copy was found. Only the 51 challenge names were used for overlap review.
Byte-level overlap and other-lane overlap remain untested. Any later exact
challenge match must stay out of conversion and training. The BSEE/library
and CGG/Viridien acquisition hypotheses remain unproved by this state lane.

All files stay in this lane. No evaluation image, evaluation truth, sealed
well file, training, tuning, or benchmark was used. No sign-in, message,
submission, service, child agent, or commit was made. Downloads and parsing
used Python `-I`. Requests were serial, with at least 1.1 seconds before
each transaction. Rendering used one CPU thread. No downloaded code ran.

No reuse license for submitted log images was established. All 104 saved
HTTP responses passed byte and SHA-256 checks. Access times and errors are
in `access.jsonl`. The full handoff is in `triage.json`; hashes are in
`artifact_hashes.json`. No data was admitted to training or evaluation.
