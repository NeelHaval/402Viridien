# BSEE mud-log request list

Date: 8 November 2026. Lane: bsee_list.

The public inventory is available. It has **15,923 Gulf mud-log TIFF files**.
Of these files, **959** have a RUN_DATE in 2005 or later and a WATER_DEPTH
greater than 1,000. These 959 files cover **126 API well groups**.

Use [pilot_request.csv](pilot_request.csv) for the first request. It has
**60 files from 20 well groups**, with a total listed size of
**1,490,473,041 bytes** (1.49 GB; 1.39 GiB). The pilot covers eight Gulf areas.
The owner must sign in to submit the request. No request was submitted.

## Files for the owner

| File | Use |
| --- | --- |
| [ranked_wells.csv](ranked_wells.csv) | All 126 well groups in score order. Includes the best document and the reason for its score. |
| [pilot_request.csv](pilot_request.csv) | The 60 selected files. Includes each exact document ID, API, well suffix, lease, area, block, size, and source record number. |
| [pilot_filename_ids.txt](pilot_filename_ids.txt) | One exact Filename ID per line. Use this as a copy list. Bulk upload of this text file was not checked. |
| [request_documents.csv](request_documents.csv) | All 959 files, in well rank order. All 24 source fields are kept. |
| [checks.json](checks.json) | Counts, byte totals, source hash, and data checks. |
| [portal_checks.json](portal_checks.json) | Three checks against the public query page. |

The 959 files have a total listed size of **16,035,241,458 bytes**
(16.04 GB; 14.93 GiB). This is below the current 25 GB request limit.
The owner can request the full list if that size is suitable. Select only
the listed documents. Other files for the same wells can add to the size.

## Public source

The [BSEE raw-data page](https://www.data.bsee.gov/Main/RawData.aspx) links to
the [File Request System inventory ZIP](https://www.data.bsee.gov/Other/Files/FRSWellDataRawData.zip).
The ZIP was obtained without sign-in. The server gave a last-change time of
7 November 2026, 10:33:37 UTC. The file was saved on 8 November 2026.

The archive has 1,929,275 data records. Its text member is
`FRSWellDataRawData/mv_frs_file_inventory.txt`.
The archive is 49,438,793 bytes. Its SHA-256 is:

```text
ce1f8d5f4c7e3514a0f02cb3588a4cd9a83ca3a378b81cb1effd6c0d9dcef20a
```

The source ZIP, saved web pages, and source hashes are in
[external_untrusted/bsee/sources](../../../external_untrusted/bsee/sources/).
See [SOURCES.json](../../../external_untrusted/bsee/sources/SOURCES.json)
for the URL of each saved response. External files were treated as untrusted
data. No downloaded code was run. No email address was sent.

## Selection rules

The file type must be `.tif` or `.tiff`. The title must contain `mud log`,
`mudlog`, or the same words with a hyphen or underscore between them.
The match ends at a word boundary. REGION must be `G`.

This gives the stated count of 15,923 files. A search for `mud` alone also
finds borehole imagers and other tools. Those files are outside this list.
Other regions have 1,936 more mud-log TIFF files. They are also outside this list.

The next filter uses RUN_DATE year >= 2005 and WATER_DEPTH > 1000.
This gives the stated count of 959 files. RUN_DATE is the field in the
inventory. A well spud date was not obtained. WATER_DEPTH is water depth,
not the total depth of the well. The 1,000 ft water-depth rule follows the
task context. BLI minus TLI gives the indexed log span used below.

Each candidate must have a public release date on or before the snapshot
date. All 959 pass this check.

The first ten digits of each API form a well group for this list. The full
API and WELL_NAME_SUFFIX stay in the document CSV. Do not replace them
with the ten-digit group when you place a request.

Only the names of the 51 local annotation files were read to check overlap.
They give 20 API roots. None of the 959 candidate files has one of these
roots. This check cannot detect a renamed copy with a missing or different API.
See [local_api_roots.json](local_api_roots.json).

## Rank rules and limits

The score uses metadata only. The rules were set from the stated target:
recent Gulf mud logs with a likely formation track. No image or truth
score was used to set these rules.

| Evidence in the inventory | Points |
| --- | --- |
| Area GC, MC, AT, or GB | 30 |
| Area WR, KC, AC, or EB | 20 |
| Other Gulf area | 10 |
| Title names physical formation, master log, or lithology | 25 |
| Else, title names formation evaluation | 20 |
| Else, title names engineering, drilling data, gas analysis, gas ratio, or shows | 10 |
| Scale token 1in, 2in, or 5in | 15, 10, or 5 |
| Title states MD or TVD | 5 |
| Indexed log span >= 5,000, or >= 2,000 | 10, or 5 |
| Title states final | 5 |

Use one area score and one title-content score. Add the other points.
A scale with an explicit denominator other than 100 ft gets no scale points.
For example, `1in/1000ft` gets no scale points. A bare `1in` is only a scale
token; its full scale is not confirmed. The maximum score is 90.

A well gets the score of its best file. Equal scores use the longer indexed
span, then the API root. The score sets a request order. It does not show a
measured image match or a probability.

The title can support a formation-track choice. It cannot confirm three
panels, an interpreted lithology column, or a printed legend. Colour,
resolution, scan quality, logging software, and exact interval truth are
unknown. All request rows state this uncertainty.

LOG_SOURCE and LOG_PKG_VOLUME identify image supply records. They do not
prove which mud-logging company drew the log. No score uses these fields.
No operator or named oil field was inferred from a numeric well name.

## Leading wells

These are the first eight well groups. Each has a score of 80. The CSV
files have the full API and well suffix for each document.

| Rank | API root | Area and block | Best document ID and title evidence |
| --- | --- | --- | --- |
| 1 | 6081140458 | GC 468 | clog06009; 1in TVD, formation evaluation |
| 2 | 6081741052 | MC 722 | clog16351; 1in TVD, integrated formation evaluation |
| 3 | 6081741047 | MC 24 | clog16341; 1in TVD, formation evaluation and gas analysis |
| 4 | 6081140444 | GC 652 | clog09789; 1in TVD, formation evaluation |
| 5 | 6081741055 | MC 674 | clog16345; 1in, physical formation, show reports |
| 6 | 6081140443 | GC 238 | clog14045; 1in TVD, formation evaluation |
| 7 | 6081140454 | GC 472 | clog11745; 1in, masterlog |
| 8 | 6081741058 | MC 734 | clog11915; 1in TVD, formation evaluation |

## Pilot coverage

The pilot starts with the best three wells in each of GC, MC, AT, and GB.
It adds the best well in each of WR, KC, AC, and EB. It then adds wells in
rank order until there are 20, with at most six wells per area.

For each well, select its best file. Add up to two files with a different
scale token or stated depth basis. This gives three files for each pilot
well. These files can be alternate views of the same intervals. Treat all
files from a well as one group in any later data split.

| Area | Pilot wells | Pilot files |
| --- | --- | --- |
| Green Canyon (GC) | 6 | 18 |
| Mississippi Canyon (MC) | 4 | 12 |
| Atwater Valley (AT) | 3 | 9 |
| Garden Banks (GB) | 3 | 9 |
| East Breaks (EB) | 1 | 3 |
| Keathley Canyon (KC) | 1 | 3 |
| Alaminos Canyon (AC) | 1 | 3 |
| Walker Ridge (WR) | 1 | 3 |

## Owner request steps

1. Open the [BSEE File Request System](https://www.data.bsee.gov/Other/FileRequestSystem/Default.aspx).
2. Sign in with Login.gov. The owner must do this step.
3. Open [Well Data](https://www.data.bsee.gov/Other/FileRequestSystem/WellData.aspx).
4. Copy a DOC_ID from `pilot_request.csv` into **Filename ID**. Keep its
   exact text. Do not add `.tif`. Submit the query.
5. Check the returned API, well suffix, area, block, and title against the
   CSV. Select that record. Follow the site controls to add it to the request.
6. Repeat for the other selected IDs. Check the cart size before you submit.
7. Submit the request. Download the files when BSEE makes them available.

The public Filename ID search was checked with `clog06009`, `clog16345`,
and `clog11745`. Each search gave one row. All 15 displayed data fields
matched the saved inventory, after display spaces were reduced. A fourth
ID, `US60807402290300_MUD_12147`, also gave a matching public row during
source discovery. Only the three formal checks are in `portal_checks.json`.
Cart actions and bulk ID upload were not checked because they need sign-in.

BSEE states that publicly releasable records are free to request. The
current request limit is 25 GB. File preparation can take up to 24 hours.
The download link stays active for five days. These facts are on the
File Request System start page. A separate reuse licence for each submitted
log was not established. No file download URL was invented.

## Checks and data discipline

All 959 document IDs are unique. Every raw field in the request CSV was
checked against its source record. The pilot has no duplicate IDs. Its
listed size is below 25 GB. The full list and the pilot use the same
eligibility rules. The list has no excluded records; the empty exclusion
CSV keeps its column headers.

No image, truth file, model, or inherited weight was read. No model was
trained or tuned. Thus, no inherited-weight contamination applies to this
lane. No benchmark was run. The sealed splits were not opened or rebuilt.
The one final benchmark remains with the orchestrator.

All new files are inside the two owned folders. No child agent, team
service, sign-in, email, commit, or push was used.

To build the tables again from the saved ZIP, run from the repo root:

```sh
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
LITHOLOGY_DATASET=~/lith/dataset ~/lith/.venv/bin/python -I \
  bakeoff/notes/bsee_list/build_request_list.py
```

The code gets the repo path from its own file. It gets the dataset path
from `LITHOLOGY_DATASET`, `--dataset`, or a sibling `dataset` folder.
The code has no fixed Windows or VM path. It uses the Python standard
library, streams the archive, and does not import Torch.

To repeat the three public metadata checks, use the same thread settings:

```sh
~/lith/.venv/bin/python -I bakeoff/notes/bsee_list/check_portal.py
```

No new pytest tests were added. The checks above validate the source data
and the live query results. They do not run the model harness.
