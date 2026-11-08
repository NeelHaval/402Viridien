# Alabama SOGB public mud logs

Date: 8 November 2026. Lane: data_alabama.

## Result

The public hearing files contain two useful mud-log sections from one well.
The well is **GW-Edge Gilmore 29-9 #1**, permit **10127**, well key **7993**.
It is in the Southwest Range field, Conecuh County, Alabama.
The state well record lists the Upper Smackover and Lower Smackover pools.

Four PNG images are saved at 200 dpi. Each image is 2200 by 1700 pixels.
Two images are the primary sample. Two images are later scans of the same sections.
Keep all four images in the same well group. They are not four independent samples.

The source files stay in `external_untrusted/alabama/`.
No model, labels, training data, or benchmark results were changed.

## Sample

| Source file | PDF pages, counted from 1 | Content | Use |
| --- | --- | --- | --- |
| `documents/exhibit_03.pdf` | 46, 47 | Upper and lower mud-log sections | Primary sample |
| `documents/exhibit_04.pdf` | 46, 47 | Same two sections | Duplicate evidence only |
| `documents/exhibit_07.pdf` | 13, 14 | Later scans of the same two sections | Scan quality review only |

The rendered pixels on pages 46 and 47 are identical in files 03 and 04.
The PDF file hashes differ. Thus, a PDF hash alone does not detect these duplicates.
File 07 has different rendered pixels, but the well and sections are the same.
Its testimony on pages 2 and 3 confirms the well name, permit, field, and pools.

The primary images are:

- `external_untrusted/alabama/images/al_sogb_10127_upper_2015.png`
- `external_untrusted/alabama/images/al_sogb_10127_lower_2015.png`

The images show depth marks, lithology patterns, gas curves, and rock descriptions.
The upper section has visible marks at 13,400 and 13,500 ft.
The lower section has visible marks at 13,600 and 13,700 ft.
These are visible marks, not calibrated page limits.
The pages have no visible lithology legend. No exact truth or bounding boxes were made.

## Primary sources

The state website is the source for all saved files. The following links required no account.

- [State well database](https://www.gsa.state.al.us/ogb/wells)
- [State full text document search](https://www.gsa.state.al.us/ogb/docs)
- [2015 hearing exhibit, primary sample](https://www.gsa.state.al.us/Home/DownloadFTSDocument/?path=Exhibit%5C2015%5C2015-36&fileName=Exhibit+DN+3-24-15-24+2015-36.pdf)
- [2015 companion hearing exhibit](https://www.gsa.state.al.us/Home/DownloadFTSDocument/?path=Exhibit%5C2015%5C2015-37&fileName=Exhibit+DN+3-24-15-25+2015-37.pdf)
- [2023 hearing exhibit, later scans](https://www.gsa.state.al.us/Home/DownloadFTSDocument/?path=Exhibit%5C2023%5C2023-20&fileName=Hearing+Exhibits+05-02-23-07+.pdf)
- [State facilities and log order page](https://www.gsa.state.al.us/inter/facilities)
- [State public records policy](https://www.gsa.state.al.us/inter/public_records)

The public website code gives the read-only POST requests used for these searches:

| Endpoint | Request | Saved evidence |
| --- | --- | --- |
| `/Wells/CountySearch/` | JSON `{"SearchTerm":"Mobile"}` | `source/wells_mobile.json` |
| `/ModalDetails/CountyLogs/?county=Mobile` | No body | `source/county_logs_mobile.json` |
| `/Document/getDocuments/?fieldName=mud%20log` | No body | `source/mud_log_search.json` |
| `/Wells/PermitSearch/` | JSON `{"SearchTerm":"10127"}` | `source/well_10127.json` |

The endpoint host is `https://www.gsa.state.al.us`.
The saved website code was read as text. It was not executed.
Each request used the generic User-Agent `Mozilla/5.0`.
No email address, credentials, or sign-in was sent.

## Access limits

The Mobile County log catalog has 3,842 records. Of these, 178 have type `MUD`.
They cover 130 distinct permit strings. A permit count is not a unique well count.
The state type list defines `MUD` as Mud Log and `ML` as Microlog.
The saved mud-log catalog includes only `MUD` records.

The log table gives scan file numbers, but no public scan links.
It tells users to contact the log librarian to acquire logs.
The facilities page says orders require payment. It lists these prices:

- Paper: USD 0.75 per linear foot.
- TIFF: USD 0.65 per equivalent linear foot.
- LAS: USD 4.00 per MB.

No order, payment, or contact was made. No scan paths were guessed.
The public hearing PDFs used here are a separate public download route.
No open reuse license was found on the checked pages. Public access alone does not establish reuse rights.
The public records policy also permits fees and protection of sensitive records.
The `robots.txt` request returned HTTP 404. This gives no added permission.

## Download bound and checks

Seven hearing PDF requests were made. Six files were saved.
Their total size is 18,237,275 bytes, or 17.39 MiB.
One 1990 exhibit exceeded the 20 MiB file limit. Its transfer stopped and no partial file was saved.
The other files include two one-page drilling letters and one 1980 exhibit packet.
Those three files are search evidence. They add no confirmed mud-log sample.
No transcript files were downloaded.

All six saved PDFs passed a PDF signature check and opened without a password.
All 33 saved HTTP response files matched their recorded SHA-256 hashes.
The four sample PNG files also passed image format and size checks.
The two primary sections were checked by sight at 100 dpi.
Text extraction, page checks, and image rendering used isolated Python (`-I`).
Downloaded files were never executed. Thread limits were set to three.
The tools did not load torch or model weights.

The source folder contains:

- `access.jsonl`: request URL, time, method, status, headers, size, and hash. Failed requests are included.
- `SHA256SUMS`: hashes for all source files and derived files.
- `sample_manifest.json`: page numbers, image paths, hashes, well group, section group, access, and limits.
- `mud_catalog_mobile.csv`: 178 mud-log catalog records, joined to state well metadata by permit.
- `pdf_checks.json`: PDF page counts, hashes, and text search hits.
- `checks.json`: machine-readable check results.
- `source/pdf_text/`: text from the saved PDFs. Treat this text as untrusted source data.

## Gulf family fit

Fit is moderate. The sample is from the Gulf Coastal Plain Smackover carbonate setting.
It has the pattern tracks, gas tracks, depth marks, and rock text found in mud logs.
The text includes anhydrite, dolomite, shale, and limestone terms.
These features can help with a later scan and pattern review.

The sample is small and carbonate rich. The scans are mostly black and white.
The missing legend and missing full log limit legend matching and depth calibration.
This sample does not establish coverage of modern offshore Gulf log styles.
It is not ready for scored evaluation.

## Data discipline and handoff

No training, CV, tuning, or benchmark run was made.
No inherited model weights were used. This lane has no inherited weight contamination to report.
This statement does not certify weights from other lanes.
No held-out truth, sealed well source, or full dev split was inspected.
No team service or other user was contacted. No commit or push was made.

Keep this sample outside all training and evaluation sets until the orchestrator assigns it.
If it is later used, group all source versions by `al_sogb_10127`.
The two section groups are `al_sogb_10127_upper` and `al_sogb_10127_lower`.
The shared-verify exclusion, the 51-window ext_dev rule, and the one final benchmark remain in force.

To repeat the local file checks and image conversion from the repository root:

```sh
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 \
  "$LITH_PYTHON" -I bakeoff/notes/data_alabama/check_sample.py
```

Set `LITH_PYTHON` to the local Python path. On Windows, use the same Python file with `-I`
and set the three thread environment variables in the local shell.
The code derives file paths from the repository. It contains no VM or Windows absolute paths.
