# Canada offshore search

Best result: Mizzen L-11, Datalog Canada Ltd. Similarity 2 of 3 for the Gulf family by column style. It has a full mud log and a colour symbol legend. Its layout differs from the Gulf 3-panel layout. No score-3 match found. No confirmed match for the other six families.

## Counts

- 17 PDF downloads; 16 unique documents; 1711 unique source pages.
- 15-document first sample, followed by 2 drilling composite files.
- 54 converted pages from one full mud log, including its legend.
- 53 pages have column boxes. Labelled windows: 0. These are unlabelled data.
- 18 candidate inventory entries found across 17 public well pages. This is a bounded sample, not the full hub count.

## Ranked routes

1. [C-NLOER public hub](https://home-cnlopb.hub.arcgis.com/): direct public PDFs. One full log accepted, 55 source pages. Use the Mizzen L-11 Mud Log entry. The adjacent Mudloggers Report is a summary. Geoservices reports for Bay du Nord, Baccalieu and Cappahayden and Weatherford reports for Mizzen are mainly summaries. The sampled drilling composites show engineering curves; they were not accepted.
2. [CNSOER public inventory](https://cnsopbdigitaldata.ca/geoviewer/dmc/public/query/1): 235 well records obtained through the page's getData endpoint. Cheshire, Monterey Jack and Aspy are present. Their linked-item counts are 133, 180 and 127. Counts do not prove that full mud logs are available without login. The [main DMC](https://cnsopbdigitaldata.ca/geoviewer/dmc) has a login form. No account was used.
3. [CER Frontier Information Office](https://www.cer-rec.gc.ca/en/about/library/north-offshore-data.html): records are held by the library. No direct full mud-log download found. The [access page](https://rec-cer.gc.ca/en/about/north-offshore/access-information.html) explains the request route. No request was sent.

## Files and checks

Sources: external_untrusted/round6/r6_canada_offshore/. Converted data: external_unlabelled/round6/r6_canada_offshore/. Scripts and contact sheet: bakeoff/notes/round6_local/r6_canada_offshore/.

Downloads used the required User-Agent, no email, no account, and pauses of at least 1.1 seconds. Source files were read by isolated Python. No outside test data or eval images were used. The Gulf exemplar is recorded in exemplar.json and is outside the eval list. No training was run.

Public download access does not establish a reuse licence. Individual report reuse terms need review. Full-page rendering preserves the source layout. Column boxes are approximate visual bounds; no interval truth was added.

Validation command: python -I -X utf8 bakeoff/notes/round6_local/r6_canada_offshore/check.py <viridien_work absolute path>.
Result: PASS: 16 unique documents; 1711 source pages; 54 valid images; 53 box files; 0 labelled windows.
Contact sheet checked at 1024 px. No text overlap found.

