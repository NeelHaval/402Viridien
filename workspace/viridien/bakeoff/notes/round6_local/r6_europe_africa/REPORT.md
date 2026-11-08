# Europe and Africa: round 6

Best result: four colour Geoservices pages from Norway well 7131/4-1. Similarity is 2 of 3. No exact Gulf three-panel match was found. No deepwater match was confirmed. The selected well gives a water depth of 331 m.

## Counts

- 21 source PDFs, 460 PDF pages, 46,063,527 source bytes.
- 15 completion-log documents sampled from the first 300 Sodir table rows, plus six targeted reports/logs.
- Four full pages converted to PNG, each 1785 by 2526 pixels.
- Zero labelled windows. No training, rock labels or bounding boxes added.
- The page count includes report prose and pressure logs. It is not a count of useful mud-log pages.

## Ranked routes

The order uses observed similarity, available volume and public access. Unknown volume or similarity is not given a made-up score.

| Rank | Route | Similarity | Volume and access | Result |
|---|---|---|---|---|
| 1 | [Sodir public documents](https://factpages.sodir.no/en/wellbore/TableView/With/Documents) | 2 for selected Geoservices log | 37 completion/mud-log URLs in the first 300 rows; direct PDFs | 21 PDFs saved; four pages selected |
| 2 | [Current GEUS portal](https://data.geus.dk/geusmap/?lang=en&mapname=oil_and_gas) | Not scored | Official free released well reports and logs; portal reached | No individual log downloaded |
| 3 | [NLOG](https://www.nlog.nl/datacenter/brh-overview) | Not scored | Public application returns HTTP 200; client-side page | No new wells added |
| 4 | [NSTA open well data](https://www.nstauthority.co.uk/data-and-insights/data/themes/wells/) | Not scored | Open GIS and stratigraphy; no full mud-log route verified outside NDR | No files added |
| 5 | [Ireland DCEE / former PAD](https://www.gov.ie/en/department-of-climate-energy-and-the-environment/publications/oil-and-gas-exploration-production-data/) | Not scored | Official page directs well-data users to a contact request | Skipped; no message sent |
| 6 | [PASA](https://www.petroleumagencysa.com/viewing-our-technical-data/) | Not scored | Orders require registration and issued login | Skipped |
| 7 | [NAMCOR](https://www.namcor.com.na/technical-data/) | Not scored | Mud reports and scanned logs listed; contact/purchase route | Skipped |
| 8 | [ONHYM](https://www.onhym.com/sites/default/files/rapport_onhym_vf2020.pdf) | Not scored | Official report refers to organised data rooms | No direct full log verified |

## Selected pages and comparison

The [7131/4-1 report](https://factpages.sodir.no/pbl/wellbore_documents/5093_7131_4_1_COMPLETION_GEOLOGICAL_REPORT.PDF) has a Geoservices formation evaluation log. Page 106 has the colour rock legend and track titles. Pages 109-112 have coloured cuttings and interpreted rock columns, with gas curves and geological descriptions. These four pages were converted. Earlier pages have little or no rock fill. Pages 113-116 are pressure evaluation logs and were excluded.

The colour and patterned columns support a score of 2 for the Geoservices family. The track order and page shape differ from the challenge exemplar. This is not evidence of a score-3 Gulf match. The flat contact sheet places two non-evaluation challenge exemplars beside the source samples. Long logs use a middle section for legibility. The full source PDFs remain available.

Other samples mostly have narrow black-and-white completion layouts. The 34/7-32 report lists log enclosures on its last page but does not include them. The 6608/11-4 report also lists a formation evaluation enclosure; inspected pages after the enclosure list contain sample descriptions. Do not treat an enclosure list as an acquired full log.

## Access checks and limits

The old GEUS Frisbee host failed DNS. The current [official GEUS page](https://eng.geus.dk/products-services-facilities/data-and-maps/subsurface-data-denmark) links to the working data.geus.dk portal. It states that reports, logs, spliced logs and core photos are free. Its [download guide](https://data.geus.dk/ugdata2d3d/search_guide_uk.pdf) is also public. A well-level download was not verified. This is an open follow-up route, not proof that GEUS requires login.

NLOG returned an application shell with JavaScript files, rather than well links. This lane did not recover a new well download. No excluded well was used. Other African national releases were not exhaustively searched.

Sodir public access was verified. An explicit reuse licence was not verified. The files remain untrusted external data. The table sample is biased to early well names and does not cover all Norwegian releases. Similarity is a visual judgement; it is not a model result.

## Files

- external_untrusted/round6/r6_europe_africa/triage.json: source routes and document records.
- external_untrusted/round6/r6_europe_africa/sodir/: PDFs, discovery list and previews.
- external_unlabelled/round6/r6_europe_africa/images/: four converted pages.
- external_unlabelled/round6/r6_europe_africa/manifest.csv: page provenance and file hashes.
- bakeoff/notes/round6_local/r6_europe_africa/contact_sheet.png: flat comparison sheet.
- Scripts and logs are in this report's folder. finish.py converts the reviewed pages. report.py writes this report and checks the saved files.
