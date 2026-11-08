Texas RRC source check. Date: 8 November 2026.

No Gulf 3-panel mud-log sample was obtained. The RRC image query returned HTTP 403 from this VM. No image was available to check the lithology column, legend, or layout. Family fit is unknown. Do not count an inventory file as a mud-log sample.

I downloaded and read 12 monthly inventory files. They have 7,301 records. There are 82 records in the selected Gulf Coast counties. None of those records has a Log Description that contains Mud or Lithology. The inventories have 146 Mud records in other counties. These counts are records, not unique wells. A source label can omit parts of a combined log. The zero count does not prove that the archive has no useful Gulf Coast logs.

The county set is: ARANSAS, BRAZORIA, CALHOUN, CAMERON, CHAMBERS, GALVESTON, HARRIS, JACKSON, JEFFERSON, KENEDY, KLEBERG, MATAGORDA, NUECES, REFUGIO, SAN PATRICIO, WILLACY. It does not cover every inland county in the Gulf Coast basin.

| Public inventory file | All records | Coast records | Mud records, all counties |
| --- | ---: | ---: | ---: |
| [012026_master_inventory.xls](https://www.rrc.texas.gov/media/ocxbsmv3/012026_master_inventory.xls) | 807 | 11 | 21 |
| [022026_master_inventory.xls](https://www.rrc.texas.gov/media/exsfzd4u/022026_master_inventory.xls) | 573 | 7 | 27 |
| [032026_master_inventory.xlsx](https://www.rrc.texas.gov/media/i44lyhey/032026_master_inventory.xlsx) | 216 | 3 | 2 |
| [042026_master_inventory.xlsx](https://www.rrc.texas.gov/media/dqxbbsjp/042026_master_inventory.xlsx) | 202 | 0 | 4 |
| [052026_master_inventory.xlsx](https://www.rrc.texas.gov/media/pszdyfob/052026_master_inventory.xlsx) | 260 | 0 | 3 |
| [062026_master_inventory.xlsx](https://www.rrc.texas.gov/media/5z2dprbc/062026_master_inventory.xlsx) | 264 | 5 | 6 |
| [072026_master_inventory.xlsx](https://www.rrc.texas.gov/media/egwbfdbv/072026_master_inventory.xlsx) | 334 | 0 | 0 |
| [082026_master_inventory.xlsx](https://www.rrc.texas.gov/media/kmknkuyk/082026_master_inventory.xlsx) | 361 | 2 | 1 |
| [092026_master_inventory.xlsx](https://www.rrc.texas.gov/media/ut2k3qyb/092026_master_inventory.xlsx) | 1232 | 11 | 20 |
| [february-2025.xls](https://www.rrc.texas.gov/media/snsabt1k/february-2025.xls) | 1239 | 3 | 29 |
| [july-2024.xls](https://www.rrc.texas.gov/media/f0smypqe/july-2024.xls) | 702 | 13 | 33 |
| [july2010.zip](https://www.rrc.texas.gov/media/isqarjed/july2010.zip) | 1111 | 27 | 0 |

The primary source is the [RRC Imaged Records Menu](https://www.rrc.texas.gov/resource-center/research/research-queries/imaged-records/imaged-records-menu/). It says that the public well-log image series starts in July 2004. It points to the Bureau of Economic Geology for earlier logs. The [monthly inventory page](https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/obtaining-commission-records/oil-gas-well-records-gis-well-logs/) says that images use TIFF. The [glossary](https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/obtaining-commission-records/glossary-for-well-log-list/) defines the image file name, depth limits, comments, and API number. It lists type 10 as Lithology. A depth limit in this table is not interval truth.

The [well-log query](https://rrcsearch3.neubus.com/search-profile?profileId=15) returned HTTP 403. The saved response is 118 bytes. This does not prove that a login is required. The [public GIS viewer](https://gis.rrc.texas.gov/GISViewer/) returned HTTP 200. Its published map configuration sends the Well Logs link to the same query, with profileId=15 and the search_fields-api_no parameter. I read the two saved JavaScript assets as text. I did not execute them. They give no separate TIFF download address.

The RRC menu links to [Energy-Depot](https://energy-depot.us/) for volume downloads. The RRC says that Texas Public Information Act prices apply. The Energy-Depot page returned HTTP 200 and showed a sign-in form. It also linked to Public Search on the neuDOCS host. I did not sign in, request a quote, submit a form, or send a message. The [Petrora index](https://petrora.com/logs) returned HTTP 403 with a Cloudflare challenge. I did not bypass it.

The [RRC site policy](https://www.rrc.texas.gov/site-policies/) permits copies for noncommercial use if the content stays unaltered and is not misleading, unless an individual item states other terms. The site asks for source credit and a link. This is not a confirmed grant for commercial use or for all third-party log images. Keep the raw files and their source links.

The public route for a later manual check is the RRC menu, then Oil & Gas Well Logs. Search by an eight-digit Texas API from the inventory. Check the actual image before you call it a mud log or a Gulf 3-panel match. For volume access, use the linked Energy-Depot route and check the stated price and terms. The number of useful Gulf 3-panel images is unknown. No free bulk image route was confirmed.

The raw source files, HTTP receipts, and SHA-256 values are in external_untrusted/texas/. source_manifest.json lists each request. triage.json gives the result and its limits. rrc_inventory/coast_records.csv keeps all 82 coast records. rrc_inventory/statewide_mud_records.csv keeps the 146 Mud records as source metadata only. All files remain on hold as untrusted source data. No log image was admitted to a data set.

No model or inherited weights were used. Historical weight contamination does not apply to this lane. I did not read evaluation truth, search for sealed wells, train, tune, run cross-validation, or run a benchmark. I did not read the challenge images. No child agent or team service was used. All writes stay in the two owned folders. Download and parsing Python ran with -I. The thread limits were set to 3. No downloaded file was executed. The installed xls2csv tool read XLS files as data. No credentials, personal data, or email address was sent.

Verification: each saved response passed its byte-count and SHA-256 check. All 7,301 parsed records have a county and a log description. The county and Mud totals match the per-file totals. The XLS file in the July 2010 ZIP was read by its exact file name; no archive paths were used for extraction. No new test suite was needed for this source check.
