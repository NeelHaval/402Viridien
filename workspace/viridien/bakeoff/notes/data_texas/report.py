"""Make the Texas source report from local data. Run with python -I."""
import csv
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / 'external_untrusted' / 'texas'
NOTES = Path(__file__).resolve().parent
COAST = {'BRAZORIA', 'GALVESTON', 'MATAGORDA', 'JEFFERSON', 'CHAMBERS',
         'HARRIS', 'CALHOUN', 'JACKSON', 'REFUGIO', 'NUECES', 'ARANSAS',
         'SAN PATRICIO', 'KLEBERG', 'KENEDY', 'WILLACY', 'CAMERON'}

def main():
    receipts = []
    for path in sorted(ROOT.glob('*/*.receipt.json')):
        receipt = json.loads(path.read_text())
        source = ROOT / receipt['path']
        assert source.is_relative_to(ROOT)
        raw = source.read_bytes()
        assert len(raw) == receipt['bytes'], source
        assert hashlib.sha256(raw).hexdigest() == receipt['sha256'], source
        receipts.append(receipt)
    rows = json.loads((ROOT / 'rrc_inventory/all_inventory_records.json').read_text())
    summary = json.loads((ROOT / 'rrc_inventory/inventory_summary.json').read_text())
    candidates = json.loads((ROOT / 'rrc_inventory/coast_candidates.json').read_text())
    assert len(rows) == sum(x['rows'] for x in summary)
    assert all(r.get('County Name') and r.get('Log Description') for r in rows)
    coast = [r for r in rows if r['County Name'].strip().upper() in COAST]
    mud = [r for r in rows if 'mud' in r['Log Description'].lower()]
    matched = [r for r in coast if any(v in r['Log Description'].lower() for v in ('mud', 'lith'))]
    assert len(coast) == sum(x['coast_rows'] for x in summary)
    assert len(mud) == sum(x['mud_rows'] for x in summary)
    assert len(matched) == len(candidates) == 0
    fields = list(rows[0])
    for filename, records in [('coast_records.csv', coast), ('statewide_mud_records.csv', mud)]:
        with (ROOT / 'rrc_inventory' / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    source_manifest = [{k: r.get(k) for k in ('url', 'final_url', 'utc', 'status', 'error', 'path', 'bytes', 'sha256', 'content_type')} for r in receipts]
    (ROOT / 'source_manifest.json').write_text(json.dumps(source_manifest, indent=2), encoding='utf-8')
    triage = {
        'lane': 'data_texas', 'family': 'gulf_3panel', 'date_utc': '2026-11-08',
        'status': 'No mud-log image sample was obtained.',
        'admission': 'Hold all files as untrusted source data. No data is approved for training or selection.',
        'scope': {'counties': sorted(COAST), 'inventory_months': len(summary),
                  'records': len(rows), 'coast_records': len(coast), 'statewide_mud_records': len(mud),
                  'coast_mud_or_lithology_records': len(candidates), 'downloaded_log_images': 0,
                  'verified_family_matches': 0},
        'sources': [
            {'name': 'Texas RRC monthly well-log inventories',
             'urls_seen': [r['url'] for r in receipts if r['path'].startswith(('rrc/', 'rrc_inventory/'))],
             'access': 'The public pages and all 12 selected inventory files returned HTTP 200. No sign-in was used.',
             'format': 'XLSX, XLS, and a ZIP with an XLS file. The RRC states that log images use TIFF.',
             'bulk_route': 'The RRC menu links to Energy-Depot for volume downloads. Public Information Act prices apply.',
             'n_available_estimate': 'Unknown. The 12 inventory files have 7301 records. These are not counts of unique wells or usable mud logs.',
             'license_terms': 'The RRC site permits unaltered, non-misleading copies for noncommercial use unless a file states other terms. Rights for commercial use and third-party log images were not established.',
             'documents': summary},
            {'name': 'Texas RRC neuDOCS well-log image query',
             'urls_seen': ['https://rrcsearch3.neubus.com/search-profile?profileId=15'],
             'access': 'HTTP 403 from this VM. No image query or image download was possible. This does not prove that an account is required.',
             'n_available_estimate': None, 'documents': [],
             'family_similarity': None, 'has_lith_column': None, 'has_legend': None},
            {'name': 'Texas RRC public GIS viewer',
             'urls_seen': [r['url'] for r in receipts if r['path'].startswith('gis/')],
             'access': 'HTTP 200 for the public page and two text assets. The assets were read as text only.',
             'finding': 'The published configuration links Well Logs to neuDOCS profile 15. It does not give a separate TIFF file URL.',
             'documents': []},
            {'name': 'Energy-Depot', 'urls_seen': ['https://energy-depot.us/'],
             'access': 'HTTP 200. The page has a sign-in form and a Public Search link to the neuDOCS host. No sign-in or form submission was made.',
             'license_terms': 'The RRC says that Public Information Act prices apply to volume downloads. No price quote was requested.',
             'documents': []},
            {'name': 'Petrora discovery index', 'urls_seen': ['https://petrora.com/logs'],
             'access': 'HTTP 403 with a Cloudflare challenge. No challenge was bypassed.', 'documents': []}
        ],
        'family_fit': {'score': None, 'reason': 'No source log image was obtained. The lithology column, printed legend, and three-panel layout could not be checked.'},
        'limits': [
            'Only 12 monthly inventory files were checked. This is not a search of the full archive.',
            'The county set is explicit. It does not include every inland county in the Gulf Coast basin.',
            'Log Description is a source label. A combined log can have lithology that its label does not show.',
            'Image file names in the inventories are not full download URLs. They were not treated as file hashes.',
            'A metadata depth range is not interval truth.',
            'No contact sheet was made because no candidate log image was obtained.'
        ],
        'data_discipline': {
            'models_or_weights_used': False, 'historical_weight_contamination': 'Not applicable. No model or inherited weights were used.',
            'training_cv_tuning_or_benchmark': False, 'evaluation_or_truth_files_read': False,
            'sealed_well_search': False, 'team_services_used': False, 'child_agents_used': False,
            'files_written': ['external_untrusted/texas/', 'bakeoff/notes/data_texas/']
        }
    }
    (ROOT / 'triage.json').write_text(json.dumps(triage, indent=2), encoding='utf-8')
    table = '\n'.join(f"| [{x['url'].split('/')[-1]}]({x['url']}) | {x['rows']} | {x['coast_rows']} | {x['mud_rows']} |" for x in sorted(summary, key=lambda r: r['url'].split('/')[-1]))
    report = f"""Texas RRC source check. Date: 8 November 2026.

No Gulf 3-panel mud-log sample was obtained. The RRC image query returned HTTP 403 from this VM. No image was available to check the lithology column, legend, or layout. Family fit is unknown. Do not count an inventory file as a mud-log sample.

I downloaded and read 12 monthly inventory files. They have {len(rows):,} records. There are {len(coast)} records in the selected Gulf Coast counties. None of those records has a Log Description that contains Mud or Lithology. The inventories have {len(mud)} Mud records in other counties. These counts are records, not unique wells. A source label can omit parts of a combined log. The zero count does not prove that the archive has no useful Gulf Coast logs.

The county set is: {', '.join(sorted(COAST))}. It does not cover every inland county in the Gulf Coast basin.

| Public inventory file | All records | Coast records | Mud records, all counties |
| --- | ---: | ---: | ---: |
{table}

The primary source is the [RRC Imaged Records Menu](https://www.rrc.texas.gov/resource-center/research/research-queries/imaged-records/imaged-records-menu/). It says that the public well-log image series starts in July 2004. It points to the Bureau of Economic Geology for earlier logs. The [monthly inventory page](https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/obtaining-commission-records/oil-gas-well-records-gis-well-logs/) says that images use TIFF. The [glossary](https://www.rrc.texas.gov/oil-and-gas/research-and-statistics/obtaining-commission-records/glossary-for-well-log-list/) defines the image file name, depth limits, comments, and API number. It lists type 10 as Lithology. A depth limit in this table is not interval truth.

The [well-log query](https://rrcsearch3.neubus.com/search-profile?profileId=15) returned HTTP 403. The saved response is 118 bytes. This does not prove that a login is required. The [public GIS viewer](https://gis.rrc.texas.gov/GISViewer/) returned HTTP 200. Its published map configuration sends the Well Logs link to the same query, with profileId=15 and the search_fields-api_no parameter. I read the two saved JavaScript assets as text. I did not execute them. They give no separate TIFF download address.

The RRC menu links to [Energy-Depot](https://energy-depot.us/) for volume downloads. The RRC says that Texas Public Information Act prices apply. The Energy-Depot page returned HTTP 200 and showed a sign-in form. It also linked to Public Search on the neuDOCS host. I did not sign in, request a quote, submit a form, or send a message. The [Petrora index](https://petrora.com/logs) returned HTTP 403 with a Cloudflare challenge. I did not bypass it.

The [RRC site policy](https://www.rrc.texas.gov/site-policies/) permits copies for noncommercial use if the content stays unaltered and is not misleading, unless an individual item states other terms. The site asks for source credit and a link. This is not a confirmed grant for commercial use or for all third-party log images. Keep the raw files and their source links.

The public route for a later manual check is the RRC menu, then Oil & Gas Well Logs. Search by an eight-digit Texas API from the inventory. Check the actual image before you call it a mud log or a Gulf 3-panel match. For volume access, use the linked Energy-Depot route and check the stated price and terms. The number of useful Gulf 3-panel images is unknown. No free bulk image route was confirmed.

The raw source files, HTTP receipts, and SHA-256 values are in external_untrusted/texas/. source_manifest.json lists each request. triage.json gives the result and its limits. rrc_inventory/coast_records.csv keeps all {len(coast)} coast records. rrc_inventory/statewide_mud_records.csv keeps the {len(mud)} Mud records as source metadata only. All files remain on hold as untrusted source data. No log image was admitted to a data set.

No model or inherited weights were used. Historical weight contamination does not apply to this lane. I did not read evaluation truth, search for sealed wells, train, tune, run cross-validation, or run a benchmark. I did not read the challenge images. No child agent or team service was used. All writes stay in the two owned folders. Download and parsing Python ran with -I. The thread limits were set to 3. No downloaded file was executed. The installed xls2csv tool read XLS files as data. No credentials, personal data, or email address was sent.

Verification: each saved response passed its byte-count and SHA-256 check. All {len(rows):,} parsed records have a county and a log description. The county and Mud totals match the per-file totals. The XLS file in the July 2010 ZIP was read by its exact file name; no archive paths were used for extraction. No new test suite was needed for this source check.
"""
    (NOTES / 'REPORT.md').write_text(report, encoding='utf-8')
    verification = {'source_receipts_checked': len(receipts), 'response_bytes': sum(r['bytes'] for r in receipts),
                    'inventory_files': len(summary), 'records': len(rows), 'coast_records': len(coast),
                    'statewide_mud_records': len(mud), 'matched_candidates': len(candidates), 'hash_checks': 'pass'}
    (NOTES / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    main()
