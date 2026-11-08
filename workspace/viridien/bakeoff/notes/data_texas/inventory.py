"""Read bounded RRC XLSX files as data. Run with python -I."""
import collections
import csv
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3] / 'external_untrusted' / 'texas'
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
COAST = {'BRAZORIA', 'GALVESTON', 'MATAGORDA', 'JEFFERSON', 'CHAMBERS',
         'HARRIS', 'CALHOUN', 'JACKSON', 'REFUGIO', 'NUECES', 'ARANSAS',
         'SAN PATRICIO', 'KLEBERG', 'KENEDY', 'WILLACY', 'CAMERON'}

def xlsx_rows(path):
    with zipfile.ZipFile(path) as archive:
        if sum(v.file_size for v in archive.infolist()) > 50 * 1024 * 1024:
            raise ValueError('Workbook exceeds read cap')
        shared = ET.fromstring(archive.read('xl/sharedStrings.xml'))
        strings = [''.join(v.itertext()) for v in shared.findall('m:si', NS)]
        for name in sorted(archive.namelist()):
            if not (name.startswith('xl/worksheets/sheet') and name.endswith('.xml')):
                continue
            rows = ET.fromstring(archive.read(name)).findall('m:sheetData/m:row', NS)
            header = None
            for row in rows:
                values = {}
                for cell in row:
                    value = cell.find('m:v', NS)
                    value = value.text if value is not None and value.text else ''
                    if value and cell.get('t') == 's':
                        value = strings[int(value)]
                    if cell.get('t') == 'inlineStr':
                        value = ''.join(cell.find('m:is', NS).itertext())
                    column = ''.join(c for c in cell.get('r') if c.isalpha())
                    values[column] = value
                if header is None:
                    header = values
                elif values.get('B', '').isdigit():
                    yield {header.get(k, k): v for k, v in values.items()}

def main():
    allrows = []
    summaries = []
    for path in sorted((ROOT / 'rrc_inventory').glob('*.xlsx')):
        receipt = json.loads(path.with_suffix('.receipt.json').read_text())
        rows = list(xlsx_rows(path))
        for row in rows:
            row.update(inventory_url=receipt['url'], inventory_sha256=receipt['sha256'],
                       inventory_path=str(path.relative_to(ROOT)))
        allrows.extend(rows)
        summaries.append({'url': receipt['url'], 'rows': len(rows),
                          'coast_rows': sum(r.get('County Name') in COAST for r in rows),
                          'mud_rows': sum('mud' in r.get('Log Description', '').lower() for r in rows),
                          'descriptions': dict(collections.Counter(r.get('Log Description') for r in rows))})
    for old in sorted((ROOT / 'rrc_inventory').glob('*.xls')):
        parsed = subprocess.run(['xls2csv', str(old)], capture_output=True, timeout=30, check=True)
        csv_path = old.with_suffix('.csv')
        csv_path.write_bytes(parsed.stdout)
        rows = list(csv.DictReader(io.StringIO(parsed.stdout.decode('utf-8', errors='replace').replace('\x0c', ''))))
        rows = [r for r in rows if r.get('API Number', '').isdigit()]
        receipt_path = old.with_suffix('.receipt.json') if old.stem != 'july2010' else old.parent / '4c62a4f038c9.receipt.json'
        receipt = json.loads(receipt_path.read_text())
        for row in rows:
            row.update(inventory_url=receipt['url'], inventory_sha256=receipt['sha256'],
                       inventory_path=str(csv_path.relative_to(ROOT)))
        allrows.extend(rows)
        summaries.append({'url': receipt['url'], 'rows': len(rows),
                          'coast_rows': sum(r.get('County Name') in COAST for r in rows),
                          'mud_rows': sum('mud' in r.get('Log Description', '').lower() for r in rows),
                          'descriptions': dict(collections.Counter(r.get('Log Description') for r in rows))})
    candidates = []
    for row in allrows:
        description = row.get('Log Description', '').lower()
        if row.get('County Name') in COAST and any(w in description for w in ['mud', 'lith']):
            row.update(image_downloaded=False, has_lith_column=None, has_legend=None,
                       family_similarity=None,
                       family_fit='Not checked. Image access is blocked.',
                       admission='Hold. Do not use in training or selection.')
            candidates.append(row)
    target = ROOT / 'rrc_inventory'
    (target / 'inventory_summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    (target / 'coast_candidates.json').write_text(json.dumps(candidates, indent=2), encoding='utf-8')
    (target / 'all_inventory_records.json').write_text(json.dumps(allrows, indent=2), encoding='utf-8')
    print(json.dumps({'summaries': summaries, 'coast_candidates': candidates}, indent=2))

if __name__ == '__main__':
    main()
