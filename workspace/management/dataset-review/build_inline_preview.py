"""Build a bounded, self-contained view of the real review catalogue."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent


def main():
    catalog = json.loads((ROOT / 'catalog.json').read_text(encoding='utf-8'))
    groups = {}
    for record in catalog.get('records', []):
        key = (record.get('sourceKind'), record.get('family'))
        groups.setdefault(key, []).append(record)
    for values in groups.values():
        values.sort(key=lambda r: (not bool(r.get('ready')), bool(r.get('supersededBy')), -float(r.get('reviewPriority') or 0), str(r.get('id'))))
    selected = []
    for offset in range(1):
        for values in groups.values():
            if offset < len(values):
                selected.append(values[offset])
        if len(selected) >= 18:
            break
    selected = selected[:18]
    assets = []
    bindings = []
    suggestion_bindings = []
    selected_ids = {r.get('id') for r in selected}
    for key in ('suggestions', 'reviewSuggestions'):
        if isinstance(catalog.get(key), list):
            catalog[key] = [s for s in catalog[key] if s.get('recordId', s.get('parentId')) in selected_ids][:60]
    for record in selected:
        if str(record.get('split', '')).lower() in {'sealed', 'verify'} or record.get('sealed'):
            record['image'] = None
            record['thumbnail'] = None
            record['intervals'] = []
            continue
        for field in ('thumbnail', 'image'):
            value = record.get(field)
            if not isinstance(value, str) or value.startswith(('data:', 'http:', 'https:')):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.gif'} and path.stat().st_size <= 300000:
                asset_id = 'preview-asset-' + str(len(assets))
                assets.append('<img hidden id="' + asset_id + '" src="' + path.resolve().as_posix() + '">')
                bindings.append((record.get('id'), field, asset_id))
            else:
                record[field] = None
        if isinstance(record.get('suggestions'), list):
            record['suggestions'] = record['suggestions'][:5]
        if isinstance(record.get('reviewSuggestions'), list):
            record['reviewSuggestions'] = record['reviewSuggestions'][:5]
            for suggestion in record['reviewSuggestions']:
                for field in ('evidenceCrop', 'evidenceCropPath', 'contextCropPath', 'legendContextPaths'):
                    values = suggestion.get(field)
                    if not values:
                        continue
                    for index, value in enumerate(values if isinstance(values, list) else [values]):
                        if not isinstance(value, str) or value.startswith(('data:', 'http:', 'https:')):
                            continue
                        path = Path(value)
                        if not path.is_absolute():
                            path = ROOT / path
                        if path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                            asset_id = 'preview-asset-' + str(len(assets))
                            assets.append('<img hidden id="' + asset_id + '" src="' + path.resolve().as_posix() + '">')
                            suggestion_bindings.append([record.get('id'), suggestion.get('id'), field, index if isinstance(values, list) else None, asset_id])
    total = len(catalog.get('records', []))
    catalog['records'] = selected
    catalog['sources'] = []
    for record in selected:
        record['intervals'] = record.get('intervals', [])[:200]
        record['issues'] = record.get('issues', [])[:8]
    catalog['previewOnly'] = True
    catalog['fullCatalogRecordCount'] = total
    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    css = (ROOT / 'app.css').read_text(encoding='utf-8')
    js = (ROOT / 'app.js').read_text(encoding='utf-8')
    data = json.dumps(catalog, ensure_ascii=False).replace('</', '<\\/')
    html = re.sub(r'<link\s+rel="stylesheet"\s+href="app.css"\s*>', lambda _: '<style>' + css + '</style>', html)
    html = html.replace('<script src="catalog.js"></script>', '<script>window.DATASET_CATALOG=' + data + ';</script>')
    binding_js = 'for(const [id,key,asset] of ' + json.dumps(bindings) + '){const r=window.DATASET_CATALOG.records.find(r=>r.id===id);if(r)r[key]=document.getElementById(asset).src;}'
    binding_js += 'for(const [parent,id,key,index,asset] of ' + json.dumps(suggestion_bindings) + '){const r=window.DATASET_CATALOG.records.find(r=>r.id===parent);const s=(r?.reviewSuggestions||[]).find(s=>s.id===id);if(s){if(index===null)s[key]=document.getElementById(asset).src;else s[key][index]=document.getElementById(asset).src;}}'
    html = html.replace('<script src="app.js"></script>', ''.join(assets) + '<script>' + binding_js + '</script><script src="app.js"></script>')
    html = html.replace('<script src="app.js"></script>', '<script>' + js.replace('</script', '<\\/script') + '</script>')
    notice = '<p style="padding:12px;border:1px solid var(--border);border-radius:var(--radius)">Inline sample: ' + str(len(selected)) + ' records from ' + str(total) + '. Open the local review page for the full catalogue and persistent review.</p>'
    html = html.replace('<main id="app">', '<main id="app">' + notice)
    output = ROOT / 'inline-preview.html'
    output.write_text(html, encoding='utf-8')
    print(f'INLINE_PREVIEW_OK selected={len(selected)} total={total} bytes={output.stat().st_size}')


if __name__ == '__main__':
    main()
