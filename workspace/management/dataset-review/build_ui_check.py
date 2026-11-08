"""Prepare a small real-data browser check. No source dataset is changed."""
from pathlib import Path
import base64
import copy
import json
import io
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parent


def inline(path_value):
    if not isinstance(path_value, str) or path_value.startswith('data:'):
        return path_value
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.webp'} and path.stat().st_size < 800_000:
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail((256, 256))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=45)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return None


def main():
    catalog = json.loads((ROOT / 'catalog.json').read_text(encoding='utf-8'))
    chosen = []
    for kind in ('current', 'synthetic', 'outside'):
        candidates = [r for r in catalog['records'] if r.get('sourceKind') == kind]
        candidates.sort(key=lambda r: (not any(s.get('legendContextPaths') for s in r.get('reviewSuggestions', [])), not bool(r.get('reviewSuggestions')), not bool(r.get('image')), r['id']))
        chosen.extend(candidates[:1])
    small = copy.deepcopy(catalog)
    fields = {'id', 'sourceKind', 'family', 'split', 'truthStatus', 'ready', 'image', 'thumbnail', 'imageWidth', 'imageHeight', 'boxes', 'sourceBoxes', 'imageBounds', 'thumbnailBounds', 'issues', 'metricReadiness', 'baseWell', 'groupId', 'reviewSuggestions', 'modelReview', 'showTruthInReview', 'sealed'}
    small['records'] = [{k: copy.deepcopy(v) for k, v in r.items() if k in fields} for r in chosen]
    for record in small['records']:
        for field in ('image', 'thumbnail'):
            record[field] = inline(record.get(field))
        record['reviewSuggestions'] = sorted(record.get('reviewSuggestions', []), key=lambda s: (not bool(s.get('legendContextPaths')), -(s.get('reviewPriority') or 0)))[:1]
        record['boxes'] = record.get('boxes', [])[:8]
        for suggestion in record['reviewSuggestions']:
            suggestion['evidenceCropPath'] = inline(suggestion.get('evidenceCropPath'))
            suggestion['contextCropPath'] = inline(suggestion.get('contextCropPath'))
            suggestion['legendContextPaths'] = [inline(p) if isinstance(p, str) else p for p in suggestion.get('legendContextPaths', [])]
    html = (ROOT / 'index.html').read_text(encoding='utf-8')
    html = html.replace('<link rel="stylesheet" href="app.css">', '<style>' + (ROOT / 'app.css').read_text(encoding='utf-8') + '</style>')
    data = json.dumps(small, ensure_ascii=False).replace('</', '<\\/')
    html = html.replace('<script src="catalog.js"></script>', '<script>window.DATASET_CATALOG=' + data + ';</script>')
    html = html.replace('<script src="app.js"></script>', '<script>' + (ROOT / 'app.js').read_text(encoding='utf-8') + '</script>')
    checks = (ROOT / 'ui-browser-check.js').read_text(encoding='utf-8')
    real_check = """<script>
const current=DatasetReview.state.records.find(r=>r.sourceKind==='current');
const synth=DatasetReview.state.records.find(r=>r.sourceKind==='synthetic');
if(current)DatasetReview.pinRecord(current.id);
if(synth)DatasetReview.pinRecord(synth.id);
console.log('REAL_CATALOG_CHECK records='+DatasetReview.state.records.length+' suggestions='+DatasetReview.allSuggestions().length+' pinned='+document.querySelectorAll('[data-viewer]').length);
Promise.all([...document.images].map(i=>i.complete?Promise.resolve(i.naturalWidth>0):new Promise(resolve=>{i.onload=()=>resolve(true);i.onerror=()=>resolve(false)}))).then(values=>console.log('REAL_IMAGE_CHECK '+values.filter(Boolean).length+'/'+values.length));
</script>"""
    html = html.replace('</body>', '<script>' + checks + '</script>' + real_check + '</body>')
    (ROOT / 'ui-check-real.html').write_text(html, encoding='utf-8')
    if '--emit-html' in sys.argv:
        print(json.dumps({'html': html}))
    else:
        print(f'UI_CHECK_HTML_READY records={len(small["records"])} bytes={len(html.encode("utf-8"))} output=ui-check-real.html')


if __name__ == '__main__':
    main()
