"""Document seeds, saved reviews, and source-image crops."""
from __future__ import annotations
import copy
import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from PIL import Image

HOME = Path(__file__).resolve().parent
WORK = HOME.parent
ROOT = WORK.parent
BAKEOFF = WORK / 'bakeoff'
DATA = HOME / 'data'
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, str(BAKEOFF))
from harness import io as image_io
DATASET = image_io.DATASET


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('w',encoding='utf-8') as output:
        output.write(json.dumps(value,ensure_ascii=False))
        output.flush()
        os.fsync(output.fileno())
    # Windows can hold a file briefly during a reader or virus scan.
    for attempt in range(8):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.025 * (attempt + 1))


def normalize(value):
    return ''.join(str(value).casefold().split())


def doc_id(stem):
    return hashlib.sha256(stem.encode()).hexdigest()[:16]


def choose_pattern_sources(doc):
    """Choose clean accepted crops against this page's actual legend colour."""
    import numpy as np
    images = Images()
    column_box = doc['boxes']['interpreted_lithology_column']
    column = np.asarray(images.crop(doc, column_box))
    for segment in doc['segments']:
        segment['example'] = False
    for entry in doc['legend']:
        swatch = np.asarray(images.crop(doc, entry['box']))[3:-3, 3:-3]
        pixels = swatch.reshape(-1, 3).astype(float)
        pixels = pixels[pixels.max(1) > 100]
        if not len(pixels):
            continue
        quantized = (pixels // 24).astype(int)
        values, counts = np.unique(quantized, axis=0, return_counts=True)
        colour = np.median(pixels[(quantized == values[counts.argmax()]).all(1)], axis=0)
        ranked = []
        for segment in doc['segments']:
            x0, y0, x1, y1 = segment['box']
            if (segment['legend_ids'] != [entry['id']] or segment.get('flags')
                    or x1-x0 < 12 or y1-y0 < 30):
                continue
            crop = column[max(0,y0-column_box[1]):y1-column_box[1],
                          max(0,x0-column_box[0]+2):max(1,x1-column_box[0]-2)].astype(float)
            if not crop.size:
                continue
            ratio = float((np.sqrt(((crop-colour)**2).sum(2)) < 45).mean())
            # Colour agreement comes first. Prefer enough rows for repeat cycles.
            score = ratio + .08 * min(y1-y0, 160)/160
            if ratio >= .45:
                ranked.append((score, ratio, segment))
        if ranked:
            _, ratio, best = max(ranked, key=lambda value:value[0])
            best['example'] = True
            best['example_match'] = dict(method='legend background colour and crop length', colour_fraction=round(ratio,4))
    if images.current:
        images.current.close()


def prepare(force=False):
    """Reuse the existing queue. Never write to an existing source-label store."""
    seed_dir = DATA / 'seeds'
    index = DATA / 'documents.json'
    if index.exists() and not force:
        return read(index)
    seed_dir.mkdir(parents=True, exist_ok=True)
    imported = read(DATA / 'imported/hadrian.json')
    documents = {}
    for path in sorted((DATASET / 'bbox_annotation').glob('*.json')):
        meta = read(path)
        image = DATASET / 'images' / meta['image_name']
        if not image.exists():
            continue
        stem = image.stem
        ident = doc_id(stem)
        legend_path = BAKEOFF / 'out/_cache/legend_v2' / (stem + '.json')
        entries = read(legend_path).get('entries', []) if legend_path.exists() else []
        legend = [dict(id=e['id'], label_printed=e['label'], box=e['swatch_box']) for e in entries if e.get('is_lithology', True)]
        documents[stem] = dict(id=ident, name=stem, image_name=image.name, width=meta['width'], height=meta['height'], boxes=meta['bounding_boxes'], legend=legend, legend_source='Machine legend. Check the names.', segments=[], revision=0, depth_calibration={})
    queue = BAKEOFF / 'labeltool/data/prefill.jsonl'
    if not queue.exists():
        queue = BAKEOFF / 'labeltool/data/queue.jsonl'
    if queue.exists():
        with queue.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                doc = documents.get(row.get('log'))
                if not doc or doc['image_name'] == imported['image_name']:
                    continue
                col = doc['boxes']['interpreted_lithology_column']
                a, b = max(col[1], int(row['top_y'])), min(col[3], int(row['bottom_y']))
                if b <= a:
                    continue
                match = [e['id'] for e in doc['legend'] if normalize(e['label_printed']) == normalize(row.get('suggestion', ''))]
                doc['segments'].append(dict(id=str(row['id']), box=[col[0], a, col[2], b], legend_ids=match[:1], status='suggested', approved=False, example=False, provenance='saved machine guess', suggestion=row.get('suggestion', ''), reason=row.get('reason', ''), tier=row.get('tier', 'low'), comments=[]))
    for doc in documents.values():
        if doc['image_name'] == imported['image_name']:
            doc['legend'] = imported['legend']
            doc['legend_source'] = 'Your supplied labels'
            doc['depth_calibration'] = imported.get('depth_calibration', {})
            doc['source_layer_count'] = len(imported['layers'])
            for i, layer in enumerate(imported['layers']):
                for j, comp in enumerate(layer['composition']):
                    x0, x1 = comp.get('x_range', doc['boxes']['interpreted_lithology_column'][::2])
                    if x1 <= x0 or layer['bottom_px'] <= layer['top_px']:
                        continue
                    doc['segments'].append(dict(id=f'import-{i}-{j}', layer_id=i, box=[x0, layer['top_px'], x1, layer['bottom_px']], legend_ids=[comp['legend_id']], status='approved', approved=True, example=False, provenance='user supplied labels', proportion=comp.get('proportion'), flags=layer.get('flags', []), comments=[]))
            # Seed one clean source per pattern present in the supplied labels.
            for entry in doc['legend']:
                candidates = [s for s in doc['segments'] if s['legend_ids'] == [entry['id']] and not s.get('flags') and s['box'][2]-s['box'][0] >= 8]
                if candidates:
                    best = max(candidates, key=lambda s: min(s['box'][3]-s['box'][1], 600)*(s['box'][2]-s['box'][0]))
                    best['example'] = True
        if not doc['segments']:
            col = doc['boxes']['interpreted_lithology_column']
            for a in range(col[1], col[3], 600):
                doc['segments'].append(dict(id=f'blank-{a}', box=[col[0], a, col[2], min(a+600, col[3])], legend_ids=[], status='suggested', approved=False, example=False, provenance='unlabelled interval', suggestion='', reason='No saved label guess. Select a legend entry.', comments=[]))
        if doc['legend_source'] == 'Your supplied labels':
            choose_pattern_sources(doc)
        doc['segments'].sort(key=lambda s: (s['box'][1], s['box'][0]))
        write(seed_dir / (doc['id'] + '.json'), doc)
    catalog = [dict(id=d['id'], name=d['name'], count=len(d['segments']), legend_count=len(d['legend']), supplied=d['legend_source']=='Your supplied labels') for d in documents.values()]
    catalog.sort(key=lambda d: (not d['supplied'], d['name']))
    write(index, catalog)
    return catalog


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, data=DATA):
        self.data = Path(data)
        self.lock = threading.RLock()
        self.catalog = read(self.data / 'documents.json')
        self.ids = {d['id'] for d in self.catalog}

    def get(self, ident):
        if ident not in self.ids:
            raise ValueError('Unknown document')
        with self.lock:
            path = self.data / 'reviews' / (ident + '.json')
            return read(path if path.exists() else self.data / 'seeds' / (ident + '.json'))

    def save(self, ident, revision, segment=None, delete_id=None, legend=None):
        with self.lock:
            doc = self.get(ident)
            if revision != doc['revision']:
                raise Conflict('A newer review is saved. Export your draft, then reload this document.')
            if segment is not None:
                self.validate(doc, segment)
                current = next((s for s in doc['segments'] if s['id']==segment['id']), None)
                if current:
                    doc['segments'][doc['segments'].index(current)] = copy.deepcopy(segment)
                else:
                    doc['segments'].append(copy.deepcopy(segment))
            if delete_id:
                doc['segments'] = [s for s in doc['segments'] if s['id'] != delete_id]
            if legend is not None:
                if not isinstance(legend, list) or len(legend) > 200:
                    raise ValueError('Invalid legend')
                ids = [e['id'] for e in legend]
                if len(ids) != len(set(ids)):
                    raise ValueError('Legend IDs must be unique')
                for entry in legend:
                    if not isinstance(entry.get('label_printed'), str) or not entry['label_printed'].strip():
                        raise ValueError('Enter a legend name')
                    self.validate_box(doc, entry['box'])
                if any(i not in ids for s in doc['segments'] for i in s['legend_ids']):
                    raise ValueError('A segment still uses this legend entry')
                doc['legend'] = copy.deepcopy(legend)
            doc['revision'] += 1
            doc['updated_at'] = time.time()
            write(self.data / 'reviews' / (ident + '.json'), doc)
            event = dict(document_id=ident, revision=doc['revision'], timestamp=doc['updated_at'], segment=segment, delete_id=delete_id, legend=legend)
            with (self.data / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            return doc['revision']

    @staticmethod
    def validate_box(doc, box):
        if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Use four whole pixel coordinates')
        if not (0 <= box[0] < box[2] <= doc['width'] and 0 <= box[1] < box[3] <= doc['height']):
            raise ValueError('The box must be inside the image')

    def validate(self, doc, segment):
        if not isinstance(segment.get('id'), str) or not 1 <= len(segment['id']) <= 100:
            raise ValueError('Invalid segment ID')
        self.validate_box(doc, segment['box'])
        ids = segment.get('legend_ids', [])
        if not isinstance(ids, list) or len(ids) != len(set(ids)) or any(i not in {e['id'] for e in doc['legend']} for i in ids):
            raise ValueError('Select labels from this document legend')
        status = segment.get('status')
        if status not in ('suggested', 'approved', 'rejected'):
            raise ValueError('Invalid review status')
        if segment.get('approved') != (status=='approved') or (status=='approved' and not ids):
            raise ValueError('Select a label before you accept the segment')
        if segment.get('example') and status != 'approved':
            raise ValueError('Accept a segment before you use it as a pattern source')
        comments = segment.get('comments', [])
        if not isinstance(comments, list) or len(comments) > 500 or any(not isinstance(c, dict) or not isinstance(c.get('text'), str) or len(c['text']) > 10000 for c in comments):
            raise ValueError('Invalid comments')


class Images:
    def __init__(self):
        self.lock = threading.RLock()
        self.current = None
        self.ident = None

    def crop(self, doc, box):
        Store.validate_box(doc, box)
        with self.lock:
            if self.ident != doc['id']:
                if self.current:
                    self.current.close()
                meta = DATASET / 'bbox_annotation' / (doc['name'] + '.json')
                self.current = image_io.Log(Image.open(DATASET / 'images' / doc['image_name']), read(meta)['bounding_boxes'], doc['name'])
                self.ident = doc['id']
            return Image.fromarray(image_io.read_rgb(self.current, box))
