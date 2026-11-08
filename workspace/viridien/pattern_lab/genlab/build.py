"""Rebuild patterns, fakes, dataset and scores from the current human labels.

Only patterns whose approved boxes changed are relearned; everything is
written under data/genlab. Run directly for a one-off rebuild.
"""
from __future__ import annotations

import hashlib
import copy
import json
import shutil
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import model as labstore  # noqa: E402  (pattern_lab/model.py, read-only use)

from . import candidates, learn, paths, proposals, render, score, selection  # noqa: E402
from normalized import letterbox
from feedback import canonical
from .reading import ReadOnlyStore

SAMPLES = 4          # per pattern, shown on the page
SAMPLE_ROWS = 240
COLUMNS = 12         # per document, written to the dataset
COLUMN_ROWS = 1200
SHOW_COLUMNS = 3
COLUMN_MIN_BOXES = 3  # a pattern joins stacked columns once it has this many boxes


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
    for i in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(.05 * (i + 1))
    tmp.replace(path)


def _png(rgb, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return path.relative_to(paths.OUT).as_posix()


def ratings():
    latest = {}
    if paths.RATINGS.exists():
        for line in paths.RATINGS.read_text(encoding='utf-8').splitlines():
            try:
                r = json.loads(line)
                latest[r['file']] = r
            except (ValueError, KeyError):
                pass
    return latest


class Builder:
    def __init__(self):
        self.images = labstore.Images()
        self.crops = learn.Crops(self.images)
        self.cache = {}        # pattern key -> (signature, model, data, realism)
        self.doc_cache = {}    # doc id -> (label-set signature, rendered outputs)
        self.prefill = score.load_prefill()
        self.queue = score.queue_guesses()
        self.candidate_cache = {}
        self.transfer_pairs = {}
        self.column_cache = {}

    def build(self):
        t0 = time.time()
        self.prefill = score.load_prefill()
        dependencies = [paths.DATA/'normalized'/'split-policy.json',paths.DATA/'normalized'/'current.json',paths.PREFILL]
        dependencies.append(paths.OUT/'reprov'/'ancestry.jsonl')
        dependencies += [Path(module.__file__) for module in (learn,render,score,candidates)] + [Path(__file__)]
        dependency_hash = hashlib.sha256()
        for path in dependencies:
            if path.exists():
                dependency_hash.update(path.read_bytes())
        dependency_sig = dependency_hash.hexdigest()
        self.queue = score.queue_guesses()
        store = ReadOnlyStore(labstore.Store())
        docs, groups = learn.human_examples(store)
        human_groups = dict(groups)
        human_boxes = sum(len(v) for v in human_groups.values())
        metadata = {}
        proposal_rows = []
        # Reports are produced by the explicit scanner. A changed human label
        # set refreshes just that document; polling never scans missing reports.
        for doc_id, info in docs.items():
            doc_proposals = [learn.reproduced_proposal(p,docs) for p in proposals.inventory(doc_id)]
            proposal_rows.extend(doc_proposals)
            if not info['training_allowed']:continue
            path = paths.CANDIDATES / (doc_id + '.json')
            report = None
            if path.exists():
                stamp = path.stat().st_mtime_ns
                cached = self.candidate_cache.get(doc_id)
                report = cached[1] if cached and cached[0] == stamp else json.loads(path.read_text(encoding='utf-8'))
                if report.get('signature') != candidates.signature(info['doc'], human_groups):
                    rows = next((v for n, v in self.prefill.items() if learn.norm(n) == learn.norm(info['name'])), [])
                    report = candidates.refresh(info['doc'], self.crops, human_groups, rows,
                                                donors=candidates.donor_pool(), docs=docs)
                    candidates.summary()
                    stamp = path.stat().st_mtime_ns
                self.candidate_cache[doc_id] = (stamp, report)
            vp = paths.VERDICTS / (doc_id + '.json')
            verdicts = json.loads(vp.read_text(encoding='utf-8')) if vp.exists() else {}
            for entry in (report or {}).get('entries', []):
                key = (doc_id, entry['legend_id'])
                tier = entry['tier']
                if tier == 'not-lithology':
                    groups.pop(key, None)
                    continue
                if tier == 'borrowed':
                    donor = entry['donor']
                    dp = paths.VERDICTS / (donor['doc_id'] + '.json')
                    dv = json.loads(dp.read_text(encoding='utf-8')) if dp.exists() else {}
                    if (learn.generation_split(docs[donor['doc_id']]['doc'])[1]!='train' or
                            dv.get(str(donor['legend_id']), {}).get('verdict') == 'reject'):
                        continue
                    donor_path = paths.CANDIDATES / (donor['doc_id'] + '.json')
                    donor_report = json.loads(donor_path.read_text(encoding='utf-8')) if donor_path.exists() else {}
                    donor_entry = next((x for x in donor_report.get('entries', [])
                                        if x['legend_id'] == donor['legend_id']), {})
                    # A label edit can revoke a donor while this target report
                    # remains cached. Never generate from that stale approval.
                    if (donor_entry.get('tier') not in ('human', 'verified')
                            or donor_entry.get('data_file') != donor['data_file']
                            or donor_report.get('signature') != candidates.signature(docs[donor['doc_id']]['doc'], human_groups)):
                        continue
                eligible = tier in ('verified', 'borrowed') or (
                    tier == 'swatch' and verdicts.get(str(key[1]), {}).get('verdict') == 'confident')
                if eligible and key not in human_groups:
                    groups[key] = (candidates.examples(info['doc'], entry) if tier == 'verified' else
                                   [dict(id=f'{tier}-{key[1]}', doc_id=doc_id, legend_id=key[1],
                                         label=entry['label'], box=entry['swatch_box'],
                                         provenance=tier, flagged=False)])
                    metadata[key] = dict(source='machine-verified' if tier == 'verified' else tier,
                                         column_matches=entry['matches'],
                                         candidate=entry, verdict=verdicts.get(str(key[1])))
                    if tier == 'borrowed':
                        metadata[key]['donor_tile_source'] = donor_entry.get('model', {}).get('tile_source', 'column')
            for key in list(groups):
                if key[0] != doc_id:
                    continue
                meta = metadata.setdefault(key, dict(source='human', column_matches=0))
                if meta['source'] == 'human':
                    meta['candidate'] = next((e for e in (report or {}).get('entries', []) if e['legend_id'] == key[1]), None)
                meta['verdict'] = verdicts.get(str(key[1]))
                if (meta['verdict'] or {}).get('verdict') == 'reject':
                    del groups[key]
            # Human and verified sources win; approved proposals replace only
            # borrowed/swatch sources. A legacy owner reject still excludes all.
            excluded = {e['legend_id'] for e in (report or {}).get('entries', [])
                        if e['tier'] == 'not-lithology'}
            for p in doc_proposals:
                if not learn.lineage_allowed(p, docs):
                    continue
                p['pixel_sha256'] = hashlib.sha256(Path(p['tile_path']).read_bytes()).hexdigest()
                with Image.open(p['tile_path']) as im:
                    if not learn.pixels_allowed(np.asarray(im.convert('RGB'))):
                        continue
                key = (doc_id, p['legend_id'])
                if (not p['approved'] or key[1] in excluded
                        or verdicts.get(str(key[1]), {}).get('verdict') == 'reject'
                        or (key in groups and metadata[key]['source'] in ('human', 'machine-verified'))):
                    continue
                if p['source'] == 'agent-made' and key[1] not in info['legend']:
                    continue
                groups[key] = proposals.examples(p, info['doc'])
                metadata[key] = dict(source=p['source'], column_matches=0,
                                     proposal=p, verdict=p['reviewer'])
        patterns, changed_docs = {}, set()
        # one document at a time keeps the TIFF reader on one page
        for key in sorted(groups, key=lambda k: k[0]):
            exs = groups[key]
            meta = metadata[key]
            sig = hashlib.sha256((dependency_sig + learn.MODEL_VERSION + str(candidates.VERSION) + learn.signature(exs) + json.dumps(meta, sort_keys=True)).encode()).hexdigest()[:16]
            hit = self.cache.get(key)
            if hit and hit[0] == sig:
                patterns[key] = hit
                continue
            if meta['source'] in ('agent-made', 'unlisted'):
                try:
                    m, data = learn.load_proposal(meta['proposal'], exs, docs)
                except ValueError:
                    groups.pop(key,None)
                    continue
            elif meta['source'] not in ('human','machine-verified'):
                entry = meta['candidate']
                m = copy.deepcopy(entry['model'])
                data = candidates.load_data(paths.CANDIDATES / entry['data_file'])
            else:
                column_sig = dependency_sig + learn.signature(exs)
                column_hit = self.column_cache.get(key)
                if column_hit and column_hit[0]==column_sig:
                    m,data = copy.deepcopy(column_hit[1:])
                else:
                    m, data = learn.learn(self.crops, docs, key, exs)
                    self.column_cache[key]=(column_sig,copy.deepcopy(m),copy.deepcopy(data))
                entry = meta.get('candidate')
                if data is not None:
                    self.transfer_pairs.pop(key,None)
                    group, split = learn.generation_split(docs[key[0]]['doc'])
                    if split=='train' and data['kind']=='tile' and entry:
                        _, sw, repeat = candidates.swatch(self.crops.get(docs[key[0]]['doc'],entry['swatch_box']))
                        if not repeat['seam_risk']:
                            rows = next((v for n,v in self.prefill.items() if learn.norm(n)==learn.norm(docs[key[0]]['name'])),[])
                            self.transfer_pairs[key] = dict(doc_id=key[0],legend_id=key[1],family=rows[0].get('family','unknown') if rows else 'unknown',
                                period_delta=int(data['tile'].shape[0]-sw['tile'].shape[0]),
                                stroke_delta=learn.stroke_weight(data['tile'])-learn.stroke_weight(sw['tile']),
                                colour_delta=(learn.background(data['tile'])-learn.background(sw['tile'])).tolist(),
                                lineage=dict(doc_id=key[0],source_group=group,split=split,source_revision=docs[key[0]]['doc'].get('revision'),
                                             legend_id=key[1],column_boxes=m.get('training_boxes',[]),swatch_box=entry['swatch_box']))
                    swdata, risk = None, True
                    if entry and not entry.get('seam_risk', True):
                        _, swdata, repeat = candidates.swatch(self.crops.get(docs[key[0]]['doc'], entry['swatch_box']))
                        risk = repeat['seam_risk']
                    m, data = candidates.choose_tile(self.crops, docs[key[0]]['doc'], exs, m, data, swdata, risk)
            reference_kind = ('target column' if meta['source'] in ('human','machine-verified','unlisted') or
                              meta.get('proposal',{}).get('column_boxes') else 'legend')
            reference_doc, reference_examples = docs[key[0]]['doc'], exs
            if meta['source']=='borrowed':
                donor = meta['candidate']['donor']
                reference_kind = 'donor column'
                reference_doc = docs[donor['doc_id']]['doc']
                reference_examples = [dict(e,id=f'donor-{i}') for i,e in enumerate(donor['evidence'])]
            if data is not None and data['kind']=='quilt':
                frame_examples=[e for k,ex in human_groups.items() if k[0]==key[0] and k in groups for e in ex]
                if frame_examples:m['validation_rails']=learn.rails(self.crops,docs,key[0],frame_examples)
            real = score.realism(m,data,self.crops,reference_doc,reference_examples,reference_kind=reference_kind)
            if meta['source']=='machine-verified' and real.get('held_out'):
                real['disjoint_fit_crops']=real['held_out']
                real.update(held_out=0,independent_depth_ranges=0,reused_training=True,
                    evidence_note='Depth-disjoint from tile fitting; earlier swatch verification used these crops, so this is reconstruction evidence.')
            validated=(reference_kind=='target column' and bool(real.get('acceptable')) and
                       data is not None and (data['kind']!='quilt' or
                       bool(real.get('morphology_acceptable') and real.get('independent_depth_ranges',0)>0)))
            m.update(reference_kind=reference_kind,column_validated=bool(validated),provisional=not validated)
            m['signature'] = sig
            m.update(source=meta['source'], column_matches=meta['column_matches'], verdict=meta['verdict'])
            m.setdefault('tile_source', 'legend swatch' if meta['source'] == 'swatch' else 'column')
            if meta['source'] == 'borrowed':
                m['donor'] = meta['candidate']['donor']
                m['tile_source'] = meta['donor_tile_source']
            if meta['source'] not in ('agent-made','unlisted'):
                # These sources are fitted here from documented native inputs,
                # rather than inferred from an opaque historical proposal.
                source_doc=reference_doc
                source_boxes=(m.get('training_boxes') or [e['box'] for e in reference_examples])
                source_boxes=list(source_boxes)+[b for b in m.get('native_source_boxes',[]) if b not in source_boxes]
                if m.get('tile_source')=='legend swatch' and meta.get('candidate'):
                    source_boxes=source_boxes+[meta['candidate']['swatch_box']]
                group,split=learn.generation_split(source_doc)
                pixel_hashes=[]
                for box in source_boxes:
                    rgb=self.crops.get(source_doc,box);h,w=rgb.shape[:2]
                    pixel_hashes.append(hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest())
                m['lineage']=[dict(doc_id=source_doc['id'],legend_id=(m.get('donor') or {}).get('legend_id',key[1]),source_revision=source_doc.get('revision'),
                    source_group=group,split=split,boxes=source_boxes,pixel_sha256=pixel_hashes,kind='native fit inputs')]
                m['ancestor_docs']=sorted({source_doc['id'],key[0]})
                m['normalized_sources']=[dict(sample_id=e['normalized_sample_id'],annotation_version=e['annotation_version'])
                    for e in reference_examples if 'normalized_sample_id' in e and e['id'] in m.get('sources',[])]
                if source_doc['id']!=key[0] and meta.get('candidate'):
                    target_doc=docs[key[0]]['doc'];box=meta['candidate']['swatch_box']
                    rgb=self.crops.get(target_doc,box);h,w=rgb.shape[:2]
                    group,split=learn.generation_split(target_doc)
                    m['lineage'].append(dict(doc_id=key[0],legend_id=key[1],source_revision=target_doc.get('revision'),
                        source_group=group,split=split,boxes=[box],
                        pixel_sha256=[hashlib.sha256(f'{w}x{h}:'.encode()+rgb.tobytes()).hexdigest()],kind='target legend selection'))
            if learn.generation_split(docs[key[0]]['doc'])[1]=='train' and not learn.lineage_allowed(m,docs):
                groups.pop(key,None)
                continue
            patterns[key] = (sig, m, data, real)
            changed_docs.add(key[0])
        transfer = learn.fit_transfer([v for k,v in self.transfer_pairs.items() if k in patterns])
        _write(paths.OUT/'recover'/'transfer.json',transfer)
        for key,(sig,m,data,real) in list(patterns.items()):
            if m['reference_kind']!='legend' or data is None:
                continue
            rows = next((v for n,v in self.prefill.items() if learn.norm(n)==learn.norm(docs[key[0]]['name'])),[])
            family = rows[0].get('family','unknown') if rows else 'unknown'
            rule = transfer['families'].get(family)
            m['legend_column_transfer'] = dict(family=family,usable=bool(rule and rule['usable']),
                reason=(rule or {}).get('reason','No training pairs'),uncertainty=(rule or {}).get('uncertainty'))
            transferred = data if data.get('transfer') else learn.apply_transfer(data,rule)
            if transferred is not data and not np.array_equal(transferred['tile'],data['tile']):
                sig = hashlib.sha256((sig+json.dumps(transferred['transfer'],sort_keys=True)).encode()).hexdigest()[:16]
                m['signature']=sig;m['tile_size']=[transferred['tile'].shape[1],transferred['tile'].shape[0]]
                m['period']=dict(x=transferred['tile'].shape[1],y=transferred['tile'].shape[0])
                real = score.realism(m,transferred,self.crops,docs[key[0]]['doc'],groups[key],reference_kind='legend')
                m['legend_column_transfer']['applied']=transferred['transfer']
                patterns[key]=(sig,m,transferred,real)
                changed_docs.add(key[0])
        self.cache = patterns

        rated = ratings()
        cards, doc_reports = [], []
        manifest = dict(generated_at=time.time(), patterns=[], columns=[],
                        schema='strata.synthetic.v1',kind='synthetic',split='train',source_policy='Test wells cannot generate or donate. Human > machine-verified > agent-made > borrowed > swatch. Agent-made and unlisted proposals require a confident review of the current round.',
                        note='Every pattern records its source and donor, when applicable.')
        by_doc = defaultdict(dict)
        rebuilt_docs = []
        for key, (sig, m, data, real) in patterns.items():
            if data is not None:
                by_doc[key[0]][key[1]] = data
        for doc_id, models in by_doc.items():
            sig_doc = '-'.join(sorted(patterns[(doc_id, k)][0][:6] for k in models))
            hit = self.doc_cache.get(doc_id)
            if not hit or hit[0] != sig_doc:
                hit = (sig_doc, self._render_doc(doc_id, models, docs, groups, patterns, changed_docs, sig_doc))
                self.doc_cache[doc_id] = hit
                rebuilt_docs.append(doc_id)
            doc_cards, report, m_patterns, m_columns = hit[1]
            for c in doc_cards:
                votes = Counter(rated[f]['verdict'] for f in c['samples'] if f in rated)
                c['ratings'] = dict(good=votes['good'], bad=votes['bad'])
            cards.extend(doc_cards)
            doc_reports.append(report)
            manifest['patterns'].extend(m_patterns)
            manifest['columns'].extend(m_columns)
        self.doc_cache = {k: v for k, v in self.doc_cache.items() if k in by_doc}
        self._prune(patterns, by_doc)

        # machine-vs-human accuracy
        full = []
        for d in docs.values():
            if d['doc'].get('legend_source') == 'Your supplied labels':
                r = score.full_truth_doc(d['doc'], self.prefill.get(d['name'], []))
                if r:
                    r['column'] = d['column']
                    full.append(r)
        reviewed = score.reviewed_boxes(docs, self.prefill, self.queue)
        acc = [i for i in reviewed if i['status'] == 'approved']
        summary = dict(
            reviewed=len(reviewed), accepted=len(acc), rejected=len(reviewed) - len(acc),
            queue_label_right=sum(i['guess_ok'] for i in acc if i['guess']),
            queue_label_total=sum(1 for i in acc if i['guess']),
            row_guess_right=sum(i['row_guess_ok'] for i in acc if i['row_guess']),
            row_guess_total=sum(1 for i in acc if i['row_guess']),
            median_iou=(float(np.median([i['iou'] for i in acc if i['iou'] is not None]))
                        if any(i['iou'] is not None for i in acc) else None))
        cards.sort(key=lambda c: (learn.norm(c['label']), -c['examples']))
        state = dict(built_at=time.time(), seconds=round(time.time() - t0, 1),
                     human_boxes=human_boxes,
                     human_pattern_count=sum(c['source'] == 'human' for c in cards),
                     verified_pattern_count=sum(c['source'] == 'machine-verified' for c in cards),
                     borrowed_pattern_count=sum(c['source'] == 'borrowed' for c in cards),
                     swatch_pattern_count=sum(c['source'] == 'swatch' for c in cards),
                     agent_made_pattern_count=sum(c['source'] == 'agent-made' for c in cards),
                     unlisted_pattern_count=sum(c['source'] == 'unlisted' for c in cards),
                     rebuilt_documents=rebuilt_docs,
                     pattern_count=len(cards), cards=cards, documents=doc_reports,
                     legend_swatch_tile_count=sum(c.get('tile_source') == 'legend swatch' for c in cards),
                     coverage=score.coverage(docs, list(candidates.reports()), self.prefill, patterns, proposal_rows),
                     accuracy=dict(full=full, reviewed=reviewed, summary=summary,
                                   reviewer=score.reviewer_accuracy(rated, manifest['patterns'])),
                     dataset=dict(samples=len(manifest['patterns']), columns=len(manifest['columns']),
                                  folder=str(paths.GEN)))
        _write(paths.GEN / 'manifest.json', manifest)
        _write(paths.STATE, state)
        return state

    def _render_doc(self, doc_id, models, docs, groups, patterns, changed_docs, sig_doc):
        """Render samples and columns for one document (cached by label set)."""
        cards, manifest = [], dict(patterns=[], columns=[])
        rated = {}
        d = docs[doc_id]
        exs = [e for k, v in groups.items() if k[0] == doc_id for e in v]
        column_exs = [e for e in exs if patterns[(doc_id, e['legend_id'])][1]['source'] in ('human', 'machine-verified')]
        rail_exs = [e for e in column_exs if patterns[(doc_id, e['legend_id'])][1]['source'] == 'human']
        if not rail_exs:
            col = d['column']
            rail_exs = [dict(e, box=[col[0], e['box'][1], col[2], e['box'][3]]) for e in column_exs]
            if not rail_exs:
                rail_exs = [dict(box=[col[0], col[1], col[2], min(col[3], col[1] + 96)])]
        rails = learn.rails(self.crops, docs, doc_id, rail_exs)
        folder = paths.GEN / doc_id
        rng = np.random.default_rng(int(hashlib.sha256(sig_doc.encode()).hexdigest()[:8], 16))
        samples_rgb = {}
        for lid, data in models.items():
            key = (doc_id, lid)
            sig, m, _, real = patterns[key]
            payload=learn.export_model(data,folder,sig,m)
            m['render_model']=payload
            m.update({k:payload['placement'][k] for k in ('phase','anchored','anchored_x','anchored_y','x_repeat','x_origin')})
            files = []
            rgbs = []
            for k in range(SAMPLES):
                rgb = render.sample(data, rails, SAMPLE_ROWS, rng)
                rgbs.append(rgb)
                rel = folder / ('p%s-%s-%d.png' % (lid, sig[:8], k))
                if not rel.exists() or doc_id in changed_docs:
                    _png(rgb, rel)
                files.append(rel.relative_to(paths.OUT).as_posix())
            samples_rgb[lid] = rgbs
            tile_file = None
            preview_file = None
            if data['kind'] == 'tile':
                tile_file = _png(data['tile'], folder / ('tile-%s-%s.png' % (lid, sig[:8])))
                preview_file = _png(render.repeat_preview(data, rails['x1']-rails['x0'], bool(m.get('period', {}).get('x')),x0=rails['x0']),
                                    folder / ('repeat-%s-%s.png' % (lid, sig[:8])))
            bad = {x['id'] for x in m.get('suspects', [])}
            # show only boxes that agree with the pattern; suspects get their own row
            real_boxes = [e['box'][:3] + [min(e['box'][3], e['box'][1] + SAMPLE_ROWS)]
                          for e in sorted((e for e in groups[key] if e['id'] not in bad),
                                          key=lambda e: -(e['box'][3] - e['box'][1]))[:5]]
            real_doc_id = doc_id
            if m['source'] == 'borrowed':
                real_doc_id = m['donor']['doc_id']
                real_boxes = [e['box'][:3] + [min(e['box'][3], e['box'][1] + SAMPLE_ROWS)]
                              for e in m['donor']['evidence'][:5]]
            for x in m.get('suspects', []):
                x['looks_like'] = self._nearest_label(d, models, lid, x['box'])
            votes = Counter(rated[f]['verdict'] for f in files if f in rated)
            cards.append(dict(m, doc_name=d['name'], realism=real, samples=files, tile=tile_file,
                              repeat_preview=preview_file,
                              repeat_size=([m['tile_size'][0] if m.get('period', {}).get('x') else rails['x1']-rails['x0'],
                                            m['tile_size'][1]] if data['kind']=='tile' else None),
                              real=real_boxes, real_doc_id=real_doc_id,
                              ratings=dict(good=votes['good'], bad=votes['bad']),
                              width=rails['x1'] - rails['x0']))
            for f in files:
                native=paths.OUT/f
                with Image.open(native) as image:
                    image=image.convert('RGB');normalized,mask,transform=letterbox(image)
                    pixel_hash=hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest()
                fixed=native.with_name(native.stem+'-rgb224.png');valid=native.with_name(native.stem+'-mask.png')
                normalized.save(fixed);mask.save(valid)
                sources=[dict(sample_id=e['normalized_sample_id'],annotation_version=e['annotation_version']) for e in groups[key] if e['id'] in m['sources'] and 'normalized_sample_id' in e]
                target=(m['key'] if m['source'] == 'unlisted' else
                        'rock-'+hashlib.sha256(canonical(m['label']).encode()).hexdigest()[:12])
                manifest['patterns'].append(dict(file=f, document=d['name'], legend_id=lid, label=m['label'],
                                                 pattern_id=m['key'], doc_id=doc_id,
                                                 looks_like=m.get('looks_like'), method=m.get('method'),
                                                 proposal_round=m.get('proposal_round'),
                                                 source=m['source'], verdict=m['verdict'],
                                                 tile_source=m.get('tile_source', 'column' if m['source'] == 'borrowed' else 'legend swatch'),
                                                 tile_scores=m.get('tile_scores'), repeat_preview=preview_file,
                                                 reference_kind=m['reference_kind'], column_validated=m['column_validated'],
                                                 provisional=m['provisional'], realism=real,
                                                 donor=m.get('donor'), source_doc_id=real_doc_id,
                                                 lineage=m.get('lineage'),recovery=m.get('recovery'),
                                                 ancestor_docs=m.get('ancestor_docs'),render_model=payload,signature=sig,
                                                 machine_evidence=([dict(box=e['box'], provenance=e['provenance']) for e in groups[key] if e['id'] in m['sources']] if m['source'] == 'machine-verified' else []),
                                                 source_boxes=(real_boxes if m['source'] == 'borrowed' else [e['box'] for e in groups[key]][:50]),kind='synthetic',split='train',generation_review='unreviewed',sample_id='synthetic-'+hashlib.sha256(json.dumps([f,sources,target,pixel_hash],sort_keys=True).encode()).hexdigest()[:24],targets=[dict(id=target,key=canonical(m['label']),legend_id=lid,printed=m['label'])],label_ids=[target],normalized_sources=sources,native_png=native.relative_to(paths.GEN).as_posix(),normalized_png=fixed.relative_to(paths.GEN).as_posix(),valid_mask_png=valid.relative_to(paths.GEN).as_posix(),normalization=transform,pixel_sha256=pixel_hash))
        # stacked columns: needs at least two patterns in this document
        check = None
        columns = []
        stackable = {k: v for k, v in models.items()
                     if patterns[(doc_id, k)][1]['source'] in ('human', 'machine-verified')
                     and len(groups[(doc_id, k)]) >= COLUMN_MIN_BOXES}
        if len(stackable) >= 2:
            ordered = sorted((e for e in exs if e['legend_id'] in stackable), key=lambda e: e['box'][1])
            thick = [e['box'][3] - e['box'][1] for e in ordered]
            trans = defaultdict(Counter)
            for a, b in zip(ordered, ordered[1:]):
                if b['box'][1] - a['box'][3] <= 4 and a['legend_id'] != b['legend_id']:
                    trans[a['legend_id']][b['legend_id']] += 1
            for k in range(COLUMNS):
                rgb, truth = render.column(stackable, rails, thick, trans, COLUMN_ROWS, rng)
                for t in truth:
                    layer_model = patterns[(doc_id, t['legend_id'])][1]
                    t['label'] = layer_model['label']
                    t['label_id']=(layer_model['key'] if layer_model['source'] == 'unlisted' else
                                   'rock-'+hashlib.sha256(canonical(t['label']).encode()).hexdigest()[:12])
                    t['box_document_normalized']=[0,t['top']/rgb.shape[0],1,t['bottom']/rgb.shape[0]]
                f = _png(rgb, folder / ('column-%s-%02d.png' % (sig_doc[:20], k)))
                (folder / ('column-%s-%02d.json' % (sig_doc[:20], k))).write_text(
                    json.dumps(dict(image=f, document=d['name'], width=rgb.shape[1], height=rgb.shape[0],
                                    layers=truth), ensure_ascii=False), encoding='utf-8')
                columns.append(dict(file=f, layers=truth))
                manifest['columns'].append(dict(file=f, document=d['name'], layers=truth))
            # Machine evidence is training data, not human truth for this check.
            truth_groups = {k: v if patterns[k][1]['source'] == 'human' else [] for k, v in groups.items()}
            check = score.fake_label_check(stackable, {k: samples_rgb[k] for k in stackable}, self.crops, d['doc'], truth_groups)
            # a real stretch with the most distinct labelled patterns
            window = min(_best_window(ordered, COLUMN_ROWS), max(d['column'][1], d['column'][3] - COLUMN_ROWS))
        report = (dict(doc_id=doc_id, name=d['name'], patterns=len(models),
                                rails=dict(x0=rails['x0'], x1=rails['x1']),
                                columns=columns[:SHOW_COLUMNS], column_count=len(columns),
                                real_window=(window if len(stackable) >= 2 else None),
                                real_window_bottom=(min(window + COLUMN_ROWS, d['column'][3], d['doc']['height'])
                                                    if len(stackable) >= 2 else None),
                                fake_label_check=check))
        return cards, report, manifest['patterns'], manifest['columns']

    def _nearest_label(self, d, models, own, box):
        """Which other pattern of this document a suspect box resembles."""
        c, _ = learn.clean(self.crops.get(d['doc'], box))
        if c.size == 0:
            return None
        bg = learn.background(c)
        best, dist = None, 40.0
        for lid, data in models.items():
            if lid == own:
                continue
            gap = float(np.sqrt(((np.asarray(data['bg'], float) - bg) ** 2).sum()))
            if gap < dist:
                best, dist = d['legend'].get(lid, str(lid)), gap
        return best

    def _prune(self, patterns, by_doc):
        """Drop images from superseded label sets so the dataset stays current."""
        keep_sig = {(k[0], str(k[1])): v[0][:8] for k, v in patterns.items()}
        if not paths.GEN.exists():
            return
        for folder in paths.GEN.iterdir():
            if not folder.is_dir():
                continue
            if not folder.resolve().is_relative_to(paths.GEN.resolve()):
                continue
            if folder.name not in by_doc:
                shutil.rmtree(folder, ignore_errors=True)
                continue
            sig_doc = '-'.join(sorted(patterns[(folder.name, k)][0][:6] for k in by_doc[folder.name]))[:20]
            for f in folder.iterdir():
                n = f.name
                if n.startswith('column-') and not n.startswith('column-%s-' % sig_doc):
                    f.unlink(missing_ok=True)
                elif n.startswith(('p', 'tile-', 'repeat-')) and not n.startswith('column'):
                    parts = n.split('-')
                    lid = parts[0][1:] if n.startswith('p') else parts[1]
                    sig = parts[1] if n.startswith('p') else parts[2].split('.')[0]
                    if keep_sig.get((folder.name, lid)) != sig:
                        f.unlink(missing_ok=True)


def _best_window(ordered, rows):
    best, best_n = None, -1
    for i, e in enumerate(ordered):
        top = e['box'][1]
        labels = {o['legend_id'] for o in ordered[i:i + 80] if o['box'][1] < top + rows}
        if len(labels) > best_n:
            best, best_n = top, len(labels)
    return best


if __name__ == '__main__':
    s = Builder().build()
    print(json.dumps(dict(seconds=s['seconds'], patterns=s['pattern_count'], human_boxes=s['human_boxes'],
                          dataset=s['dataset'], summary=s['accuracy']['summary'],
                          full=[{k: f[k] for k in ('name', 'row_accuracy', 'rows_scored', 'boundary')}
                                for f in s['accuracy']['full']]), indent=1))
