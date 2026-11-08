"""Scores: machine guesses against human labels, and fakes against real crops."""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from . import paths
from .learn import norm, clean, _block_tile, _align, background


def load_prefill():
    rows = defaultdict(list)
    if paths.PREFILL.exists():
        with paths.PREFILL.open(encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                rows[r['log']].append(r)
    return rows


def queue_guesses():
    """Initial machine state of every queue box, before any human edit."""
    out = {}
    if paths.QUEUE.exists():
        q = json.loads(paths.QUEUE.read_text(encoding='utf-8'))
        for g in q.get('groups', []):
            for c in g.get('candidates', []):
                out[c['id']] = dict(box=c['segment']['box'], legend_ids=c['segment'].get('legend_ids', []),
                                    group=g['key'])
    return out


def seed_guess(doc_id, seg_id, cache={}):
    if doc_id not in cache:
        p = paths.SEEDS / (doc_id + '.json')
        cache[doc_id] = {s['id']: s for s in json.loads(p.read_text(encoding='utf-8'))['segments']} if p.exists() else {}
    return cache[doc_id].get(seg_id)


def _rows_truth(doc):
    """{normalised label: bool[height]} from the human-approved boxes."""
    names = {e['id']: e['label_printed'] for e in doc['legend']}
    truth = {}
    edges = set()
    for s in doc['segments']:
        if s.get('status') != 'approved':
            continue
        y0, y1 = s['box'][1], s['box'][3]
        edges.update((y0, y1))
        for lid in s['legend_ids']:
            n = norm(names.get(lid))
            truth.setdefault(n, np.zeros(doc['height'], bool))[y0:y1] = True
    pretty = {norm(v): v for v in names.values()}
    return truth, sorted(edges), pretty


def full_truth_doc(doc, prefill_rows):
    """Row-by-row score of every saved machine guess on a fully labelled doc."""
    truth, edges, pretty = _rows_truth(doc)
    if not truth:
        return None
    covered = np.zeros(doc['height'], bool)
    for arr in truth.values():
        covered |= arr
    tot = Counter(); ok = Counter()
    by_pred = defaultdict(Counter); by_true = defaultdict(Counter)
    conf = Counter(); by_tier = defaultdict(Counter); by_source = defaultdict(Counter)
    wrong = []
    medges = []
    for r in prefill_rows:
        a, b = int(r['top_y']), int(r['bottom_y'])
        if b <= a:
            continue
        medges += [a, b]
        cov = covered[a:b]
        n = int(cov.sum())
        if not n:
            continue
        pred = norm(r.get('suggestion'))
        hit = truth.get(pred, np.zeros(doc['height'], bool))[a:b] & cov
        good = int(hit.sum())
        # majority human label over these rows, for the confusion table
        votes = {k: int(v[a:b].sum()) for k, v in truth.items()}
        true = max(votes, key=votes.get)
        tot['rows'] += n; ok['rows'] += good
        by_tier[r.get('tier', '?')]['rows'] += n; by_tier[r.get('tier', '?')]['ok'] += good
        by_pred[pred]['rows'] += n; by_pred[pred]['ok'] += good
        by_true[true]['rows'] += n; by_true[true]['ok'] += good
        if good < n:
            conf[(pred, true)] += n - good
            wrong.append(dict(top=a, bottom=b, pred=pretty.get(pred, r.get('suggestion')), true=pretty.get(true, true),
                              rows=n - good, tier=r.get('tier'), reason=r.get('reason', '')[:160]))
        for s in r.get('suggestions') or []:
            src = s.get('source', '?')
            hit_s = truth.get(norm(s.get('label')), np.zeros(doc['height'], bool))[a:b] & cov
            by_source[src]['rows'] += n; by_source[src]['ok'] += int(hit_s.sum())
    medges = np.array(sorted(set(medges)))
    inner = [e for e in edges if 0 < e < doc['height']]
    dist = [int(np.abs(medges - e).min()) for e in inner] if len(medges) else []
    pct = lambda c: round(100 * c['ok'] / c['rows'], 1) if c['rows'] else None
    unguessed = int((covered & ~_mask(prefill_rows, doc['height'])).sum())
    wrong.sort(key=lambda w: -w['rows'])
    return dict(
        doc_id=doc['id'], name=doc['name'],
        rows_scored=tot['rows'], row_accuracy=pct(dict(ok=ok['rows'], rows=tot['rows'])),
        labelled_rows=int(covered.sum()), labelled_rows_with_no_guess=unguessed,
        guesses=len(prefill_rows),
        by_tier={k: dict(rows=v['rows'], accuracy=pct(v)) for k, v in by_tier.items()},
        by_source=sorted(({'source': k, 'rows': v['rows'], 'accuracy': pct(v)} for k, v in by_source.items()),
                         key=lambda d: -(d['accuracy'] or 0)),
        predicted=sorted(({'label': pretty.get(k, k), 'rows': v['rows'], 'precision': pct(v)} for k, v in by_pred.items()),
                         key=lambda d: -d['rows']),
        actual=sorted(({'label': pretty.get(k, k), 'rows': v['rows'], 'recall': pct(v)} for k, v in by_true.items()),
                      key=lambda d: -d['rows']),
        confusion=[dict(guessed=pretty.get(p, p), actual=pretty.get(t, t), rows=n) for (p, t), n in conf.most_common(10)],
        boundary=dict(count=len(dist), median_px=float(np.median(dist)) if dist else None,
                      within_5px=round(100 * float(np.mean(np.array(dist) <= 5)), 1) if dist else None,
                      within_20px=round(100 * float(np.mean(np.array(dist) <= 20)), 1) if dist else None),
        worst=wrong[:24])


def _mask(rows, height):
    m = np.zeros(height, bool)
    for r in rows:
        m[int(r['top_y']):int(r['bottom_y'])] = True
    return m


def reviewed_boxes(docs, prefill, queue):
    """Every box a human accepted or rejected outside the supplied file."""
    items = []
    for d in docs.values():
        doc = d['doc']
        names = d['legend']
        col = d['column']
        for s in doc['segments']:
            if s.get('provenance') == 'user supplied labels' or s.get('status') not in ('approved', 'rejected'):
                continue
            first = queue.get(s['id']) or seed_guess(doc['id'], s['id']) or {}
            guess_ids = first.get('legend_ids') or []
            guess_box = first.get('box')
            a, b = s['box'][1], s['box'][3]
            # what the saved row-level guesses said for these exact rows
            votes = Counter()
            for r in prefill.get(doc['name'], []):
                o = min(b, int(r['bottom_y'])) - max(a, int(r['top_y']))
                if o > 0:
                    votes[r.get('suggestion') or '(none)'] += o
            row_guess = votes.most_common(1)[0][0] if votes else None
            human = [names.get(i, str(i)) for i in s['legend_ids']]
            hn = {norm(h) for h in human}
            iou = None
            if guess_box:
                gx0, gy0, gx1, gy1 = guess_box; x0, y0, x1, y1 = s['box']
                inter = max(0, min(x1, gx1) - max(x0, gx0)) * max(0, min(y1, gy1) - max(y0, gy0))
                union = (x1 - x0) * (y1 - y0) + (gx1 - gx0) * (gy1 - gy0) - inter
                iou = round(inter / union, 3) if union else None
            items.append(dict(
                doc_id=doc['id'], doc=doc['name'], id=s['id'], status=s['status'], box=s['box'],
                guess_box=guess_box, iou=iou,
                guess=[names.get(i, str(i)) for i in guess_ids],
                guess_ok=(bool(guess_ids) and {norm(names.get(i)) for i in guess_ids} == hn) if s['status'] == 'approved' else False,
                row_guess=row_guess,
                row_guess_ok=(row_guess is not None and norm(row_guess) in hn) if s['status'] == 'approved' else False,
                human=human, comments=[c.get('text', '') for c in s.get('comments', [])][:3],
                column=col))
    items.sort(key=lambda i: (i['doc'], i['box'][1]))
    return items


# ---------------------------------------------------------------- fakes vs real
def foreground(rgb, bg=None):
    """Marks relative to the dominant background, including coloured strokes."""
    bg = background(rgb) if bg is None else np.asarray(bg)
    return np.linalg.norm(rgb.astype(float)-bg,axis=2)>35


def fidelity(rgb, data):
    from .candidates import pixel_match, symbol_match
    pixel = pixel_match(rgb,data)
    bg = background(rgb)
    blank = float((np.linalg.norm(rgb.astype(float)-bg,axis=2)<45).mean())
    dice = symbol_match(rgb,data)
    has_marks = bool(foreground(rgb,bg).any())
    # Colour-weighted foreground prevents large backgrounds approving blanks.
    gain = pixel-blank
    occupancy = float(foreground(data['tile'],data['bg']).mean())
    actual = float(foreground(rgb,bg).mean())
    ratio = occupancy/max(actual,1e-9)
    acceptable = pixel>=.85 and (not has_marks or (dice>=.8 and gain>.005 and .5<=ratio<=2))
    return dict(pixel=pixel,blank_baseline=blank,blank_improvement=gain,
                foreground_dice=dice,foreground_ratio=ratio,acceptable=bool(acceptable))


def component_profile(rgb):
    """Ink-weighted geometry prevents a few large marks hiding in a median."""
    from scipy.ndimage import label,find_objects
    ids,n=label(foreground(rgb),structure=np.ones((3,3)))
    spans=[]
    for i,sl in enumerate(find_objects(ids),1):
        if sl is None:continue
        area=int((ids[sl]==i).sum())
        if area>=2:spans.append([sl[0].stop-sl[0].start,sl[1].stop-sl[1].start,area])
    a=np.asarray(spans,float).reshape(-1,3)
    def quantile(axis,q):
        if not len(a):return 0.
        order=np.argsort(a[:,axis]);v=a[order]
        return float(v[min(len(v)-1,np.searchsorted(np.cumsum(v[:,2]),q*v[:,2].sum())),axis])
    return dict(components=a.tolist(),median=np.median(a[:,:2],0).tolist() if len(a) else [0,0],
                height_q95=quantile(0,.95),weighted_quantiles=[[quantile(axis,q) for q in (.25,.5,.75,.95,.99)] for axis in (0,1)])


def component_comparison(reference,generated):
    ref=reference if isinstance(reference,dict) else component_profile(reference)
    got=generated if isinstance(generated,dict) else component_profile(generated)
    a=np.asarray(ref['components'],float).reshape(-1,3);b=np.asarray(got['components'],float).reshape(-1,3)
    if not len(a) or not len(b):
        return dict(acceptable=bool(len(a)==len(b)),median_error=0 if len(a)==len(b) else 1,
                    upper_tail_ink_excess=[0,0] if len(a)==len(b) else [1,1],weighted_errors=[0,0] if len(a)==len(b) else [1,1])
    median_error=float(np.max(np.abs(np.array(got['median'])-ref['median'])/np.maximum(1,ref['median'])))
    excess=[];errors=[]
    for axis in (0,1):
        cap=max(v[axis] for v in a)
        excess.append(float(b[b[:,axis]>cap,2].sum()/b[:,2].sum()))
        rq=np.array(ref['weighted_quantiles'][axis]);gq=np.array(got['weighted_quantiles'][axis])
        errors.append(float(np.max(np.abs(gq-rq)/np.maximum(1,rq))))
    # Count medians remain diagnostic: many tiny clipped edge marks can change
    # them while carrying almost no ink. Admission uses the ink distribution.
    return dict(acceptable=bool(excess[0]<=.1 and excess[1]<=.1 and errors[0]<=.35 and errors[1]<=.5),
                median_error=median_error,upper_tail_ink_excess=excess,weighted_errors=errors,
                reference_weighted_quantiles=ref['weighted_quantiles'],generated_weighted_quantiles=got['weighted_quantiles'])


def realism(model, data, crops, doc, examples, limit=24, reference_kind='target column'):
    """Pixel match of the learned tile against held-out real crops.

    Each real crop is aligned to the tile (its own phase), then compared
    pixel by pixel: share of pixels within a colour distance of 45.
    """
    if data is None:
        return dict(status='unavailable',reference_kind=reference_kind,median=None,crops=0,held_out=0)
    if data['kind'] != 'tile':
        from . import render
        rail=model.get('validation_rails')
        if rail is None:
            fitting=[e for e in examples if e['box'] not in model.get('heldout_boxes',[])] or examples
            rail=__import__('genlab.learn',fromlist=['rails']).rails(crops,{doc['id']:dict(doc=doc)},doc['id'],fitting)
        x0,x1=rail['x0'],rail['x1'];width=x1-x0
        windows=[render.frame(render.layer(data,x0,x1,0,240,np.random.default_rng(seed)),rail) for seed in range(4)]
        checks=[realism(model,dict(kind='tile',tile=rgb,bg=background(rgb)),crops,doc,examples,limit,reference_kind)
                for rgb in windows]
        measured=[v for v in checks if v.get('median') is not None]
        if not measured:return checks[0]
        result=dict(measured[0])
        result.update(windows_tested=4,window_scores=checks,
            median=round(float(np.median([v['median'] for v in measured])),3),
            worst=min(v['worst'] for v in measured),
            foreground_dice=round(float(np.median([v['foreground_dice'] for v in measured])),4))
        # A generous colour overlap cannot certify repeated dash/dot geometry.
        refs=[clean(crops.get(doc,b))[0] for b in result['boxes']]
        ref_profiles=[component_profile(a) for a in refs]
        window_profiles=[component_profile(clean(a)[0]) for a in windows]
        comparisons=[component_comparison(ref,got) for ref in ref_profiles for got in window_profiles]
        bank_checks=[];extent_ok=True
        if data['kind']=='quilt':
            origins=data.get('strip_origins',[x0 if data.get('quilt_local') else data.get('x_origin',x0)]*len(data['strips']))
            extent_ok='strip_origins' in data
            interior=np.flatnonzero(~np.asarray(rail['mask'],bool))+x0
            for index,(strip,origin) in enumerate(zip(data['strips'],origins)):
                supported=bool('strip_origins' in data and (not len(interior) or (origin<=interior.min() and origin+strip.shape[1]>interior.max())))
                extent_ok &= supported
                starts=sorted(set(range(0,max(1,len(strip)-239),120))|{max(0,len(strip)-240)})
                checks=[]
                for top in starts:
                    bank=dict(data,strips=[strip[top:top+240]],strip_origins=[origin])
                    a=render.frame(render.layer(bank,x0,x1,0,min(240,len(strip)),np.random.default_rng(0)),rail)
                    profile=component_profile(clean(a)[0])
                    details=[component_comparison(ref,profile) for ref in ref_profiles]
                    checks.append(dict(top=top,acceptable=all(d['acceptable'] for d in details),comparisons=details))
                bank_checks.append(dict(member=index,supported_extent=supported,windows=checks,acceptable=bool(supported and all(c['acceptable'] for c in checks))))
        errors=[c['median_error'] for c in comparisons]
        shape_ok=all(c['acceptable'] for c in comparisons) and all(c['acceptable'] for c in bank_checks) and extent_ok
        result.update(morphology_relative_errors=errors,
            morphology_acceptable=bool(shape_ok and all(v['foreground_dice']>=.9 for v in measured)),
            acceptable=bool(all(v.get('acceptable') for v in measured) and shape_ok and
                            all(v['foreground_dice']>=.9 for v in measured)),
            bank_morphology=bank_checks,component_comparisons=comparisons,scored_width=width,emitted_width=width,
            supported_native_extent=bool(extent_ok),
            validation_scope='Four seeded native-width framed windows and every bank member at 120-row steps; finite height only')
        result['clean_texture_fidelity']=dict(median=result['median'],foreground_dice=result['foreground_dice'])
        return result
    tile = data['tile']
    py, px = tile.shape[:2]
    used = set(model.get('sources', [])) | {x['id'] for x in model.get('suspects', [])}
    training_boxes = model.get('training_boxes') or [e['box'] for e in examples if e['id'] in used]
    from .learn import boxes_overlap
    held = [e for e in examples if e['id'] not in used and
            (not model.get('heldout_ids') or e['id'] in model['heldout_ids']) and
            not any(boxes_overlap(e['box'],b,max(py,model.get('evidence_guard_rows',py))) for b in training_boxes)]
    independent = bool(held)
    held = held or [e for e in examples if e['id'] in model.get('sources', [])] or examples
    held = sorted(held, key=lambda e: -(e['box'][3] - e['box'][1]))[:limit]
    scores, details = [], []
    distinct = []
    for e in held:
        if any(boxes_overlap(e['box'],b) for b in distinct):
            continue
        c, _ = clean(crops.get(doc, e['box']))
        if c.size:
            detail = fidelity(c,data)
            details.append(detail)
            scores.append(detail['pixel'])
            distinct.append(e['box'])
    if not scores:
        return dict(status='unavailable',reference_kind=reference_kind,median=None,crops=0,held_out=0)
    return dict(held_out=len(scores) if independent else 0, crops=len(scores), reused_training=not independent,
                median=round(float(np.median(scores)), 3),
                worst=round(float(np.min(scores)), 3),status='measured',reference_kind=reference_kind,
                independent_depth_ranges=len(scores) if independent else 0,
                foreground_dice=round(float(np.median([v['foreground_dice'] for v in details])),4),
                blank_baseline=round(float(np.median([v['blank_baseline'] for v in details])),4),
                blank_improvement=round(float(np.median([v['blank_improvement'] for v in details])),4),
                acceptable=all(v['acceptable'] for v in details),
                clean_texture_fidelity=dict(median=round(float(np.median(scores)),3),foreground_dice=round(float(np.median([v['foreground_dice'] for v in details])),4)),
                scan_appearance_fidelity=dict(status='not independently measured',reason='Cleaned reference similarity does not measure scan artifacts.'),
                boxes=distinct)


def coverage(docs, reports, prefill, patterns, proposal_rows=()):
    """Entry counts and row weights from saved machine guesses, never truth."""
    generated = set(patterns)
    rows = {}
    names = {norm(r['name']): r['doc_id'] for r in reports}
    names.update({norm(d['name']):did for did,d in docs.items() if d.get('name')})
    from .learn import generation_split
    training = {did for did, d in docs.items() if
                (generation_split(d.get('doc',d))[1]=='train' if 'id' in d.get('doc',d)
                 else bool(d.get('training_allowed',False)))}
    legend_keys = {(did, lid) for did, d in docs.items() for lid in d['legend']}
    counts = dict(all_entries=0, training_entries=0, generated_entries=len(generated & legend_keys),
                  generated_patterns=len(generated),
                  column_rows=0, generated_column_rows=0, training_column_rows=0,
                  training_generated_column_rows=0, basis='estimated from machine guesses')
    labels = {}
    for did, d in docs.items():
        for lid, label in d['legend'].items():
            key = norm(label)
            r = rows.setdefault(key, dict(label=label, normalized_label=key, generated_entries=0,
                                           all_entries=0, estimated_column_rows=0, generated_column_rows=0))
            r['all_entries'] += 1
            r['generated_entries'] += (did, lid) in generated
            counts['all_entries'] += 1
            counts['training_entries'] += did in training
            labels.setdefault((did, key), set()).add((did, lid))
    entry_weights = Counter()
    for name, guesses in prefill.items():
        did = names.get(norm(name))
        for guess in guesses:
            weight = max(0, int(guess['bottom_y'])-int(guess['top_y']))
            label = norm(guess.get('suggestion'))
            covered = bool(labels.get((did, label), set()) & generated)
            matches = labels.get((did,label),set())
            for key in matches:
                entry_weights[key] += weight/len(matches)
            counts['column_rows'] += weight
            counts['generated_column_rows'] += weight * covered
            if did in training:
                counts['training_column_rows'] += weight
                counts['training_generated_column_rows'] += weight * covered
            if label in rows:
                rows[label]['estimated_column_rows'] += weight
                rows[label]['generated_column_rows'] += weight * covered
    for r in rows.values():
        r['estimated_share_of_column_rows'] = r['estimated_column_rows'] / max(1, counts['column_rows'])
    counts['estimated_row_coverage'] = counts['generated_column_rows'] / max(1, counts['column_rows'])
    counts['training_estimated_row_coverage'] = counts['training_generated_column_rows'] / max(1, counts['training_column_rows'])
    counts['entries'] = [dict(doc_id=did,legend_id=lid,label=label,
                              estimated_column_rows=entry_weights[(did,lid)],
                              share_of_column_rows=entry_weights[(did,lid)]/max(1,counts['training_column_rows']) if did in training else None,
                              training_allowed=did in training,
                              occurrence='machine guessed' if entry_weights[(did,lid)] else 'unknown occurrence',
                              generated=(did,lid) in generated,
                              column_validated=bool(isinstance(patterns,dict) and (did,lid) in patterns and patterns[(did,lid)][1].get('column_validated')))
                         for did,d in docs.items() for lid,label in d['legend'].items()]
    counts['validated_column_entries'] = sum(e['column_validated'] for e in counts['entries'])
    counts['provisional_generated_entries'] = sum(e['generated'] and not e['column_validated'] for e in counts['entries'])
    counts['labels'] = sorted(rows.values(), key=lambda r: (-r['estimated_column_rows'], r['normalized_label']))
    # Unlisted descriptions are separate classes, not extra legend entries or
    # aliases for looks_like. Their fleet row shares are reported separately:
    # adding them to guessed legend coverage could count the same rows twice.
    unlisted = []
    for p in proposal_rows:
        if p['source'] != 'unlisted' or p.get('status') != 'made':
            continue
        key = (p['doc_id'], p['legend_id'])
        unlisted.append(dict(label=p['label'], pattern_id=f'{key[0]}:{key[1]}',
                             doc_id=key[0], generated_entries=int(key in generated), all_entries=1,
                             training_entries=int(key[0] in training), looks_like=p.get('looks_like'),
                             estimated_share_of_document_rows=p.get('share_of_column_rows')))
    counts['unlisted'] = unlisted
    counts['unlisted_entries'] = len(unlisted)
    counts['training_unlisted_entries'] = sum(r['training_entries'] for r in unlisted)
    counts['generated_unlisted_entries'] = sum(r['generated_entries'] for r in unlisted)
    if isinstance(patterns, dict):
        counts['by_source'] = dict(Counter(v[1]['source'] for v in patterns.values()))
    return counts


def reviewer_accuracy(rated, manifest):
    """One latest owner decision per strip; clear/unknown votes do not count."""
    current = {p['file']: p for p in manifest}
    sources = ('agent-made', 'unlisted')
    methods = ('column', 'swatch', 'reconstructed', 'mixed')
    counts = Counter()
    for file, rating in rated.items():
        if rating.get('verdict') not in ('good', 'bad'):
            continue
        provenance = current.get(file, rating)
        source, method = provenance.get('source'), provenance.get('method') or 'unknown'
        if source in sources:
            counts[source, method, rating['verdict']] += 1

    def row(source=None, method=None):
        good = sum(n for (s, m, v), n in counts.items()
                   if (source is None or s == source) and (method is None or m == method) and v == 'good')
        bad = sum(n for (s, m, v), n in counts.items()
                  if (source is None or s == source) and (method is None or m == method) and v == 'bad')
        return dict(source=source, method=method, real=good, wrong=bad, total=good+bad,
                    percent_real=round(100*good/(good+bad), 1) if good+bad else None)

    if any(m == 'unknown' for _, m, _ in counts):
        methods += ('unknown',)
    return dict(total=row(), by_source=[row(source=s) for s in sources],
                by_method=[row(method=m) for m in methods],
                by_source_method=[row(s, m) for s in sources for m in methods])


def features(rgb):
    q = (rgb // 64).reshape(-1, 3).astype(int)
    hist = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64).astype(float)
    hist /= hist.sum() or 1
    ink = (rgb.max(2) < 80).mean(1)
    ink = ink - ink.mean()
    ac = np.array([float((ink[:-k] * ink[k:]).mean()) if len(ink) > 2 * k else 0. for k in range(1, 41)])
    ac /= (np.abs(ac).max() or 1)
    # colour mix decides; the repeat profile only breaks near-ties
    return np.concatenate([hist * 10, ac * .15])


def fake_label_check(doc_models, doc_samples, crops, doc, groups):
    """Train on fakes only, then label the human's real boxes. Per document."""
    cents = {}
    for lid, samples in doc_samples.items():
        cents[lid] = np.mean([features(s) for s in samples], axis=0)
    if len(cents) < 2:
        return None
    total = good = 0
    confusion = Counter()
    for lid in cents:
        exs = sorted(groups[(doc['id'], lid)], key=lambda e: -(e['box'][3] - e['box'][1]))[:40]
        for e in exs:
            c, _ = clean(crops.get(doc, e['box']))
            if c.shape[0] < 8:
                continue
            f = features(c)
            pred = min(cents, key=lambda k: float(np.square(cents[k] - f).sum()))
            total += 1; good += pred == lid
            if pred != lid:
                confusion[(pred, lid)] += 1
    return dict(real_boxes=total, accuracy=round(100 * good / total, 1) if total else None,
                confusion=[dict(fake=a, real=b, n=n) for (a, b), n in confusion.most_common(5)])
