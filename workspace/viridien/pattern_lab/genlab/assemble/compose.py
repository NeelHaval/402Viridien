"""Assemble training-only synthetic lithology columns and banded real pages.

python -X utf8 -m genlab.assemble.compose column --doc ID --n 1 --seed 1 --height 20000
python -X utf8 -m genlab.assemble.compose page --doc ID --n 1 --seed 1
python -X utf8 -m genlab.assemble.compose starter --seed 101 --height 20000
"""
from __future__ import annotations

from genlab.assemble import training_v2
import argparse
import hashlib
import json
import re
import time
import shutil
import copy
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import model
from genlab import paths, render
from genlab.learn import norm
from . import noise

OUT = noise.ROOT / 'out'
STATS = noise.ROOT / 'stats.json'
GAP = '__gap__'


def probability(counts):
    total = sum(counts.values())
    return {k: v/total for k, v in sorted(counts.items())} if total else {}


def distribution(values):
    a = np.asarray(values, float)
    return dict(dist='lognormal', mu=float(np.log(a).mean()), sigma=float(np.log(a).std()),
                quantiles_px=np.quantile(a, [.05, .25, .5, .75, .95]).tolist(), n=len(a))


def load_stats(path=None):
    path = Path(path) if path else STATS
    if not path.exists():
        raise ValueError('Build asm-2 training statistics before generation')
    stats = noise.read(path)
    if stats.get('schema') != 'asm-2':
        raise ValueError('Expected stats schema asm-2; rebuild asm-1 statistics')
    return stats, path


def native_geometry(real,fake):
    """Measure course scale and ink without resizing either native image."""
    a=real[:,2:-2].max(2)<100;b=fake[:,2:-2].max(2)<100
    ink_a=float(a.mean());ink_b=float(b.mean());period=None;mismatch=None;real_mismatch=None
    if ink_a>.005:
        for lag in range(3,min(160,len(a)//2)+1):
            error=float(np.mean(a[:-lag]!=a[lag:]))
            baseline=2*ink_a*(1-ink_a)
            if error<=min(.01,max(1e-7,.25*baseline)):
                period=lag;real_mismatch=error;mismatch=float(np.mean(b[:-lag]!=b[lag:]));break
    from scipy.ndimage import distance_transform_edt
    def stroke(mask):
        return float(np.quantile(distance_transform_edt(mask)[mask],.9)*2) if mask.any() else 0.
    sa,sb=stroke(a),stroke(b)
    # Compare nonblack colours, including coloured motifs, without rail pixels.
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import jensenshannon
    ra=real[:,2:-2][~a];fb=fake[:,2:-2][~b]
    colour_error=0.;centre_error=0.;palette_js=0.
    if len(ra) and len(fb):
        palette=np.unique(ra.reshape(-1,3),axis=0)
        distances=cKDTree(palette.astype(float)).query(fb.astype(float),p=np.inf)[0]
        colour_error=float(np.mean(distances>16))
        centre_error=float(np.linalg.norm(np.median(ra,axis=0)-np.median(fb,axis=0)))
        def hist(pixels):
            q=pixels.astype(np.int32)//16
            return np.bincount(q[:,0]*256+q[:,1]*16+q[:,2],minlength=4096)
        palette_js=float(jensenshannon(hist(ra),hist(fb),base=2)**2)
    elif len(ra)!=len(fb):colour_error=1.;centre_error=442.
    palette_ok=colour_error<=.05 and centre_error<=20
    repeat_limit=None if mismatch is None else real_mismatch+max(.003,.25*ink_a)
    return dict(eligible=palette_ok and abs(ink_a-ink_b)<=max(.03,.35*ink_a) and abs(sa-sb)<=max(1.5,.35*sa)
        and (mismatch is None or mismatch<=repeat_limit),real_repeat_rows=period,
        real_mismatch_at_repeat=real_mismatch,repeat_mismatch_limit=repeat_limit,
        palette_eligible=bool(palette_ok),palette_outside_fraction=colour_error,
        colour_centre_distance=centre_error,palette_js_bits=palette_js,
        generated_mismatch_at_real_repeat=mismatch,real_ink_fraction=ink_a,generated_ink_fraction=ink_b,
        real_stroke_px=sa,generated_stroke_px=sb)


class Sampler:
    def __init__(self, stats, doc_id, family, available, seed, zone_scheme=None):
        self.stats, self.doc_id, self.family = stats, doc_id, family
        self.available = available  # target class ID -> exact/same-name generated pattern
        self.rng = np.random.default_rng(seed)
        self.used = Counter()
        self.projected = set()
        self.levels = [(k, stats['levels'][k]) for k in ('doc:'+doc_id, 'family:'+family, 'global')
                       if k in stats['levels']]
        if not available or not self.levels:
            raise ValueError('No generated classes or no training statistics')
        self.zone_scheme=zone_scheme or self.levels[0][1].get('zone_scheme','all')
        self.mapping=self.levels[0][1].get('depth_mapping',{})
        for zone,comp in self.levels[0][1].get('composition',{}).items():
            if zone!='all' and not zone.startswith(self.zone_scheme+'-'):
                continue
            missing={k:p for k,p in comp.items() if k not in available and p>0}
            if any(p>.1 for p in missing.values()) or sum(missing.values())>.05+1e-9:
                raise ValueError(f'Unsupported composition mass in {zone}: {sum(missing.values()):.3f}; representative minimum 0.95')

    def key(self, cid, level):
        return cid if level.startswith('doc:') else self.stats['classes'][cid]['norm']

    def zone(self, frac):
        # Document sections never inherit pooled ranks or silently lose to fifths.
        for level, block in self.levels[:1]:
            for z in block.get('zones', []):
                if self.zone_scheme!='all' and not z['name'].startswith(self.zone_scheme+'-'):
                    continue
                if self.zone_scheme=='all':
                    continue
                if z['top_frac'] <= frac < z['bottom_frac'] or frac == 1 == z['bottom_frac']:
                    return z['name']
        return 'all'

    def candidates(self, table, level):
        groups = defaultdict(list)
        for cid in self.available:
            groups[self.key(cid, level)].append(cid)
        return {cid: float(table[k])/len(ids) for k, ids in groups.items() if float(table.get(k, 0)) > 0
                for cid in ids}

    def marginalize_missing(self, table, level, zone):
        """Skip withheld states without turning their predecessors into traps.

        Integrate their measured order-one successor probabilities; draw no pixels
        for them. The truth records every skipped intermediate state explicitly.
        """
        supported = {self.key(cid,level) for cid in self.available}
        comp=self.stats['levels'][level].get('composition',{}).get('all',{})
        if any(comp.get(k,0)>.1 for k in table if k not in supported):
            return {}
        transitions = self.stats['levels'][level].get('transitions',{})
        result = Counter()
        pending = dict(table)
        for _ in range(24):
            following = Counter()
            for key,p in pending.items():
                if key in supported:
                    result[key] += p
                else:
                    self.projected.add(level+'/'+key)
                    row = transitions.get('by_zone',{}).get(zone,{}).get(key)
                    row = row or transitions.get('order1',{}).get(key,{})
                    for nxt,q in row.items():
                        following[nxt] += p*q
            pending = following
            if sum(pending.values()) < 1e-10:
                break
        return dict(result)

    def next(self, history, zone):
        # Initial entry probabilities are bed probabilities, not row prevalence.
        if not history:
            for level,block in self.levels:
                chain=block.get('transitions',{}).get('order1',{})
                if not chain:
                    continue
                comp=block.get('composition',{}).get(zone) or block.get('composition',{}).get('all',{})
                probs=self.candidates(comp,level)
                probs={cid:p/self.expected_thickness(cid,zone) for cid,p in probs.items()
                       if chain.get(self.key(cid,level))}
                if probs:
                    return self.choose(level,'initial-bed-probability',probs)
            raise ValueError('No transition evidence; iid composition fallback forbidden')
        for level, block in self.levels:
            t = block.get('transitions', {})
            keys = [self.key(c, level) for c in history[-2:]]
            options = []
            order=t.get('selected_order',{}).get('order',1)
            if len(keys) == 2 and order==2:
                pair = '|'.join(keys)
                options.extend([('order2_by_zone',t.get('order2_by_zone',{}).get(zone,{}).get(pair,{}),
                    t.get('counts_order2_by_zone',{}).get(zone,{}).get(pair,{})),
                    ('order2',t.get('order2',{}).get(pair,{}),t.get('counts_order2',{}).get(pair,{}))])
            options.extend([('by_zone',t.get('by_zone',{}).get(zone,{}).get(keys[-1],{}),
                             t.get('counts_by_zone',{}).get(zone,{}).get(keys[-1],{})),
                            ('order1',t.get('order1',{}).get(keys[-1],{}),t.get('counts_order1',{}).get(keys[-1],{}))])
            for kind,table,counts in options:
                n=sum(counts.values()) if counts else (block.get('n_layers',0) if kind=='order1' else 0)
                if n<=0 or (kind!='order1' and n<5):
                    continue
                # Use the same shrinkage family evaluated during order selection.
                if kind.startswith('order2'):
                    prior=t.get('order1',{}).get(keys[-1],{})
                    names=set(table)|set(prior)
                    table={k:(counts.get(k,0)+10*prior.get(k,0))/(n+10) for k in names}
                elif kind=='by_zone':
                    prior=t.get('order1',{}).get(keys[-1],{})
                    names=set(table)|set(prior)
                    table={k:(counts.get(k,0)+5*prior.get(k,0))/(n+5) for k in names}
                probs=self.candidates(self.marginalize_missing(table,level,zone),level)
                if probs:
                    return self.choose(level,kind if n>=5 else kind+'-low-support',probs)
        raise ValueError('No supported transition successor; iid composition fallback forbidden')

    def choose(self, level, field, probs):
        keys = sorted(probs)
        p = np.array([probs[k] for k in keys], float)
        self.used[level+'/'+field] += 1
        return keys[int(self.rng.choice(len(keys), p=p/p.sum()))]

    def thickness(self, cid, zone):
        level,field,d=self.duration_model(cid,zone)
        return self.draw_thickness(level,field,d)

    def duration_model(self,cid,zone):
        level,field,d=self.base_duration_model(cid,zone)
        profile=self.stats.get('bench_duration_profile')
        if not profile:return level,field,d
        if profile.get('schema')!='bench-duration-1' or profile.get('mode')!='thin-beds':
            raise ValueError('Unsupported bench duration profile')
        caps=profile.get('caps_by_class',{}).get(cid,{})
        cap=caps.get(zone,caps.get('all'))
        if not isinstance(cap,int) or isinstance(cap,bool) or cap<1:
            raise ValueError('Thin-bed profile has no positive integer cap for '+cid+'/'+zone)
        values,weights=self.duration_distribution(level,d)
        converted=dict(samples_px=np.minimum(values,cap).tolist(),sample_weights=weights.tolist(),
            n=len(values),complete_n=len(values),bench_duration_cap=cap,bench_source_level=level,
            fallback_level=d.get('fallback_level',level),shrinkage_local_weight=d.get('shrinkage_local_weight'))
        return 'doc:'+self.doc_id,'bench-duration-1/thin-beds',converted

    def base_duration_model(self,cid,zone):
        sparse=None
        for level, block in self.levels:
            if level=='global':
                continue  # a shared name is not a compatible printed appearance family
            key = self.key(cid, level)
            for field, table in [('thickness_by_zone', block.get('thickness_by_zone', {}).get(zone, {})),
                                 ('thickness', block.get('thickness', {}))]:
                d = table.get(key)
                if not d or d.get('complete_n',d.get('n',0))<1:
                    continue
                compatible=level.startswith('doc:') or (self.mapping.get('rows_per_ft') and
                    d.get('samples_ft_by_coordinate',{}).get(self.mapping.get('coordinate')))
                if not compatible:
                    continue
                values,weights=self.duration_distribution(level,d)
                support=d.get('effective_n',d.get('complete_n',d.get('n',0)))
                if not level.startswith('doc:'):support=float(weights.sum())
                if level.startswith('doc:') and support<5:
                    if sparse is None:sparse=(values,weights,support)
                    continue
                if not level.startswith('doc:') and (support<5 or len(np.unique(values))<2):
                    continue
                if sparse is not None:
                    local,lw,n=sparse
                    alpha=n/(n+5.)
                    mixed=dict(samples_px=np.r_[local,values].tolist(),
                        sample_weights=np.r_[alpha*lw/lw.sum(),(1-alpha)*weights/weights.sum()].tolist(),
                        n=len(local)+len(values),complete_n=len(local)+len(values),
                        fallback_level=level,shrinkage_local_weight=alpha)
                    return 'doc:'+self.doc_id,'thickness-family-shrinkage',mixed
                return level,field,d
        raise ValueError('No compatible complete-bed thickness evidence for '+cid)

    def duration_values(self,level,d):
        if not level.startswith('doc:'):
            return np.asarray(d['samples_ft_by_coordinate'][self.mapping['coordinate']])*self.mapping['rows_per_ft']
        return np.asarray(d.get('samples_px') or d['quantiles_px'],float)

    def duration_distribution(self,level,d):
        values=self.duration_values(level,d)
        if level.startswith('doc:'):
            weights=d.get('sample_weights')
            # Old quantile-only fixtures explicitly represent equal-probability quantiles.
            if weights is None and not d.get('samples_px'):weights=np.ones(len(values))
        else:
            weights=d.get('sample_weights_by_coordinate',{}).get(self.mapping['coordinate'])
        if weights is None or len(weights)!=len(values):
            raise ValueError('Duration values/weights mismatch; rebuild coordinate-aware statistics')
        weights=np.asarray(weights,float)
        if not len(values) or not np.isfinite(values).all() or (values<=0).any() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
            raise ValueError('Invalid duration distribution')
        return values,weights

    def expected_thickness(self,cid,zone):
        level,_,d=self.duration_model(cid,zone)
        values,weights=self.duration_distribution(level,d)
        return max(1.,float(np.average(values,weights=weights)))

    def draw_thickness(self, level, field, d):
        self.used[level+'/'+field] += 1
        values,weights=self.duration_distribution(level,d)
        p=weights/weights.sum()
        h=self.rng.choice(values,p=p)
        return max(1, int(round(h)))

    def gap(self, thickness):
        choices = []
        for level, block in self.levels:
            if level == 'global':
                continue
            if block.get('gaps',{}).get('verified'):
                choices.append((level, block))
                if block.get('confidence', {}).get('gaps', block.get('confidence', {}).get('composition')) in ('high', 'medium'):
                    break
        if not choices:
            return 0
        level, block = choices[-1]
        d = block['gaps']
        if self.rng.random() >= 1-np.exp(-d.get('rate_per_10k_rows', 0)*thickness/10000):
            return 0
        self.used[level+'/gaps'] += 1
        q = d.get('height_quantiles_px', [1]*5)
        return max(1, int(round(np.interp(self.rng.random(), np.linspace(0,1,len(q)), q))))

    def sequence(self, height, max_layers=None, start_frac=0., span_frac=1., clip_to_zone=False,
                 start_at_boundary=None):
        result, y, history = [], 0, []
        arbitrary=(start_frac>0) if start_at_boundary is None else not start_at_boundary
        while y < height and (max_layers is None or len(result) < max_layers):
            zone = self.zone(min(1.,start_frac+span_frac*y/height))
            age=0
            if not history and y==0 and arbitrary:
                level,block=self.levels[0]
                probs=self.candidates(block.get('composition',{}).get(zone) or block['composition']['all'],level)
                cid=self.choose(level,'initial-row-occupancy',probs)
                level,field,d=self.duration_model(cid,zone)
                values,weights=self.duration_distribution(level,d)
                durations=np.maximum(1,np.rint(values).astype(int))
                sampled=int(self.rng.choice(durations,p=weights*durations/np.sum(weights*durations)))
                age=int(self.rng.integers(sampled))
            else:
                cid = self.next(history, zone)
                sampled=self.thickness(cid, zone)
            remaining=sampled-age
            h = min(remaining, height-y)
            # A conditioning bin never supplies an observed bed boundary. A bed
            # and transition history continue across it until the sampled bed ends.
            item=dict(top=y, bottom=y+h, class_id=cid,
                norm=self.stats['classes'][cid]['norm'], pattern_source=self.available[cid]['pattern_source'], zone=zone,
                sampled_duration_rows=sampled,sampled_age_rows=age,
                top_censored=bool(y==0 and arbitrary),bottom_censored=remaining>height-y,
                boundary_top_confirmed=not (y==0 and arbitrary),boundary_bottom_confirmed=remaining<=height-y,
                boundary_basis='constructed sequence; viewport cuts censored')
            if result and result[-1]['class_id']==cid and result[-1]['bottom']==y:
                if self.stats.get('bench_duration_profile'):
                    _,_,d=self.duration_model(cid,zone)
                    if y+h-result[-1]['top']>d['bench_duration_cap']:
                        raise ValueError('Thin-bed profile would exceed cap after adjacent equal-class merge')
                result[-1]['bottom']=y+h
                result[-1]['sampled_duration_rows']+=sampled
                result[-1]['bottom_censored']=item['bottom_censored']
                result[-1]['boundary_bottom_confirmed']=item['boundary_bottom_confirmed']
            else:
                result.append(item)
            y += h
            history.append(cid)
            gap = min(self.gap(h), height-y)
            if gap:
                result.append(dict(top=y, bottom=y+gap, class_id=GAP, norm=GAP,
                                   pattern_source='unlabelled-gap', zone=self.zone(min(1.,start_frac+span_frac*y/height))))
                y += gap
                history = []
        return result


class Composer:
    def __init__(self, stats=None, stats_path=None, realism=None,state_path=None,noise_path=None):
        if stats is None:
            self.stats, self.stats_path = load_stats(stats_path)
        else:
            self.stats, self.stats_path = stats, Path(stats_path or '<provided>')
        self.stats_hash=hashlib.sha256(self.stats_path.read_bytes() if self.stats_path.exists() else
            json.dumps(self.stats,sort_keys=True).encode()).hexdigest()
        self.store = model.Store()
        self.docs = {d['id']: self.store.get(d['id']) for d in self.store.catalog if training_v2.training_id(d['id'])}
        for cls in self.stats['classes'].values():
            if cls['doc_id'] in self.docs:noise.require_training(self.docs[cls['doc_id']])
        context_file=Path(state_path).parent/'context.json' if state_path else None
        self.frozen_context=bool(context_file and context_file.exists())
        self.model_root=Path(state_path).parent if self.frozen_context else None
        index=noise.read(paths.DATA/'genlab/fasttrack/training-v2.json')
        index=dict(index,records=[r for r in index['records'] if r['document_id'] in self.docs and noise.training(self.docs[r['document_id']])])
        self.context=noise.read(context_file) if self.frozen_context else dict(
            normalized=index,documents={d:doc for d,doc in self.docs.items() if noise.training(doc)})
        for did,doc in self.context['documents'].items():
            if self.docs.get(did)!=doc:raise ValueError('Archived document geometry/revision changed: '+did)
        if self.context['normalized'].get('snapshot')!=self.stats.get('inputs',{}).get('normalized_snapshot'):
            raise ValueError('Statistics/context label snapshot mismatch; rebuild statistics')
        if index.get('snapshot')!=self.context['normalized'].get('snapshot'):
            raise ValueError('Current/frozen normalized snapshot mismatch; revalidate archived sources')
        self.context_hash=hashlib.sha256(json.dumps(self.context,sort_keys=True).encode()).hexdigest()
        stats_docs = {c['doc_id'] for c in self.stats['classes'].values()}
        stats_docs.update(did for block in self.stats['levels'].values() for did in block.get('document_ids',[]))
        for did in stats_docs:
            if did not in self.docs:
                raise ValueError('Unknown stats contributor: '+did)
            noise.require_training(self.docs[did])
        if self.stats.get('schema')!='asm-2':
            raise ValueError('Expected stats schema asm-2; rebuild all priors')
        policy_path=paths.DATA/'normalized/split-policy.json'
        if self.stats.get('inputs',{}).get('split_policy_sha256')!=hashlib.sha256(policy_path.read_bytes()).hexdigest():
            raise ValueError('Missing or changed split policy provenance; rebuild statistics')
        policy=noise.read(policy_path)
        doclevels={k[4:] for k in self.stats['levels'] if k.startswith('doc:')}
        if set(self.stats['levels'].get('global',{}).get('document_ids',[]))!=doclevels:
            raise ValueError('Aggregate contributor provenance does not match document levels')
        for block in self.stats['levels'].values():
            expected={policy['document_groups'][d] for d in block.get('document_ids',[])}
            if set(block.get('source_wells',[]))!=expected:
                raise ValueError('Missing or inconsistent contributor-well provenance')
        self.fmap = noise.families()
        cards={c['key']:c for c in noise.read(state_path or paths.STATE)['cards']}
        self.cards={}
        self.rejected_cards={}
        for key,card in cards.items():
            try:
                parents=noise.ancestors(card,cards)
                from genlab.assemble import learn_v2 as learn
                if not learn.lineage_allowed(card,self.docs):raise ValueError('Revoked declared lineage')
            except ValueError as error:
                self.rejected_cards[key]=str(error)
                continue
            if all(d in self.docs and noise.training(self.docs[d]) for d in parents):
                self.cards[key]=dict(card,ancestor_docs=sorted(parents))
            else:self.rejected_cards[key]='validation/test or unknown ancestor'
        self.realism = realism if realism is not None else noise.read(noise_path or noise.NOISE / 'measurements.json')
        self.noise_root=(Path(noise_path).parent/'noise-assets' if (Path(noise_path).parent/'noise-assets').exists()
                         else Path(noise_path).parent) if noise_path else noise.NOISE
        self.rejected_stale_measurements=[]
        if realism is None:
            kept=[]
            for m in self.realism['measurements']:
                if (m['class_id'] in self.cards and m.get('tile_signature')==self.cards[m['class_id']].get('signature')
                    and (set(self.cards[m['class_id']]['ancestor_docs'])|{m['source_doc']})<=set(m.get('ancestor_docs',[]))
                    and m.get('binding') and m['binding']==noise.source_binding(m['source_doc'],m['box'],m['class_id'])):
                    for field in ('mask','colours','residual'):noise.asset_path(self.noise_root,m,field)
                    kept.append(m)
                else:
                    self.rejected_stale_measurements.append(m['id'])
            self.realism=dict(self.realism,measurements=kept)
        appearance_docs = {m['source_doc'] for m in self.realism['measurements']}
        appearance_docs.update(self.realism.get('rails',{}))
        appearance_docs.update(self.realism.get('gap_sources',{}))
        appearance_docs.update(b['doc'] for rows in self.realism.get('boundaries',{}).values() for b in rows)
        for did in appearance_docs:
            noise.require_training(self.docs[did])
        for m in self.realism['measurements']:
            for did in m.get('ancestor_docs',[]):
                noise.require_training(self.docs[did])
        if realism is None and not self.frozen_context and self.realism.get('provenance')!=noise.cache_provenance():
            raise ValueError('Stale noise cache provenance; rebuild noise library')
        self.windows = noise.Windows()
        self.models = {}
        self.appearance_checks={}
        self.quarantined=defaultdict(list)
        for r in noise.read(paths.DATA/'genlab/fasttrack/training-v2.json')['records']:
            if r.get('quality_issues') or not r.get('usable_for_generation'):
                self.quarantined[r['document_id']].append(r['box_xyxy'])

    def assert_context_current(self):
        current=noise.read(paths.DATA/'genlab/fasttrack/training-v2.json')
        if current['snapshot']!=self.context['normalized']['snapshot']:
            raise ValueError('Labels changed during generation; rebuild/freeze context')
        by_id={r['sample_id']:r for r in current['records']}
        for row in self.context['normalized']['records']:
            if row.get('usable_for_generation') and row['document_id'] in self.docs and noise.training(self.docs[row['document_id']]):
                if by_id.get(row['sample_id'])!=row:
                    raise ValueError('Archived context source revoked or relabelled: '+row['sample_id'])

    def close(self):
        self.windows.close()

    def family(self, did):
        return self.fmap.get(self.docs[did]['name'], 'One-off layout')

    def changed_fraction(self,did,rgb,origin):
        doc=self.docs[did];x0,_,x1,_=doc['boxes']['interpreted_lithology_column'];changed=0
        for top in range(0,len(rgb),512):
            end=min(len(rgb),top+512)
            original=self.windows.get(doc,[x0,origin+top,x1,origin+end])
            changed+=int(np.any(original!=rgb[top:end],axis=2).sum())
        return changed/(len(rgb)*rgb.shape[1])

    def available(self, did, mode='column'):
        noise.require_training(self.docs[did])
        family = self.family(did)
        choices, skipped = {}, []
        classes = self.stats['classes']
        for cid, c in sorted(classes.items()):
            if c['doc_id'] != did:
                continue
            source = cid if cid in self.cards else None
            # A shared lithology name is not proof of shared printed appearance.
            if source and mode=='page' and (self.cards[source].get('donor') or
                self.cards[source].get('real_doc_id',did)!=did):
                skipped.append(dict(class_id=cid,reason='page-legend-requires-document-appearance'))
                continue
            # Rendering classes with no within-family thickness evidence is impossible.
            has_thickness = any((cid if level.startswith('doc:') else c['norm']) in block.get('thickness', {})
                for level, block in self.stats['levels'].items() if level in ('doc:'+did, 'family:'+family))
            if source is None or not has_thickness:
                reason=getattr(self,'rejected_cards',{}).get(cid)
                skipped.append(dict(class_id=cid, reason=('pattern-ancestry-rejected: '+reason if reason else 'no-same-name-pattern') if source is None else 'no-family-thickness'))
                continue
            card=self.cards[source]
            if not card.get('column_validated') or card.get('reference_kind')!='target column' or card.get('provisional'):
                skipped.append(dict(class_id=cid,reason='native-column-appearance-unvalidated'))
                continue
            try:
                def permitted_crop(box):
                    a,b,c,d=box
                    return not any(min(c,z)>max(a,x) and min(d,w)>max(b,y) for x,y,z,w in
                        getattr(self,'quarantined',{}).get(self.cards[source].get('real_doc_id',did),[]))
                if source and any(not permitted_crop(b) for b in self.cards[source].get('real',[])):
                    skipped.append(dict(class_id=cid,reason='quarantined-pattern-evidence'))
                    continue
                if source not in self.models:
                    card_to_load=copy.deepcopy(self.cards[source])
                    if self.model_root and card_to_load.get('render_model'):
                        asset=card_to_load['render_model']['asset']
                        asset['path']=Path(asset['path']).name
                    self.models[source] = noise.load_model(card_to_load,self.model_root)
                if source not in self.appearance_checks:
                    self.appearance_checks[source]=self.native_appearance(source)
                if not self.appearance_checks[source]['eligible']:
                    reason='native-column-appearance-mismatch' if self.appearance_checks[source]['checks'] else 'unconfirmed-class-reference'
                    skipped.append(dict(class_id=cid,reason=reason,check=self.appearance_checks[source]))
                    continue
                probe=Sampler.__new__(Sampler)
                probe.stats=self.stats;probe.doc_id=did;probe.family=family
                probe.mapping=self.stats['levels'].get('doc:'+did,{}).get('depth_mapping',{})
                probe.levels=[(level,self.stats['levels'][level]) for level in ('doc:'+did,'family:'+family) if level in self.stats['levels']]
                probe.duration_model(cid,'all')
            except ValueError as error:
                skipped.append(dict(class_id=cid,reason=str(error)));continue
            except FileNotFoundError:
                skipped.append(dict(class_id=cid, reason='lab-model-asset-missing'))
                continue
            choices[cid] = dict(pattern_id=source, pattern_source='document' if source == cid else 'family-substitute')
        return choices, skipped

    def native_appearance(self,key):
        """Independent native column scale/ink check; unconfirmed class identity fails closed."""
        card=self.cards[key];did=card.get('real_doc_id',card['doc_id']);data=self.models[key]
        checks=[]
        for box in card.get('real',[])[:8]:
            binding=noise.source_binding(did,box,key,self.context['normalized']['records'])
            if binding['kind']!='normalized-target':continue
            bounded=[box[0],box[1],box[2],min(box[3],box[1]+512)]
            cached=paths.CROPS/did/('%d_%d_%d_%d.png'%tuple(box))
            if cached.exists():
                with Image.open(cached) as im:real=np.asarray(im.convert('RGB').crop((0,0,im.width,min(512,im.height))))
            else:real=self.windows.get(self.docs[did],bounded)
            from genlab.assemble import learn_v2 as learn
            if not training_v2.pixels_allowed(real):raise ValueError('Quarantined native appearance crop')
            h,w=real.shape[:2];digest=hashlib.sha256(f'{w}x{h}:'.encode()+real.tobytes()).hexdigest()
            for record in binding['normalized_sources']:
                if record['box_xyxy']==bounded and record['pixel_sha256']!=digest:
                    raise ValueError('Cached native pixels disagree with normalized source')
            fake=render.layer(data,box[0],box[2],box[1],len(real),np.random.default_rng(41))
            checks.append(native_geometry(real,fake))
        # Different finite crop phases legitimately have different ink counts and
        # stroke quantiles. Compare against the measured same-class envelope,
        # rather than demanding equality to each individual crop phase.
        if checks:
            inks=[c['real_ink_fraction'] for c in checks];strokes=[c['real_stroke_px'] for c in checks]
            ink_bounds=[max(0.,min(inks)-.02),min(1.,max(inks)+.02)]
            stroke_bounds=[max(0.,min(strokes)-1.),max(strokes)+1.]
            eligible=all(ink_bounds[0]<=c['generated_ink_fraction']<=ink_bounds[1] and
                stroke_bounds[0]<=c['generated_stroke_px']<=stroke_bounds[1] and
                c['palette_eligible'] and (c['generated_mismatch_at_real_repeat'] is None or
                 c['generated_mismatch_at_real_repeat']<=c['repeat_mismatch_limit']) for c in checks)
        else:eligible=False;ink_bounds=stroke_bounds=None
        return dict(eligible=bool(eligible),real_ink_bounds=ink_bounds,real_stroke_bounds=stroke_bounds,
                    source='class-confirmed training column; native 1:1; observed variation envelope',checks=checks)

    def snapshot_inputs(self,root):
        root=Path(root);bundle=root/'inputs';bundle.mkdir(parents=True,exist_ok=True)
        # Freeze actual loaded appearance bytes so live lab pruning is irrelevant.
        frozen_cards=copy.deepcopy(self.cards)
        for key,data in self.models.items():
            card=frozen_cards[key]
            from genlab.assemble import learn_v2 as learn
            card['render_model']=learn.export_model(data,bundle,card.get('signature'))
            card['render_model']['asset']['path']=Path(card['render_model']['asset']['path']).name
            if card['render_model'].get('binding_sha256'):
                card['render_model']['binding_sha256']=learn.model_binding(card['render_model'])
        # Freeze also withheld/unloaded cards; replay must never depend on live pruning.
        for key,card in frozen_cards.items():
            if key in self.models or not card.get('render_model'):continue
            from genlab.assemble import learn_v2 as learn
            original=card['render_model'];learn.load_model(original,self.model_root or paths.OUT,card.get('signature'))
            source=(self.model_root or paths.OUT)/original['asset']['path']
            target=bundle/Path(original['asset']['path']).name
            if source.resolve()!=target.resolve():shutil.copyfile(source,target)
            original['asset']['path']=target.name
            if original.get('binding_sha256'):original['binding_sha256']=learn.model_binding(original)
        noise.write(bundle/'state.json',dict(cards=list(frozen_cards.values())))
        assets=bundle/'noise-assets';assets.mkdir(exist_ok=True)
        for m in self.realism['measurements']:
            for field in ('mask','colours','residual'):
                source=noise.asset_path(self.noise_root,m,field)
                target=assets/m[field];target.parent.mkdir(parents=True,exist_ok=True)
                if source.resolve()!=target.resolve():shutil.copyfile(source,target)
        for rail in self.realism.get('rails',{}).values():
            source=noise.asset_path(self.noise_root,rail,'file');target=assets/rail['file']
            if source.resolve()!=target.resolve():shutil.copyfile(source,target)
        noise.write(bundle/'noise.json',self.realism)
        noise.write(bundle/'context.json',self.context)
        dependency_hashes={p.relative_to(bundle).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in bundle.rglob('*') if p.is_file() and p.name!='dependencies.json'}
        noise.write(bundle/'dependencies.json',dependency_hashes)
        return str(bundle/'state.json')

    def coverage(self,did):
        choices,skipped=self.available(did)
        block=self.stats['levels'].get('doc:'+did,{})
        masses={z:sum(p for cid,p in row.items() if cid in choices) for z,row in block.get('composition',{}).items()
                if sum(row.values())>0 and (z=='all' or z.startswith(block.get('zone_scheme','depth')+'-'))}
        mass=min(masses.values()) if masses else 0.
        measured=block.get('complete_human_rows',0)>=3000
        required=sorted(cid for cid,p in block.get('composition',{}).get('all',{}).items() if p>0)
        return dict(supported_composition_mass=mass,by_zone=masses,threshold=.95,
                    status='representative' if mass>=.95 and measured else 'partial',
                    composition_evidence='human-bed' if measured else 'machine-estimated-or-unidentified',
                    complete_human_rows=block.get('complete_human_rows',0),
                    missing_complete_human_rows=max(0,3000-block.get('complete_human_rows',0)),
                    required_class_ids=required,missing_class_ids=[cid for cid in required if cid not in choices],
                    compatible_duration_available={cid:self.stats.get('backoff',{}).get(cid,{}).get('compatible_duration_available',False) for cid in required},
                    skipped_classes=skipped)

    def appearance(self, did, rgb, layers, rng, iteration=2, return_visibility=False):
        visible=np.ones(rgb.shape[:2],bool)
        records=noise.read(paths.DATA/'genlab/fasttrack/training-v2.json')['records']
        docs=getattr(self,'docs',None)
        measurements = self.realism['measurements']
        pool = [m for m in measurements if m['doc_id'] == did]
        basis = 'doc'
        if not pool:
            pool = [m for m in measurements if m['family'] == self.family(did)]
            basis = 'family'
        events = []
        m = pool[int(rng.integers(len(pool)))] if pool else None
        basis=basis if pool else 'unmeasured-partial'
        params=dict(basis=basis,status='measured' if pool else 'partial-no-compatible-residuals',
                    iteration=iteration,unrestricted_overlays=False)
        # Native tiles already contain scan blur: only measured incremental blur.
        if m and iteration >= 1:
            rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(m['blur']))).copy()
        drift = np.asarray(m['drift'] if m else [0,0,0], np.float32)
        if m and iteration >= 2:
            # Resample the measured low-amplitude residual field in nonrepeating blocks.
            with np.load(noise.asset_path(self.noise_root,m,'residual')) as cache:
                field = cache['residual'].astype(np.float32)
                valid = cache['valid']
            field[~valid] = 0
            field -= np.median(field[valid], axis=0) if valid.any() else 0
        for top in range(0, len(rgb), 512):
            bottom = min(len(rgb), top+512)
            a = rgb[top:bottom].astype(np.float32)+drift
            if m and iteration >= 2:
                # Draw contiguous residual blocks once, without periodic wrapping.
                # A new source phase per block prevents an exact source-height comb.
                field_out=np.zeros_like(a)
                y=0
                while y<len(a):
                    # Vary block length as well as phase: full-height blocks would
                    # retain the old residual-height periodicity after quantization.
                    h=min(int(rng.integers(max(1,len(field)//4),len(field)+1)),len(a)-y)
                    sy=int(rng.integers(max(1,len(field)-h+1)))
                    w=min(field.shape[1],a.shape[1])
                    sx=int(rng.integers(max(1,field.shape[1]-w+1)))
                    piece=field[sy:sy+h,sx:sx+w].copy()
                    if rng.random()<.5:
                        piece=piece[::-1]
                    # Smooth random amplitude measured in the source's own range.
                    piece*=rng.uniform(.8,1.2,size=(h,1,1))
                    field_out[y:y+h,:w]=piece
                    y+=h
                a += field_out
            elif m and iteration >= 1 and m['noise_sigma']:
                a += rng.normal(0, m['noise_sigma'], a.shape)
            rgb[top:bottom] = np.clip(a, 0, 255).astype(np.uint8)
        # Residual ink contains lithology strokes. It is never a supervised nuisance
        # overlay until independently validated component identities are available.
        # A shared family name is insufficient evidence for borrowing drawn contacts.
        bb = [b for b in self.realism.get('boundaries', {}).get(self.family(did), [])
              if b.get('doc')==did and noise.nuisance_allowed(b,docs,records)]
        if iteration >= 2 and bb:
            for layer in layers[1:]:
                b = bb[int(rng.integers(len(bb)))]
                y = layer['top']
                if b['line']:
                    width = max(1, b['line_width'])
                    rgb[y:min(len(rgb), y+width)] = b['ink']
                    visible[y:min(len(rgb),y+width)]=False
                    events.append(dict(kind='boundary-line', top=y, width=width, source_doc=b['doc']))
                elif y > 0 and b.get('antialias_alpha',0) and rng.random() < b['antialias']:
                    alpha = b['antialias_alpha']
                    rgb[y] = np.rint((1-alpha)*rgb[y-1].astype(float)+alpha*rgb[y]).astype(np.uint8)
                    visible[y]=False
                    events.append(dict(kind='boundary-antialias',top=y,alpha=alpha,source_doc=b['doc']))
        rail = self.realism.get('rails', {}).get(did)
        if rail and not noise.nuisance_allowed(rail,docs,records):rail=None
        if rail and rail['columns']:
            with Image.open(noise.asset_path(self.noise_root,rail,'file')) as im:
                pixels = np.asarray(im.convert('RGB'))
            yy = np.arange(len(rgb)) % len(pixels)
            for i, x in enumerate(rail['columns']):
                rgb[:, x] = pixels[yy, i]
                visible[:,x]=False
        if m:params.update({k:m[k] for k in ('blur','drift','noise_sigma','jpeg_block_phase_energy','source_doc')})
        params.update(measurement_id=m['id'] if m else None,residual_scan_field=bool(m and iteration>=2),
                      rails=rail,overlay_fields=len(events))
        return (rgb,params,events,visible) if return_visibility else (rgb, params, events)

    def column(self, did, seed, height, folder, mode='column', iteration=2, page_top=None,
               context_consistent=False,appearance_transfer=None,window_index=None,depth_stratum=None,
               regions=None):
        started = time.perf_counter()
        self.assert_context_current()
        doc = self.docs[did]
        noise.require_training(doc)
        choices, skipped = self.available(did,mode)
        sampler = Sampler(self.stats, did, self.family(did), choices, seed)
        duration_evidence={}
        for cid in choices:
            level,field,d=sampler.duration_model(cid,'all')
            values,weights=sampler.duration_distribution(level,d)
            duration_evidence[cid]=dict(level=level,field=field,source_level=d.get('fallback_level',level),
                observations=len(values),unique_durations=len(np.unique(values)),
                expected_rows=float(np.average(values,weights=weights)),
                shrinkage_local_weight=d.get('shrinkage_local_weight'))
        x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
        origin = y0 if page_top is None else page_top
        if mode == 'column' and page_top is None:
            # Seeded strata cover the entire supported source span. The index is
            # explicit so a build visits each stratum before repeating one.
            span=y1-y0;effective=min(height,span);latest=max(y0,y1-effective)
            if depth_stratum not in (None,'shallow','middle','deep'):
                raise ValueError('depth_stratum must be shallow, middle or deep')
            bins=max(1,int(np.ceil(span/max(1,effective))))
            index=(window_index if window_index is not None else seed)%bins
            lo=y0+(latest-y0)*index//bins;hi=y0+(latest-y0)*(index+1)//bins
            if depth_stratum:
                index=('shallow','middle','deep').index(depth_stratum);bins=3
                lo=y0+(latest-y0)*index//3;hi=y0+(latest-y0)*(index+1)//3
            origin=int(np.random.default_rng(seed).integers(lo,hi+1))
        if not y0<=origin<y1:
            raise ValueError('Source context origin outside column')
        requested_height=height
        height=min(height,y1-origin)
        # Partial columns keep the original scan's depth position and pixel units.
        start_frac = (origin-y0)/max(1,y1-y0)
        span_frac = min(1.-start_frac,height/max(1,y1-y0))
        layers = sampler.sequence(height,start_frac=start_frac,span_frac=span_frac,
                                  clip_to_zone=iteration>=2,start_at_boundary=False)
        if context_consistent:
            # Existing interpreted lithology is the strongest available same-page
            # proxy for cuttings. Preserve unlabelled pixels and ignore their loss.
            from . import stats as statistics
            rows=self.context['normalized']['records']
            peers=[r for r in rows if r['document_id']==did and r.get('usable_for_generation') and not r.get('quality_issues')]
            intervals=[]
            for r in peers:
                if statistics.infer_box_kind(r,peers)!='bed' or len(r.get('targets',[]))!=1:
                    continue
                cid=f"{did}:{r['targets'][0]['legend_id']}"
                if cid in choices:
                    intervals.append(dict(top=r['box_xyxy'][1]-origin,bottom=r['box_xyxy'][3]-origin,code=sorted(choices).index(cid)))
            labels=statistics.human_raster(intervals,height)
            layers=[dict(top=a,bottom=b,class_id=sorted(choices)[k] if k>=0 else GAP,
                         norm=self.stats['classes'][sorted(choices)[k]]['norm'] if k>=0 else GAP,
                         pattern_source='document' if k>=0 else 'original-unlabelled-ignore',zone=sampler.zone(start_frac+span_frac*a/height))
                    for a,b,k in statistics.runs(labels)]
            for l in layers:
                matching=[r for r in peers if len(r.get('targets',[]))==1 and
                    f"{did}:{r['targets'][0]['legend_id']}"==l['class_id'] and
                    statistics.infer_box_kind(r,peers)=='bed']
                top=any(r['box_xyxy'][1]==origin+l['top'] and
                    r.get('boundary_top_confirmed',True) for r in matching)
                bottom=any(r['box_xyxy'][3]==origin+l['bottom'] and
                    r.get('boundary_bottom_confirmed',True) for r in matching)
                l.update(boundary_top_confirmed=top,boundary_bottom_confirmed=bottom,
                    top_censored=not top,bottom_censored=not bottom,
                    boundary_basis='source annotation edges retained only when visible')
        width = x1-x0
        rgb = np.full((height, width, 3), 255, np.uint8)
        mask = np.full((height, width),255,np.uint8)
        indices = {cid: i+1 for i, cid in enumerate(sorted(choices))}
        for layer in layers:
            cid, a, b = layer['class_id'], layer['top'], layer['bottom']
            if cid == GAP:
                if context_consistent:
                    for top in range(a,b,512):
                        rgb[top:min(b,top+512)]=self.windows.get(doc,[x0,origin+top,x1,origin+min(b,top+512)])
                layer.update(ignore=True,source_doc=did)
                continue
            source = choices[cid]['pattern_id']
            layer['pattern_id'] = source
            rgb[a:b] = render.layer(self.models[source], x0, x1, origin+a, b-a, sampler.rng)
            mask[a:b] = indices[cid]
            cls=self.stats['classes'][cid]
            entry=next((e for e in doc['legend'] if e['id']==cls.get('legend_id')),None)
            layer.update(legend_id=cls.get('legend_id'),legend_label_printed=cls['name'],
                         legend_swatch_box=entry['box'] if entry else None)
        mapping=sampler.mapping
        for layer in layers:
            from .stats import depth_at_page_row
            layer['depth_top_ft']=depth_at_page_row(mapping,origin+layer['top'],y0)
            layer['depth_bottom_ft']=depth_at_page_row(mapping,origin+layer['bottom'],y0)
        # Optional, explicitly specified 2D regions retain host beds underneath.
        # No rates, positions or identities are inferred from unlabelled pixels.
        region_truth=[]
        for r in sorted(regions or [],key=lambda r:r.get('paint_order',0)):
            cid=r['class_id'];a,b,c,d=map(int,r['box_xyxy'])
            if not (0<=a<c<=width and 0<=b<d<=height):raise ValueError('Region outside output')
            if cid not in choices:raise ValueError('Unconfirmed or unavailable region class: '+cid)
            source=choices[cid]['pattern_id']
            rgb[b:d,a:c]=render.layer(self.models[source],x0+a,x0+c,origin+b,d-b,sampler.rng)
            mask[b:d,a:c]=indices[cid] if r.get('confirmed',False) else 255
            region_truth.append(dict(r,box_xyxy=[a,b,c,d],source_doc=did))
        latent=mask.copy()
        base_rgb=rgb.copy()
        rgb, params, overlays, visible = self.appearance(did, rgb, layers, sampler.rng, iteration,return_visibility=True)
        # Guard the final appearance against the already confirmed native base.
        for layer in layers:
            a,b=layer['top'],min(layer['bottom'],layer['top']+512)
            if layer['class_id']==GAP or b<=a:continue
            take=visible[a:b]
            if not take.any():continue
            final=rgb[a:b].copy();final[~take]=base_rgb[a:b][~take]
            if not native_geometry(base_rgb[a:b],final)['palette_eligible']:
                raise ValueError('Final rendered palette outside confirmed native envelope: '+layer['class_id'])
        mask[~visible]=255
        if context_consistent:
            for l in layers:
                if l['class_id']==GAP:
                    for top in range(l['top'],l['bottom'],512):
                        end=min(l['bottom'],top+512)
                        rgb[top:end]=self.windows.get(doc,[x0,origin+top,x1,origin+end])
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(folder / 'column.png')
        Image.fromarray(mask).save(folder / 'mask.png')
        Image.fromarray(latent).save(folder/'latent-mask.png')
        legend=Image.new('RGB',(360,max(1,len(choices))*48),'white')
        ld=ImageDraw.Draw(legend)
        legend_entries=[]
        for i,cid in enumerate(sorted(choices)):
            source=choices[cid]['pattern_id']
            patch=render.layer(self.models[source],x0,x0+72,origin,36,np.random.default_rng(seed+i))
            legend.paste(Image.fromarray(patch),(4,i*48+4))
            ld.text((84,i*48+12),self.stats['classes'][cid]['name'],fill='black')
            legend_entries.append(dict(class_id=cid,legend_id=self.stats['classes'][cid].get('legend_id'),
                label_printed=self.stats['classes'][cid]['name'],swatch_box=[4,i*48+4,76,i*48+40]))
        output_boxes={e['class_id']:e['swatch_box'] for e in legend_entries}
        for layer in layers:
            layer['legend_output_swatch_box']=output_boxes.get(layer['class_id'])
        legend.save(folder/'legend.png')
        truth = dict(synth_id=folder.name, seed=seed, mode=mode, base_doc=did, family=self.family(did),
            stats_level_used=dict(sampler.used), layers=layers, noise=params, overlays=overlays,
            duration_evidence=duration_evidence,
            regions=region_truth,latent_mask='latent-mask.png',visible_mask='mask.png',
            skipped_transition_states=sorted(sampler.projected),
            schema='asm-2',class_index={str(i): cid for cid, i in indices.items()} | {'255': GAP},ignore_index=255,
            legend='legend.png',legend_entries=legend_entries,depth_mapping=mapping,coverage=self.coverage(did),
            requested_height=requested_height,source_span_capped=height!=requested_height,
            covered_zones=[z['name'] for z in self.stats['levels'].get('doc:'+did,{}).get('zones',[])
                if z['top_frac']<start_frac+span_frac and z['bottom_frac']>start_frac],
            window=dict(index=window_index,stratum=depth_stratum,origin=origin,
                        selection='explicit page_top' if page_top is not None else 'seeded source strata'),
            context_consistent=context_consistent,
            context_policy='Observed same-page interpreted classes; unknown source rows retained and ignored' if context_consistent else 'Independent lithology; cuttings and descriptions inconsistent; context models must exclude',
            normalized_snapshot=self.stats['inputs']['normalized_snapshot'],
            width=width, height=height, zone_start_fraction=start_frac, zone_span_fraction=span_frac,
            skipped_classes=skipped, stats_path=str(self.stats_path),
            stats_sha256=self.stats_hash,
            real_depth_span=[origin,origin+height],
            split='train', source_docs=sorted({did} | {self.cards[v['pattern_id']]['doc_id'] for v in choices.values()}
                 | {e['source_doc'] for e in overlays if e.get('source_doc')}
                 | ({params['source_doc']} if params.get('source_doc') else set())),
            patterns={cid:dict(v,signature=self.cards[v['pattern_id']].get('signature')) for cid,v in choices.items()})
        truth['source_docs']=sorted(set(truth['source_docs'])|{d for v in choices.values() for d in self.cards[v['pattern_id']]['ancestor_docs']})
        policy=noise.read(paths.DATA/'normalized/split-policy.json')
        truth['ancestor_wells']=sorted({policy['document_groups'][d] for d in truth['source_docs']}|
            set(self.stats['levels']['global']['source_wells']))
        truth['label_snapshot']=truth['normalized_snapshot']
        truth['context_sha256']=self.context_hash
        self.assert_context_current()
        supervised=int(np.any(mask!=255,axis=1).sum())
        fraction=self.changed_fraction(did,rgb,origin)
        truth.update(supervised_rows=supervised,supervised_pixels=int((mask!=255).sum()),changed_pixel_fraction=fraction,
                     lithology_augmentation=supervised>0 and fraction>0,
                     localization_only=supervised==0,task='lithology' if supervised else 'localization-only')
        truth['rejected_stale_noise_measurements']=self.rejected_stale_measurements
        noise.write(folder / 'truth.json', truth)
        if appearance_transfer:
            self.transfer(folder,truth,appearance_transfer)
        print('PARTIAL column' if params['status']!='measured' else 'PASS column', folder, 'rows', height, 'layers', len(layers),
              'seconds', round(time.perf_counter()-started, 3), flush=True)
        return rgb, truth

    def transfer(self,folder,truth,target='D-582 1-bit grid'):
        """An explicit experimental style preset; no target/test pixels measured."""
        dest=Path(folder)/'appearance-transfer'
        dest.mkdir(exist_ok=True)
        with Image.open(Path(folder)/'column.png') as im:
            src=np.asarray(im.convert('RGB'))
        # Threshold measured solely from this generated training-side column.
        gray=np.asarray(Image.fromarray(src).convert('L'))
        threshold=float(np.quantile(gray,.4))
        bw=np.where(gray<=min(220,threshold),0,255).astype(np.uint8)
        width=250
        Image.fromarray(bw).resize((width,len(src)),Image.Resampling.NEAREST).save(dest/'column.png')
        with Image.open(Path(folder)/'mask.png') as im:
            im.resize((width,len(src)),Image.Resampling.NEAREST).save(dest/'mask.png')
        with Image.open(Path(folder)/'legend.png') as im:
            lg=np.asarray(im.convert('L'))
            Image.fromarray(np.where(lg<=min(220,threshold),0,255).astype(np.uint8)).save(dest/'legend.png')
        t=dict(truth,synth_id=Path(folder).name+'-appearance-transfer',width=width,
            appearance_transfer=dict(target_family=target,status='experimental-unvalidated',
                direct_training_documents=0,width_preset=width,threshold=threshold,
                source='training-derived column/legend; user-requested generic width preset; zero target/test measurements'),
            coverage=dict(status='partial',direct_target_family_coverage=0))
        noise.write(dest/'truth.json',t)
        return dest

    @staticmethod
    def scale_variants(folder,truth):
        """Withhold nominal scale changes without independently validated textures."""
        noise.write(Path(folder)/'scale-availability.json',dict(status='unavailable',outputs=[],
            reason='No matching-scale validated native texture; resizing symbols changes class appearance'))
        return []

    @staticmethod
    def crossing_mask(original):
        dark=original.max(2)<100
        rows=(dark[:,9:-9].mean(1)>.85)&((dark[:,:6].mean(1)>.8)|(dark[:,-6:].mean(1)>.8))
        return dark[:,6:-6]&rows[:,None]

    def page(self, did, seed, folder, band_rows=512,context_consistent=False,regions=None,
             native_no_data_spans=None, reserved_spans=None):
        from .safety_v2 import spans, row_selection, refresh_masks
        if not isinstance(band_rows,int) or band_rows<=0 or band_rows%256:
            raise ValueError('band_rows must be a positive multiple of 256 for tiled TIFF export')
        doc = self.docs[did]
        noise.require_training(doc)
        x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
        rgb, truth = self.column(did, seed, y1-y0, folder, mode='page',context_consistent=context_consistent,regions=regions)
        no_data = spans(list(native_no_data_spans or []) + doc.get('native_no_data_spans', []) +
                        truth.get('native_no_data_spans', []))
        reserved = spans(list(reserved_spans or []) + getattr(self, 'reserved_spans', []) +
                         doc.get('reserved_spans', []) + truth.get('reserved_spans', []))
        if context_consistent:
            raise ValueError('Independent column quilting does not support coherent cross-track context')
        folder = Path(folder)
        with Image.open(folder/'mask.png') as image:visible_mask=np.asarray(image).copy()
        bands, crossing = [], 0
        crossing_frame_rows=set();crossing_interior_rows=set()
        mm = []  # unvalidated residual components must not restore motif fragments
        restored = np.zeros(visible_mask.shape, bool)
        for top in range(0, doc['height'], band_rows):
            bottom = min(doc['height'], top+band_rows)
            band = self.windows.get(doc, [0, top, doc['width'], bottom])
            a, b = max(top, y0), min(bottom, y1)
            if a < b:
                own = band[a-top:b-top, x0:x1].copy()
                extended=band[a-top:b-top,x0-6:x1+6].copy()
                band[a-top:b-top, x0:x1] = rgb[a-y0:b-y0]
                take=self.crossing_mask(extended)
                for row in np.flatnonzero(take.any(1)):
                    absolute=a+int(row)
                    (crossing_frame_rows if min(absolute-y0,y1-1-absolute)<6 else crossing_interior_rows).add(absolute)
                band[a-top:b-top,x0:x1][take]=own[take]
                visible_mask[a-y0:b-y0][take]=255
                restored[a-y0:b-y0][take]=True
                crossing+=int(take.sum())
                # Preserve this page's actual rails at their original depth.
                rail = self.realism.get('rails', {}).get(did, {})
                for x in rail.get('columns', []):
                    band[a-top:b-top, x0+x] = own[:, x]
                    visible_mask[a-y0:b-y0,x]=255
                    restored[a-y0:b-y0,x]=True
                # Only source residual components connected to a column boundary
                # can plausibly continue a real mark from a neighbouring track.
                for m in mm:
                    ma, mb, mc, md = m['box']
                    p, q = max(a, mb), min(b, md)
                    if p >= q:
                        continue
                    with Image.open(noise.NOISE / m['mask']) as im:
                        mask = np.asarray(im) > 0
                    labels, _ = noise.ndimage.label(mask)
                    touching = set()
                    if ma <= x0+3:
                        touching.update(np.unique(labels[:, :3]))
                    if mc >= x1-3:
                        touching.update(np.unique(labels[:, -3:]))
                    touching.discard(0)
                    take = np.isin(labels[p-mb:q-mb], list(touching))
                    left, right = max(ma, x0), min(mc, x1)
                    take = take[:, left-ma:right-ma]
                    if take.any():
                        band[p-top:q-top, left:right][take] = own[p-a:q-a, left-x0:right-x0][take]
                        crossing += int(take.sum())
                keep = row_selection(a, b, no_data)
                band[a-top:b-top,x0:x1][keep] = own[keep]
                visible_mask[a-y0:b-y0][keep] = 255
                restored[a-y0:b-y0][keep] = True
            # Ignore labels alone cannot hide imagery from a training model.
            withheld = row_selection(top, bottom, reserved)
            band[withheld] = 255
            if a < b:
                hidden = row_selection(a, b, reserved)
                visible_mask[a-y0:b-y0][hidden] = 255
                restored[a-y0:b-y0][hidden] = False
                rgb[a-y0:b-y0]=band[a-top:b-top,x0:x1]
            filename = f'page-bands/band-{top:07d}.png'
            (folder / 'page-bands').mkdir(exist_ok=True)
            Image.fromarray(band).save(folder / filename)
            bands.append(dict(file=filename, top=top, bottom=bottom, width=doc['width']))
        Image.fromarray(rgb).save(folder/'column.png')
        Image.fromarray(visible_mask).save(folder/'mask.png')
        refresh_masks(folder, visible_mask)
        np.save(folder/'restored-native-mask.npy', restored, allow_pickle=False)
        provenance = folder/'pixel-sources.npz'
        if provenance.exists():
            with np.load(provenance) as source:
                maps = {k:source[k].copy() for k in source.files}
            if all(k in maps for k in ('source_index', 'page_y', 'page_x')):
                yy, xx = np.indices(visible_mask.shape)
                maps['source_index'][restored] = -1
                maps['page_y'][restored] = (yy+y0)[restored]
                maps['page_x'][restored] = (xx+x0)[restored]
                hidden = row_selection(y0, y1, reserved)
                # -2 is generated neutral redaction, not native page ancestry.
                for value in maps.values():
                    if value.shape == visible_mask.shape:
                        value[hidden] = -2
                np.savez_compressed(provenance, **maps)
        # Stream a dataset-compatible tiled TIFF from the saved bounded bands.
        import tifffile
        def tiles():
            cached, band = -1, None
            for ty in range(0, doc['height'], 256):
                index = ty // band_rows
                if index != cached:
                    with Image.open(folder / bands[index]['file']) as image:
                        band = np.asarray(image.convert('RGB'))
                    cached = index
                dy = ty-bands[index]['top']
                for tx in range(0, doc['width'], 256):
                    tile = np.full((256,256,3), 255, np.uint8)
                    h, w = min(256, len(band)-dy), min(256, doc['width']-tx)
                    tile[:h,:w] = band[dy:dy+h,tx:tx+w]
                    yield tile
        with tifffile.TiffWriter(folder / 'page.tif', bigtiff=True) as writer:
            writer.write(tiles(), shape=(doc['height'],doc['width'],3), dtype=np.uint8,
                         tile=(256,256), photometric='rgb', compression='deflate', maxworkers=1)
        original = noise.read(model.DATASET / 'bbox_annotation' / (doc['name']+'.json'))
        annotation = dict(original, image_name='page.tif', width=doc['width'], height=doc['height'],
                          bounding_boxes=doc['boxes'], image_bands=bands, synthetic_lithology_truth='truth.json')
        noise.write(folder / 'page-bands.json', dict(format='png-horizontal-bands', width=doc['width'],
                    height=doc['height'], bands=bands))
        noise.write(folder / 'bbox_annotation.json', annotation)
        truth['page'] = dict(bands=len(bands), band_rows=band_rows, crossing_overlay_pixels=crossing,
                             crossing_frame_rows=sorted(crossing_frame_rows),crossing_interior_rows=sorted(crossing_interior_rows),
                             crossing_method='direct native-band horizontal continuation; frame/interior reported separately',
                             lithology_bbox=[x0,y0,x1,y1], annotation='bbox_annotation.json')
        truth.update(native_no_data_spans=no_data, reserved_spans=reserved,
                     context_consistent=False, allowed_tasks=['column-lithology', 'localization'],
                     exposure=dict(reserved_spans=reserved, retained_reserved_spans=[],
                                   policy='full-width neutral redaction; mask=255; source_index=-2'),
                     restoration_mask='restored-native-mask.npy')
        lineage = truth.get('lineage') or {'base':[did]}
        if lineage.get('base') != [did]:
            raise ValueError('Construction base lineage disagreement')
        truth.update(training_v2.lineage_eligibility(lineage))
        truth['changed_pixel_fraction']=self.changed_fraction(did,rgb,y0)
        truth['supervised_rows']=int(np.any(visible_mask!=255,axis=1).sum())
        truth['supervised_pixels']=int((visible_mask!=255).sum())
        truth['localization_only']=truth['supervised_rows']==0
        truth['task']='lithology' if truth['supervised_rows'] else 'localization-only'
        truth['lithology_augmentation']=truth.get('supervised_rows',0)>0 and truth['changed_pixel_fraction']>0
        noise.write(folder / 'truth.json', truth)
        print('PASS page', folder, 'bands', len(bands), 'crossing_pixels', crossing, flush=True)
        return rgb, truth

    def sheet(self, did, folders, output, rows=1200, depth_top=None):
        doc = self.docs[did]
        noise.require_training(doc)
        x0, y0, x1, y1 = doc['boxes']['interpreted_lithology_column']
        first_truth = noise.read(Path(folders[0])/'truth.json')
        origin = first_truth['real_depth_span'][0]
        if depth_top is None:
            with Image.open(Path(folders[0])/'column.png') as image:
                limit = min(y1, origin+image.height)
            records = [r for r in noise.read(paths.DATA/'genlab/fasttrack/training-v2.json')['records']
                       if r['document_id']==did and r.get('usable_for_generation')
                       and origin <= r['box_xyxy'][1] <= limit-rows]
            def score(top):
                rr = [r for r in records if r['box_xyxy'][1]<top+rows and r['box_xyxy'][3]>top]
                return len({str(r['targets']) for r in rr}),len(rr)
            references = [m['box'][1] for m in self.realism['measurements']
                          if m['source_doc']==did and origin<=m['box'][1]<=limit-rows]
            top = max([r['box_xyxy'][1] for r in records] or references or [origin], key=score)
        else:
            top = max(y0, min(depth_top, y1-rows))
        rows = min(rows, y1-top)
        real = self.windows.get(doc, [x0, top, x1, top+rows])
        arrays = [real]
        for folder in folders[:2]:
            with Image.open(Path(folder) / 'column.png') as im:
                a = np.asarray(im.convert('RGB'))
            own_origin = noise.read(Path(folder)/'truth.json')['real_depth_span'][0]
            offset = max(0, min(top-own_origin, len(a)-rows))
            arrays.append(a[offset:offset+rows])
        width = x1-x0
        canvas = Image.new('RGB', (3*max(width, 110)+40, rows+64), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, a in enumerate(arrays):
            x = 10+i*(max(width,110)+10)
            canvas.paste(Image.fromarray(a), (x, 52))
            draw.text((x, 8), ['Real', 'Synthetic A', 'Synthetic B'][i], fill='black')
            draw.text((x, 25), f'{top}:{top+rows}px', fill='black')
        canvas.save(output)
        noise.write(Path(output).with_suffix('.json'), dict(doc=did, real_box=[x0,top,x1,top+rows],
                    synthetic=[str(x) for x in folders[:2]], scale='native 1:1', depth_span_rows=rows))
        return str(output)

    def diagnostic_sheets(self,root):
        """Native review panels remain separate from admitted training outputs."""
        outputs=[]
        for key in ('5709fe9e86ce942b:1','28112eef4c3c645a:13','28112eef4c3c645a:16'):
            card=self.cards.get(key)
            if not card or not card.get('real'):continue
            did=card.get('real_doc_id',card['doc_id']);box=card['real'][0]
            bounded=[box[0],box[1],box[2],min(box[3],box[1]+384)]
            if not noise.crop_allowed(did,bounded):continue
            actual=self.windows.get(self.docs[did],bounded)
            data=self.models.get(key) or noise.load_model(card);self.models[key]=data
            height=max(240,len(actual));width=actual.shape[1]
            panels=[actual]+[render.layer(data,box[0],box[2],box[1],height,np.random.default_rng(seed)) for seed in (7,11)]
            canvas=Image.new('RGB',(3*max(width+12,130),height+75),'white');draw=ImageDraw.Draw(canvas)
            for i,arr in enumerate(panels):
                x=i*max(width+12,130)+4;canvas.paste(Image.fromarray(arr),(x,70))
                draw.text((x,5),('Real training crop','Export seed 7','Export seed 11')[i],fill='black')
                draw.text((x,24),'DIAGNOSTIC ONLY',fill='red')
                draw.text((x,42),key,fill='black')
            dest=Path(root)/('diagnostic-'+key.replace(':','-')+'.png');canvas.save(dest)
            outputs.append(str(dest))
        return outputs

    def family_sheets(self,root):
        """A native training comparison or explicit refusal for every family."""
        outputs=[]
        for level,block in self.stats['levels'].items():
            if not level.startswith('family:'):continue
            family=level[7:];canvas=Image.new('RGB',(920,660),'white');draw=ImageDraw.Draw(canvas)
            draw.text((12,10),family+' | TRAINING ONLY | native 1:1',fill='black')
            draw.text((12,30),'Diagnostic texture comparison; zero admitted logs unless manifest outputs say otherwise',fill='red')
            ids=block['document_ids'];did=ids[0] if ids else None
            if did:
                x0,y0,x1,y1=self.docs[did]['boxes']['interpreted_lithology_column']
                candidates=[c for c in self.cards.values() if c['doc_id']==did and c.get('real')]
                sample=None;fake=None;reason='No admissible texture for this family'
                for card in candidates:
                    box=card['real'][0]
                    box=[max(x0,box[0]),max(y0,box[1]),min(x1,box[2]),min(y1,box[3],box[1]+512)]
                    if box[2]<=box[0] or box[3]<=box[1] or not noise.crop_allowed(did,box):continue
                    try:
                        sample=self.windows.get(self.docs[did],box)
                        data=self.models.get(card['key'])
                        if data is None:
                            payload=copy.deepcopy(card)
                            if self.model_root:payload['render_model']['asset']['path']=Path(payload['render_model']['asset']['path']).name
                            data=noise.load_model(payload,self.model_root)
                        fake=render.layer(data,box[0],box[2],box[1],len(sample),np.random.default_rng(3003))
                        reason=card['key']+'; direct texture diagnostic; production gate unchanged'
                        break
                    except (ValueError,FileNotFoundError):continue
                if sample is None:
                    for top in np.linspace(y0,max(y0,y1-512),12).astype(int):
                        box=[x0,int(top),x1,min(y1,int(top)+512)]
                        if noise.crop_allowed(did,box):
                            sample=self.windows.get(self.docs[did],box);break
                draw.text((12,55),did+' | '+reason,fill='black')
                draw.text((12,80),'Real training crop',fill='black')
                draw.text((310,80),'Synthetic texture (diagnostic) / unavailable',fill='black')
                if sample is not None:canvas.paste(Image.fromarray(sample),(12,105))
                if fake is not None:canvas.paste(Image.fromarray(fake),(310,105))
                else:draw.text((310,105),'UNAVAILABLE: no permitted native model',fill='red')
                coverage=self.coverage(did)
                draw.text((12,630),f"Min supported mass {coverage['supported_composition_mass']:.4f}; required 0.95; "+coverage['status'],fill='red')
            else:draw.text((12,100),'REFUSED: zero training documents; no source pixels read',fill='red')
            path=Path(root)/('diagnostic-family-'+slug(family)+'.png');canvas.save(path);outputs.append(str(path))
        return outputs


def exposure_ledger(stats,results,family_coverage):
    """Realized exposure, with missing slots explicit rather than synthetic counts."""
    ledger=dict(streams={'natural-mixture':dict(samples=len(results)),
                         'coverage':dict(samples=0,status='unavailable until eligible rare-class windows exist')},
                families={},classes={},severity_bins={},scale_derivatives_count_as_new_sequences=False)
    for level,b in stats['levels'].items():
        if not level.startswith('family:'):continue
        f=level[7:];ledger['families'][f]=dict(samples=0,supervised_pixels=0,supervised_rows=0,wells=[],beds=0,
            status=family_coverage.get(f,{}).get('status','unavailable'),covered_zones=[],missing_zones=[])
    for cid in stats['classes']:
        ledger['classes'][cid]=dict(supervised_pixels=0,beds=0,minimum_pixels=1000,minimum_beds=1,missing=True)
    for row in results:
        folder=Path(row['path']);t=noise.read(folder/'truth.json')
        with Image.open(folder/'mask.png') as im:mask=np.asarray(im)
        f=ledger['families'][row['family']];f['samples']+=1
        f['supervised_pixels']+=int((mask!=255).sum());f['supervised_rows']+=int(np.any(mask!=255,1).sum())
        f['wells']=sorted(set(f['wells'])|set(t['ancestor_wells']))
        f['covered_zones']=sorted(set(f['covered_zones'])|{t['base_doc']+'/'+z for z in t.get('covered_zones',[])})
        for index,cid in t['class_index'].items():
            if cid not in ledger['classes']:continue
            c=ledger['classes'][cid];c['supervised_pixels']+=int((mask==int(index)).sum())
            beds=sum(l['class_id']==cid and not l.get('top_censored') and not l.get('bottom_censored') for l in t['layers'])
            c['beds']+=beds;f['beds']+=beds
        for parameter in ('blur','noise_sigma'):
            value=t['noise'].get(parameter)
            label='unmeasured' if value is None else ('native-only' if value==0 else 'measured-positive')
            key=parameter+'/'+label;ledger['severity_bins'][key]=ledger['severity_bins'].get(key,0)+1
    total=sum(f['supervised_pixels'] for f in ledger['families'].values())
    for f in ledger['families'].values():
        f['row_exposure_fraction']=f['supervised_pixels']/total if total else None
        f['balanced_batch_weight']=total/f['supervised_pixels'] if f['supervised_pixels'] else None
        f['missing']=f['samples']==0
    for name,f in ledger['families'].items():
        docs=stats['levels']['family:'+name]['document_ids']
        required={did+'/'+z['name'] for did in docs for z in stats['levels']['doc:'+did].get('zones',[])
                  if z['name'].startswith('depth-') or z['name'].startswith('fifth-')}
        f['missing_zones']=sorted(required-set(f['covered_zones']))
    for c in ledger['classes'].values():c['missing']=c['supervised_pixels']<c['minimum_pixels'] or c['beds']<c['minimum_beds']
    ledger['severity_coverage_status']='unavailable' if not results else 'partial'
    ledger['missing_severity_evidence']=['No additional family-calibrated blur/noise/contrast/compression severity distributions']
    return ledger


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.casefold()).strip('-')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=['column','page','starter'])
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--doc')
    target.add_argument('--family')
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--height', type=int, default=20000)
    parser.add_argument('--run', default='starter')
    parser.add_argument('--stats', type=Path)
    parser.add_argument('--state',type=Path,help='Frozen cards/model paths for reproducible generation')
    parser.add_argument('--noise',type=Path,help='Frozen appearance measurement metadata')
    parser.add_argument('--iteration', type=int, choices=[0,1,2], default=2)
    parser.add_argument('--context-consistent',action='store_true')
    parser.add_argument('--appearance-transfer',action='store_true')
    parser.add_argument('--window-index',type=int)
    parser.add_argument('--page-top',type=int,help='Explicit source row origin for a column window')
    parser.add_argument('--depth-stratum',choices=['shallow','middle','deep'])
    args = parser.parse_args()
    if args.n < 1 or args.height < 1 or Path(args.run).name != args.run or args.run in ('.','..',''):
        parser.error('n/height must be positive; run must be one directory name')
    root = OUT / args.run
    if root.exists() and any(root.iterdir()):
        raise ValueError('Run directory is immutable and nonempty; choose a fresh run name: '+str(root))
    if not (noise.NOISE / 'measurements.json').exists():
        noise.fit()
    c = Composer(stats_path=args.stats,state_path=args.state,noise_path=args.noise)
    root.mkdir(parents=True, exist_ok=True)
    noise.write(root/'generation-stats.json',c.stats)
    c.stats_path=root/'generation-stats.json'
    c.stats_hash=hashlib.sha256(c.stats_path.read_bytes()).hexdigest()
    results = []
    refused=[];family_coverage={};transfers=[]
    try:
        eligible = sorted({card['doc_id'] for card in c.cards.values() if c.available(card['doc_id'])[0]})
        if args.mode == 'starter':
            groups = defaultdict(list)
            for did in sorted(k[4:] for k in c.stats['levels'] if k.startswith('doc:')):
                groups[c.family(did)].append(did)
            jobs = []
            for family in sorted(k[7:] for k in c.stats['levels'] if k.startswith('family:')):
                ids=groups.get(family,[])
                candidates=[];diagnostics={}
                for did in ids:
                    coverage=c.coverage(did);diagnostics[did]=coverage
                    if coverage['supported_composition_mass']<.95:
                        continue
                    try:
                        Sampler(c.stats,did,family,c.available(did)[0],args.seed).sequence(2000)
                    except ValueError as error:
                        diagnostics[did]['refusal']=str(error)
                        continue
                    candidates.append(did)
                candidates.sort(key=lambda did:(-c.stats['levels']['doc:'+did].get('complete_human_rows',0),
                    c.docs[did]['width']*c.docs[did]['height']))
                family_coverage[family]=dict(status='partial',training_documents=len(ids),documents=diagnostics,
                    requested_columns=3,requested_pages=2,eligible_documents=candidates,
                    direct_synthetic_coverage=bool(candidates))
                evidence_rows={did:c.stats['levels']['doc:'+did].get('coverage',{}).get('known_rows',0) for did in ids}
                total=sum(evidence_rows.values())
                family_coverage[family]['supported_mass_overall']=(sum(evidence_rows[d]*diagnostics[d]['by_zone'].get('all',0.) for d in ids)/total) if total else None
                family_coverage[family]['composition_rows']=total
                family_coverage[family]['supported_mass_by_document']={d:diagnostics[d]['supported_composition_mass'] for d in ids}
                if candidates:
                    # Exercise every eligible document. First slots prefer independent
                    # wells; scan variants are explicitly retained as the same well.
                    policy=noise.read(paths.DATA/'normalized/split-policy.json');seen_wells=set()
                    diverse=[];variants=[]
                    for did in candidates:
                        well=policy['document_groups'][did]
                        (variants if well in seen_wells else diverse).append(did);seen_wells.add(well)
                    candidates=diverse+variants
                    family_coverage[family]['eligible_documents']=candidates
                    jobs += [('column',candidates[i%len(candidates)],i,family) for i in range(max(3,len(candidates)))]
                    jobs += [('page',candidates[i%len(candidates)],i,family) for i in range(2)]
        else:
            if args.mode == 'page' and not args.doc:
                parser.error('page requires --doc of a training well')
            ids = [args.doc] if args.doc else [did for did in eligible if c.family(did) == args.family]
            if not ids:
                parser.error('Specify --doc or a family with available generated patterns')
            jobs = [(args.mode, ids[i % len(ids)], i, c.family(ids[i % len(ids)])) for i in range(args.n)]
        frozen_state=c.snapshot_inputs(root)
        for j, (mode, did, i, family) in enumerate(jobs):
            folder = root / f'{mode}-{slug(family)}-{did}-{i:02d}'
            try:
                if mode == 'page':
                    _, truth = c.page(did,args.seed+j,folder,context_consistent=args.context_consistent or i%2==1)
                else:
                    _, truth = c.column(did,args.seed+j,args.height,folder,iteration=args.iteration,
                        page_top=args.page_top,window_index=args.window_index if args.window_index is not None else i,
                        depth_stratum=args.depth_stratum)
                    if args.appearance_transfer and not transfers:
                        transfers.append(str(c.transfer(folder,truth)))
            except ValueError as error:
                if args.mode!='starter':
                    raise
                refused.append(dict(doc=did,family=family,mode=mode,reason=str(error)))
                print('PARTIAL refused',did,mode,str(error),flush=True)
                continue
            results.append(dict(path=str(folder), mode=mode, doc=did, family=family, seed=truth['seed'],
                supervised_rows=truth['supervised_rows'],changed_pixel_fraction=truth['changed_pixel_fraction'],
                localization_only=truth['localization_only'],task=truth['task'],
                lithology_augmentation=truth['lithology_augmentation'],
                context_consistent=truth['context_consistent'],appearance_status=truth['noise']['status'],
                coverage_status=truth['coverage']['status'],
                rendered_classes=sorted({l['class_id'] for l in truth['layers'] if l['class_id']!=GAP})))
        by_doc = defaultdict(list)
        for r in results:
            if r['mode'] == 'column':
                by_doc[r['doc']].append(r['path'])
        sheets = []
        for did, folders in by_doc.items():
            output = root / f'sheet-{did}.png'
            c.sheet(did, folders, output, rows=min(1200,args.height))
            sheets.append(str(output))
        if sheets:
            with Image.open(sheets[0]) as im:
                im.save(root / 'sheet.png')
        diagnostics=c.diagnostic_sheets(root)
        if hasattr(c,'family_sheets'):diagnostics+=c.family_sheets(root)
        frozen_state=c.snapshot_inputs(root)
        for family,coverage in family_coverage.items():
            coverage['columns']=sum(r['family']==family and r['mode']=='column' for r in results)
            coverage['pages']=sum(r['family']==family and r['mode']=='page' for r in results)
            coverage['rendered_classes']=sorted({cid for r in results if r['family']==family for cid in r['rendered_classes']})
            coverage['actual_documents']=sorted({r['doc'] for r in results if r['family']==family})
            if coverage['columns']>=3 and coverage['pages']==2 and all(
                coverage['documents'][did]['status']=='representative' for did in coverage['actual_documents']) and all(
                r['appearance_status']=='measured' for r in results if r['family']==family):
                coverage['status']='representative'
        c.assert_context_current()
        qualified=bool(results) and all(r.get('coverage_status')=='representative' and r['appearance_status']=='measured' for r in results)
        qualified=qualified and all(v['status']=='representative' for v in family_coverage.values())
        ledger=exposure_ledger(c.stats,results,family_coverage)
        noise.write(root/'coverage-ledger.json',ledger)
        noise.write(root / 'manifest.json', dict(schema='asm-2',status='refused' if not results else ('representative' if qualified else 'partial'),
                    stats_path=str(root/'generation-stats.json'),stats_sha256=c.stats_hash,
                    label_snapshot=c.stats['inputs']['normalized_snapshot'],ancestor_wells=c.stats['levels']['global']['source_wells'],
                    context_sha256=c.context_hash,
                    state_path=frozen_state,
                    family_coverage=family_coverage,refused=refused,appearance_transfers=transfers,
                    rejected_cards=c.rejected_cards,
                    outputs=results, sheets=sheets,diagnostic_sheets=diagnostics, seed=args.seed))
        print('PASS compose' if results else 'PARTIAL compose', args.run, len(results), 'samples', 'stats', c.stats_path, flush=True)
    finally:
        c.close()


if __name__ == '__main__':
    main()
