"""Score existing sources exclusively on reviewed rows. No gold leaks into human GT."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from labeltool.common import ROOT, HUMAN, GOLD, load_rows, external_sources, normalize


def compare(pred, human):
    """Missing predictions count as wrong. Boundaries in unreviewed gaps are ignored.

    Narrowed rows ("one of these" answers carrying `candidates`) are not scored."""
    human=[r for r in human if not r.get('candidates')]
    n=sum(r['bottom_y']-r['top_y'] for r in human)
    correct=covered=0
    for truth in human:
        for row in pred:
            overlap=max(0,min(truth['bottom_y'],row['bottom_y'])-max(truth['top_y'],row['top_y']))
            covered+=overlap
            if normalize(truth['lithology'])==normalize(row['lithology']): correct+=overlap
    hb={r[e] for r in human for e in ('top_y','bottom_y')}
    def touches(y): return any(r['top_y']<=y<=r['bottom_y'] for r in human)
    pb={r[e] for r in pred for e in ('top_y','bottom_y') if touches(r[e])}
    tolerance=max(4,min(20,.01*(max(hb)-min(hb)))) if hb else 4
    pairs=sorted((abs(p-h),p,h) for p in pb for h in hb if abs(p-h)<=tolerance)
    used_p,used_h=set(),set()
    for _,p,h in pairs:
        if p not in used_p and h not in used_h: used_p.add(p);used_h.add(h)
    return dict(n_rows=n,correct_rows=correct,covered_rows=covered,boundary_hits=len(used_h),
                human_boundaries=len(hb),predicted_boundaries=len(pb))


def source_paths(root=ROOT):
    root=Path(root)
    sources={}
    out=root/'out'
    for folder in sorted(out.iterdir()) if out.exists() else []:
        if folder.is_dir() and not folder.name.startswith('_'):
            paths={p.stem:p for p in folder.glob('*.csv') if '__synthetic_' not in p.stem}
            if paths: sources['strategy/'+folder.name]=paths
    for who in ('A','B'):
        paths={p.stem:p for p in (root/'gt'/'silver'/who).glob('*.csv')}
        if paths: sources['silver/'+who]=paths
    manifest=root/'labelling'/'out'/'manifest.json'
    from labeltool.common import read_json
    mapping=read_json(manifest,{})
    folders=root/'labelling'/'out'
    for folder in sorted(folders.iterdir()) if folders.exists() else []:
        if not folder.is_dir(): continue
        meta=read_json(folder/'summary.json',mapping.get(folder.name,{}))
        name=meta.get('log')
        if name and (folder/'consensus.csv').exists():
            sources.setdefault('labelling/consensus',{})[name]=folder/'consensus.csv'
    return sources


def score_sources(humans, sources):
    table=[]
    for source, logs in sorted(sources.items()):
        totals={k:0 for k in ('n_rows','correct_rows','covered_rows','boundary_hits','human_boundaries','predicted_boundaries')}
        for name, truth in humans.items():
            pred=logs.get(name,[])
            if isinstance(pred,(str,Path)): pred=load_rows(pred)
            result=compare(pred,truth)
            for k in totals: totals[k]+=result[k]
        n=totals['n_rows']
        denominator=totals['human_boundaries']+totals['predicted_boundaries']
        table.append(dict(source=source,**totals,row_accuracy=totals['correct_rows']/n if n else None,
                   boundary_f1=2*totals['boundary_hits']/denominator if denominator else None))
    return table


def wilson(successes,n):
    if not n:return [0.,1.]
    p=successes/n;z=1.959963984540054
    centre=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,centre-half),min(1,centre+half)]


def narrowed_summary(human_dir=HUMAN,events=None):
    """Counts of active "one of these" answers in the journal (read-only); never scored for accuracy."""
    if events is None:
        path=Path(human_dir)/'labels.jsonl'
        journal=[json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()] if path.exists() else []
        undone={e['target'] for e in journal if e.get('action')=='undo'}
        events=[e for e in journal if e.get('action')=='label' and e['event_id'] not in undone]
    events=[e for e in events if e.get('candidates') and e['log']!=GOLD]
    return dict(n=len(events),rows=sum(e['bottom_y']-e['top_y'] for e in events),
                logs=len({e['log'] for e in events}))


def score_human(root=ROOT,human_dir=HUMAN):
    humans={p.stem:load_rows(p) for p in Path(human_dir).glob('*.csv')}
    humans={k:v for k,v in humans.items() if v and k!=GOLD}
    return score_sources(humans,source_paths(root))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args()
    table=score_human()
    narrowed=narrowed_summary()
    if args.json: print(json.dumps(dict(table=table,narrowed=narrowed),indent=2))
    else:
        print(f'{"Source":45} {"Rows":>9} {"Covered":>9} {"Accuracy":>10} {"Hits / GT":>12} {"B-F1":>8}')
        for r in table:
            acc=f'{r["row_accuracy"]:.1%}' if r['row_accuracy'] is not None else '—'
            bf=f'{r["boundary_f1"]:.1%}' if r['boundary_f1'] is not None else '—'
            print(f'{r["source"]:45} {r["n_rows"]:9} {r["covered_rows"]:9} {acc:>10} {r["boundary_hits"]:5}/{r["human_boundaries"]:<6} {bf:>8}')
        print(f'Narrowed ("one of these", not scored): {narrowed["n"]} intervals, {narrowed["rows"]} rows')
