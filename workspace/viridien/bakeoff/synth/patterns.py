"""Build the stable, complete pattern catalogue without parsing or changing logs."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from synth.common import ROOT, ANALYSIS, DATA, family, norm, slug, read_json, write_json, relative


def pattern_id(fam, labels, label):
    sig = '|'.join(sorted(set(norm(x) for x in labels)))
    return f'{slug(fam)}-{hashlib.sha1(sig.encode()).hexdigest()[:6]}/{norm(label)}'


def target(share, maximum, variety=0):
    ratio = share / maximum if maximum else 0
    return max(10, min(40, round(10 + 20 * math.sqrt(ratio) + 10 * variety)))


def build_catalog(mix, swatches):
    groups = defaultdict(list)
    for row in mix:
        labels = sorted({norm(e['label']) for e in swatches.get(row['log'], [])})
        groups[family(row['log']), tuple(labels)].append(row)
    patterns, types = [], []
    for (fam, sig), rows in sorted(groups.items()):
        logs = sorted(r['log'] for r in rows)
        tid = f'{slug(fam)}-{hashlib.sha1("|".join(sig).encode()).hexdigest()[:6]}'
        types.append(dict(id=tid, family=fam, labels=list(sig), logs=logs))
        for key in sig:
            entries = [dict(e, log=log) for log in logs for e in swatches.get(log, []) if norm(e['label']) == key]
            label = sorted({e['label'] for e in entries})[0]
            count = sum(r['col_h'] * sum(v for k, v in r.get('share', {}).items() if norm(k) == key) for r in rows)
            patterns.append(dict(pid=f'{tid}/{key}', family=fam, legend_type=tid,
                                 labels=list(sig), label=label, logs=logs, swatches=entries,
                                 rows=round(count, 2), share=0, variety=0, n_target=10,
                                 candidate_count=0, status='not_mined'))
    total = sum(r['col_h'] for r in mix) or 1
    maxima = defaultdict(float)
    for p in patterns:
        p['share'] = p['rows'] / total
        maxima[p['family']] = max(maxima[p['family']], p['share'])
    for p in patterns: p['n_target'] = target(p['share'], maxima[p['family']])
    return dict(version=1, signature_encoding='sorted normalized labels joined by |; UTF-8',
                share_basis='label_mix share * col_h; ambiguous and unnamed rows remain unallocated',
                logs=len(mix), legend_types=types, patterns=patterns,
                no_candidates=[p['pid'] for p in patterns])


def main():
    catalog = build_catalog(read_json(ANALYSIS / 'label_mix.json'), read_json(ANALYSIS / 'swatches.json')['logs'])
    dest = DATA / 'swatches'
    dest.mkdir(parents=True, exist_ok=True)
    native = {}
    for log in sorted({log for p in catalog['patterns'] for log in p['logs']}):
        folder = ROOT / 'out' / '_cache' / 'legend_v2'
        meta = read_json(folder / (log + '.json'), {})
        for e in meta.get('entries', []):
            if e.get('is_lithology') and e.get('swatch_file'):
                native[log, norm(e['label'])] = folder / e['swatch_file']
    for p in catalog['patterns']:
        for e in p['swatches']:
            source = ANALYSIS / 'swatches' / e['file']
            output = dest / source.name
            if not output.exists(): shutil.copyfile(source, output)
            e['analysis_file'] = e['file']
            e['file'] = relative(output)
            source = native.get((e['log'], norm(e['label'])))
            if source and source.exists():
                output = dest / ('native-' + hashlib.sha1(str(source).encode()).hexdigest()[:16] + '.png')
                if not output.exists(): shutil.copyfile(source, output)
                e['native_file'] = relative(output)
    write_json(DATA / 'catalog.json', catalog)
    print(json.dumps(dict(logs=catalog['logs'], legend_types=len(catalog['legend_types']),
                          patterns=dict(Counter(p['family'] for p in catalog['patterns']))), indent=2))


if __name__ == '__main__': main()
