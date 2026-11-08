"""Evidence-based progress; --json appends a reading. Legacy history stays readable.

Built measures milestones. Good averages measured coverage / passed quality gates.
Pending gates earn zero. Labels and atlas are informational (weight zero).
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
G = ROOT / 'data/genlab'
HIST = G / 'tracker.jsonl'
SOL = str(Path.home() / '.claude/skills/sol-fleet/scripts/sol_jobs.py')
WEIGHTS = dict(legend_training=3, row_coverage_training=3, unlisted=1,
               fleet_sweep=1, dataset_loop=2, assembler=3, labels=0, atlas=0)
TERMINAL = {'done', 'failed', 'cancelled', 'canceled', 'error', 'completed'}


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def jread(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def history(path=None):
    """Ignore corrupt/interrupted lines; do not reinterpret legacy pct as GOOD."""
    try:
        lines = Path(path or HIST).read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    result = []
    for line in lines:
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                result.append(row)
        except ValueError:
            pass
    return result


def jobs(prefix):
    try:
        out = subprocess.run([sys.executable, '-B', '-X', 'utf8', SOL, 'status', '--prefix', prefix],
                             capture_output=True, text=True, encoding='utf-8', timeout=45,
                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                             env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        if out.returncode:
            raise OSError('Job status command failed')
        result = {}
        for line in out.stdout.splitlines():
            if line.startswith('SOLJOB '):
                row = json.loads(line[7:])
                result[row['job']] = {k: row.get(k) for k in ('job', 'state', 'model', 'elapsed_s')}
        return result
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
        return None


def fraction(a, b):
    return min(1., max(0., a / b)) if b else 0.


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gate(name, value=None, target='', n=None, status='pending', score=None, detail=''):
    return dict(name=name, value=value, target=target, n=n, status=status,
                score=(1. if status == 'pass' else 0.) if score is None else score, detail=detail)


def coverage_gate(name, count, total, detail=''):
    value = fraction(count, total)
    return gate(name, f'{count:,}/{total:,} ({value:.1%})', '100%', total,
                'pass' if total and count >= total else 'warn' if total else 'pending', value, detail)


def metric_gate(name, value, target, n, passed, warning=False, detail=''):
    if not number(value) or not n:
        return gate(name, target=target, n=n, detail=detail)
    return gate(name, value, target, n, 'pass' if passed else 'warn' if warning else 'fail', detail=detail)


def stream(name, milestones, gates, next_step, detail=''):
    return dict(name=name, built=sum(bool(ok) for _, ok in milestones) / len(milestones),
                good=sum(g['score'] for g in gates) / len(gates) if gates else 0.,
                gates=gates, next_step=next_step, detail=detail,
                milestones=[dict(name=n, done=bool(ok)) for n, ok in milestones])


def owner_row_basis(records, cards):
    """Union page-y intervals per document; overlapping owner boxes count once."""
    keys = {(c.get('doc_id'), str(c.get('legend_id'))) for c in cards}
    all_rows, covered = {}, {}
    boxes = 0
    for r in records:
        if r.get('split') != 'train' or r.get('accepted') is False or r.get('usable_for_generation') is False:
            continue
        box = r.get('box_xyxy')
        if not isinstance(box, list) or len(box) != 4 or not all(number(v) for v in box) or box[3] <= box[1]:
            continue
        did = r.get('document_id')
        interval = (box[1], box[3])
        all_rows.setdefault(did, []).append(interval)
        boxes += 1
        if any((did, str(target.get('legend_id'))) in keys for target in r.get('targets', [])):
            covered.setdefault(did, []).append(interval)
    def length(groups):
        result = 0
        for intervals in groups.values():
            end = -float('inf')
            for top, bottom in sorted(intervals):
                result += max(0, bottom-max(top, end))
                end = max(end, bottom)
        return result
    total, good = length(all_rows), length(covered)
    return dict(rows=total, covered_rows=good, boxes=boxes, documents=len(all_rows),
                coverage=fraction(good, total), basis='union of accepted, generation-usable train owner-box page-y intervals; generated class available')


def markdown_evaluation(folder):
    """Read the evaluator's labelled tables if its JSON has not been published."""
    try:
        lines = (folder / 'eval.md').read_text(encoding='utf-8').splitlines()
    except OSError:
        return {}
    result, section, headers = {}, '', []
    def numeric(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    for line in lines:
        if line.startswith('#'):
            title = line.casefold()
            section = 'closed' if 'closed-set within-document' in title else 'disc' if 'real versus fake discriminator' in title else ''
            headers = []
        cells = [s.strip() for s in line.strip().strip('|').split('|')]
        if cells[0] in ('Model / variant', 'Variant / source'):
            headers = cells
        elif headers and len(cells) == len(headers):
            row = dict(zip(headers, cells))
            if section == 'closed' and cells[0] == 'matcher / augmented':
                metric = dict(accuracy=numeric(row.get('Top-1 %')), chance=numeric(row.get('Chance %')), n=numeric(row.get('Real n')))
                result['closed_set_within_document'] = {'matcher': {'augmented': metric}}
            if section == 'disc' and cells[0] == 'augmented':
                real, fake = numeric(row.get('Held real n')), numeric(row.get('Held fake n'))
                result['discriminator'] = {'augmented': dict(accuracy=numeric(row.get('Accuracy % (50 is chance)')),
                    n=real+fake if real is not None and fake is not None else None)}
    return result


def dataset_evidence():
    base = G / 'datasets'
    versions = sorted((p for p in base.glob('v*') if p.is_dir() and re.fullmatch(r'v\d+', p.name)),
                      key=lambda p: int(p.name[1:]))
    folder = versions[-1] if versions else base / 'pending'
    ev = jread(folder / 'eval.json') or {}
    ev = {**markdown_evaluation(folder), **ev}
    promoted = (jread(base / 'latest.json') or {}).get('version')
    # Fixed algorithm and arm; never choose the best available result after evaluation.
    closed = ev.get('closed_set_within_document', {}).get('matcher', {}).get('augmented', {})
    value, chance = closed.get('accuracy'), closed.get('chance')
    recognition = metric_gate('Closed-set fake → real top-1 (matcher, augmented)', value,
        f'> {2*chance:.3f}% (2 × {chance:.3f}% chance)' if number(chance) else '> 2 × chance',
        closed.get('n'), number(chance) and number(value) and value > 2*chance,
        detail='Within-document only; cross-log accuracy is a different task.')
    if not number(chance):
        recognition.update(value=None, status='pending', score=0.)
    elif number(value):
        recognition['value'] = f'{value:.3f}%'
    disc = ev.get('discriminator', {}).get('augmented', {})
    value = disc.get('accuracy')
    discriminator = metric_gate('Augmented discriminator', value, '≤ 60%', disc.get('n'),
                                number(value) and value <= 60)
    if number(value):
        discriminator['value'] = f'{value:.3f}%'
    audit = jread(folder / 'audit.json') or jread(folder / 'split-audit.json') or ev.get('split_audit', {})
    leaks = audit.get('test_well_leaks', audit.get('test_samples'))
    leakage = metric_gate('Test-well leakage audit', leaks, '0 test samples', audit.get('n', audit.get('samples')),
                          leaks == 0, detail='A split policy or successful build alone is not an audit.')
    try:
        content = (base / 'diag' / f'{folder.name}-report.md').read_text(encoding='utf-8')
        line = next((s for s in content.splitlines() if f'audit_dataset.py {folder.name}:' in s), '')
        if 'PASS zero test samples' in line:
            leakage = gate('Test-well leakage audit', '0 test samples', '0 test samples',
                           (jread(folder / 'stats.json') or {}).get('samples'), 'pass', detail=line.strip('- '))
    except OSError:
        pass
    return folder, ev, promoted, [recognition, discriminator, leakage]


def leaves(obj):
    if isinstance(obj, dict):
        if 'status' in obj and 'value' in obj:
            yield obj
        else:
            for value in obj.values():
                yield from leaves(value)


def assembler_evidence():
    base = G / 'assemble'
    # Exclude fixtures, smoke runs and noisy-prefill diagnostics.
    runs = base / 'bench/runs/starter'
    fidelity = jread(runs / 'fidelity.json') or {}
    metrics = [m for split in ('validation', 'test') for m in leaves(fidelity.get(split, {}).get('metrics', {}))]
    counts = Counter(m.get('status', 'pending') for m in metrics)
    samples = {s: fidelity.get(s, {}).get('samples', {}) for s in ('validation', 'test')}
    fidelity_gate = gate('Bench fidelity', dict(counts), 'all pass; 0 warn / fail', len(metrics),
        'fail' if counts['fail'] else 'warn' if counts['warn'] else 'pass' if metrics else 'pending',
        fraction(counts['pass'], len(metrics)), json.dumps(samples))
    real = jread(runs / 'realism.json') or {}
    value = real.get('balanced_accuracy', real.get('accuracy'))
    realism = metric_gate('Realism discriminator (balanced)', value, '≤ 60% (warn ≤ 70%)',
        real.get('n_real', 0) + real.get('n_synthetic', 0), number(value) and value <= .60,
        number(value) and value <= .70,
        detail=f"{real.get('real_wells', '?')} real / {real.get('synthetic_wells', '?')} synthetic wells; windows are correlated.")
    if number(value):
        realism['value'] = f'{value:.2%}'
    detector = jread(runs / 'detector.json') or {}
    human = detector.get('human_all', {})
    gap = human.get('accuracy_gap')
    detector_gate = metric_gate('Detector gap synthetic − real', gap, '|gap| ≤ 10 pp (warn ≤ 20 pp)',
        human.get('rows', human.get('labelled_rows', human.get('n'))),
        number(gap) and abs(gap) <= .10, number(gap) and abs(gap) <= .20,
        detail=detector.get('comparison_basis', 'Same detector, labelled rows in both domains.'))
    if number(gap):
        detector_gate['value'] = f'{gap*100:+.2f} pp'
    return [fidelity_gate, realism, detector_gate], samples, history(base / 'bench/history.jsonl')[-1:]


def reading():
    board = get('http://127.0.0.1:8782/board.json') or {}
    lab = get('http://127.0.0.1:8781/state.json') or {}
    atlas = get('http://127.0.0.1:8783/atlas.json')
    fleet_jobs, other_jobs = jobs('fx-'), jobs('gen1108-')
    all_jobs = {**(fleet_jobs or {}), **(other_jobs or {})}
    streams = {}
    tc = board.get('training_counts') or {}
    workable = board.get('training_total', 0) - tc.get('Skipped (not a rock fill)', 0) - tc.get('Not lithology', 0)
    approved = tc.get('In the lab', 0) + tc.get('Approved, waiting for the lab', 0)
    training_docs = {d['doc_id'] for d in board.get('documents', []) if d.get('training_allowed')}
    cards = [c for c in lab.get('cards', []) if c.get('doc_id') in training_docs]
    cur = jread(ROOT / 'data/normalized/current.json') or {}
    recs = cur.get('records', [])
    human_basis = owner_row_basis(recs, cards)

    def realism_gate(selected):
        unverified = sum(not (c.get('realism') or {}).get('crops') and
                         not (c.get('real') and c.get('evidence_kind') != 'swatch' and c.get('source') != 'swatch')
                         for c in selected)
        scored = [c for c in selected if (c.get('realism') or {}).get('crops') and number(c['realism'].get('median'))]
        passing = sum(c['realism']['median'] >= .9 for c in scored)
        g = coverage_gate('Real-crop realism median ≥ 0.9', passing, len(selected),
            f'{unverified} unverified (no real crops); {len(selected)-unverified-len(scored)} awaiting scores; {len(scored)} scored. Source reuse can inflate scores.')
        g.update(unverified=unverified, scored=len(scored), passing=passing)
        return g

    streams['legend_training'] = stream('Legend coverage', [('In lab or approved', approved >= workable and workable > 0)],
        [coverage_gate('Training entries in lab or approved', approved, workable), realism_gate(cards)],
        'Label a real interval for a blocked entry, then verify its generated repeat.')
    streams['legend_training']['built'] = fraction(approved, workable)
    cov = lab.get('coverage') or {}
    rows, total = cov.get('training_generated_column_rows', 0), cov.get('training_column_rows', 0)
    human_cov = {k: v for k, v in cov.items() if any(w in k.lower() for w in ('human', 'owner', 'labelled', 'labeled'))}
    streams['row_coverage_training'] = stream('Row coverage', [('Coverage measured', bool(cov))],
        [coverage_gate('Training rows covered (machine estimate)', rows, total, cov.get('basis', ''))],
        'Add owner boxes to check the estimated coverage.',
        'Human-labelled basis: ' + (json.dumps(human_cov) if human_cov else 'not published; machine guesses are not human truth.'))
    streams['row_coverage_training']['built'] = fraction(rows, total)
    if human_basis['rows']:
        streams['row_coverage_training']['detail'] = (f"Owner-labelled basis: {human_basis['covered_rows']:,}/{human_basis['rows']:,} rows "
            f"({human_basis['coverage']:.1%}), {human_basis['boxes']} boxes / {human_basis['documents']} documents. "
            'Union of accepted, generation-usable train boxes only; class available, not a realism test. ' +
            ('Published human coverage: ' + json.dumps(human_cov) if human_cov else 'Not whole-column ground truth.'))
    unlisted = board.get('unlisted_found', 0)
    streams['unlisted'] = stream('Unlisted patterns', [('Approved', board.get('unlisted_approved', 0) >= unlisted and unlisted > 0)],
        [realism_gate([c for c in cards if str(c.get('legend_id', '')).startswith('u')])],
        'Collect real crops and realism scores for unlisted patterns.')
    streams['unlisted']['built'] = fraction(board.get('unlisted_approved', 0), unlisted)
    ps = jread(G / 'fleet/progress-sweep.json') or {}
    reviewed = sum(e.get('state') in ('In the lab', 'Approved, waiting for the lab', 'Rejected', 'Skipped (not a rock fill)', 'Not lithology')
                   for e in board.get('entries', []) if e.get('training_allowed'))
    streams['fleet_sweep'] = stream('Fleet sweep', [('Units completed', ps.get('done') == ps.get('units') and bool(ps.get('units')))],
        [coverage_gate('Training entries with resolved review', reviewed, board.get('training_total', 0))],
        'Resolve failed and unsure reviews; completed jobs do not establish usable patterns.')
    streams['fleet_sweep']['built'] = fraction(ps.get('done', 0), ps.get('units', 0))
    folder, ev, promoted, dgates = dataset_evidence()
    streams['dataset_loop'] = stream('Dataset', [
        ('Builder', (ROOT / 'genlab/dataset.py').exists()), ('Version', folder.exists()),
        ('Evaluation', bool(ev) or (folder / 'eval.md').exists()), ('Lab Dataset tab', bool(lab.get('dataset'))),
        ('Build job done', all_jobs.get('gen1108-dataset', {}).get('state') == 'done')], dgates,
        'Run the missing version-specific audit or improve the failing evaluation gate.',
        f'Newest version: {folder.name}; promoted: {promoted or "pending"}. All gates use the newest version.')
    A = G / 'assemble'
    agates, samples, bench_history = assembler_evidence()
    milestones = [(n, (p.exists() if not pattern else any(p.glob(pattern)))) for n, p, pattern in [
        ('stats.py', ROOT / 'genlab/assemble/stats.py', ''), ('stats.json', A / 'stats.json', ''),
        ('stats report', A / 'stats-report.md', ''), ('stats watcher', A / 'stats-watch.log', ''),
        ('noise.py', ROOT / 'genlab/assemble/noise.py', ''), ('compose.py', ROOT / 'genlab/assemble/compose.py', ''),
        ('starter run', A / 'out/starter', '*/truth.json'), ('page mode', A / 'out/starter', '*/bbox_annotation.json'),
        ('bench.py', ROOT / 'genlab/assemble/bench.py', ''), ('suites', A / 'suites', '*/suite.json'),
        ('report.html', A / 'report.html', '')]]
    milestones.append(('3 jobs done', sum(j.get('state') == 'done' for k, j in all_jobs.items() if k.startswith('gen1108-asm-')) >= 3))
    streams['assembler'] = stream('Assembler', milestones, agates,
        'Reduce failing fidelity differences and finish the detector comparison.',
        'Starter benchmark; human-only fidelity, excluding noisy-prefill diagnostics and smoke fixtures.')
    suite_counts = Counter((jread(p) or {}).get('status', 'pending') for p in (A / 'suites').glob('*/suite.json'))
    streams['assembler']['detail'] += ' Suites: ' + ', '.join(f'{n} {s}' for s, n in sorted(suite_counts.items())) + '.'
    counts = Counter(r.get('split', 'unknown') for r in recs)
    first = next((r['labels'] for r in history() if number(r.get('labels'))), len(recs))
    valid = sum(r.get('split') in ('train', 'test') for r in recs)
    streams['labels'] = stream('Owner labels', [('Snapshot available', bool(cur))],
        [coverage_gate('Boxes assigned train / test', valid, len(recs))],
        'Draw one real box for a high-impact entry in Label next.',
        f"{counts['train']} train / {counts['test']} test / {len(recs)-valid} other; {len(recs)-first:+d} boxes since first reading ({first}). Informational, weight 0.")
    statuses = Counter(s.get('status', 'Unknown') for s in (atlas or {}).get('symbols', []))
    eligible = sum(n for s, n in statuses.items() if s not in ('Held out', 'Skipped'))
    streams['atlas'] = stream('Atlas', [('Atlas available', bool(atlas))],
        [coverage_gate('Symbols Done', statuses['Done'], eligible) if atlas else gate('Symbols by status', target='atlas.json available')],
        'Finish symbol evidence and review the remaining atlas entries.',
        (', '.join(f'{s}: {n}' for s, n in sorted(statuses.items())) if atlas else 'Pending atlas service.') + ' Informational, weight 0.')
    good = {k: s['good'] for k, s in streams.items()}
    built = {k: s['built'] for k, s in streams.items()}
    denominator = sum(WEIGHTS.values())
    good['overall'] = sum(good[k]*w for k, w in WEIGHTS.items()) / denominator
    built['overall'] = sum(built[k]*w for k, w in WEIGHTS.items()) / denominator
    return dict(schema=2, at=time.time(), pct=good, built=built, streams=streams, weights=WEIGHTS,
        notes={k: s['detail'] or s['next_step'] for k, s in streams.items()}, labels=len(recs),
        label_counts=dict(counts), labels_growth=len(recs)-first, labels_baseline=first,
        dataset=dict(version=folder.name, promoted=promoted), atlas_statuses=dict(statuses),
        assembler_samples=samples, bench_history=bench_history,
        human_row_basis=human_basis,
        sources=dict(board=bool(board), lab=bool(lab), atlas=bool(atlas), jobs=fleet_jobs is not None and other_jobs is not None),
        live=dict(fleet=sum(j.get('state') not in TERMINAL for j in (fleet_jobs or {}).values()),
                  other={k: j.get('state') for k, j in (other_jobs or {}).items()},
                  jobs=[j for j in all_jobs.values() if j.get('state') not in TERMINAL]))


def main():
    now = reading()
    with HIST.open('a', encoding='utf-8') as output:
        output.write(json.dumps(now, ensure_ascii=False) + '\n')
    if '--json' in sys.argv:
        print(json.dumps(now, ensure_ascii=False))
        return
    print('GOOD overall', f"{now['pct']['overall']:.1%}", '| built', f"{now['built']['overall']:.1%}")
    for key, s in now['streams'].items():
        print(f"{key:24} built {s['built']:6.1%}  good {s['good']:6.1%}")
        for g in s['gates']:
            print(f"  {g['status'].upper():7} {g['name']}: {g['value'] if g['value'] is not None else 'pending'}; target {g['target']}; n={g['n']}")


if __name__ == '__main__':
    main()
