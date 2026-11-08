"""Persistent dataset scheduler, publication, retention and fleet feedback."""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
import zipfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from . import paths
from .build import _write
from .dataset import DatasetBuilder, digest, read


@contextmanager
def writer_lock(root, filename='.writer.lock'):
    """OS-released lock: crashed jobs cannot strand future builds."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / filename).open('a+b') as f:
        f.seek(0)
        f.write(b'0')
        f.flush()
        f.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def snapshot(state_path=paths.STATE, reviews=paths.DATA / 'reviews', proposals=paths.PROPOSALS,
             ratings_path=paths.RATINGS):
    state = read(state_path, {})
    patterns = {c['key']: digest([c.get('signature'), c.get('source'), c.get('verdict'), c.get('method')])
                for c in state.get('cards', [])}
    files = []
    for folder, glob in [(Path(reviews), '*.json'), (Path(proposals), '*/*.json')]:
        for p in sorted(folder.glob(glob)):
            try:
                st = p.stat()
                files.append([str(p), st.st_size, st.st_mtime_ns])
            except FileNotFoundError:
                pass
    for p in [Path(ratings_path), paths.DATA / 'normalized/current.json', paths.DATA / 'normalized/split-policy.json']:
        if p.exists():
            st = p.stat()
            files.append([str(p), st.st_size, st.st_mtime_ns])
    return dict(patterns=patterns, files=digest(files))


def trigger(baseline, current, last_build, now):
    a, b = baseline.get('patterns', {}), current.get('patterns', {})
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    any_change = bool(changed or baseline.get('files') != current.get('files'))
    due = len(changed) >= 10 or (any_change and now - last_build >= 1800)
    return dict(due=due, changed_patterns=changed, pattern_count=len(changed), any_change=any_change,
                reason='10-pattern threshold' if len(changed) >= 10 else '30-minute change window' if due else 'waiting')


def versions(root):
    return sorted([p for p in Path(root).glob('v[0-9]*') if p.is_dir() and p.name[1:].isdigit()
                   and (p / 'stats.json').exists()], key=lambda p: int(p.name[1:]))


def metrics(e):
    return dict(macro=e.get('fake_to_real', {}).get('matcher', {}).get('macro_accuracy'),
                discriminator=e.get('discriminator', {}).get('augmented', {}).get('accuracy'))


def feedback(folder, e):
    stats = read(folder / 'stats.json', {})
    classes = {c['class_id']: c for c in stats.get('classes', [])}
    issues = {}
    for r in e.get('fake_to_real', {}).get('matcher', {}).get('per_class', []):
        c = classes.get(r['class_id'], {})
        if r['accuracy'] is not None and r['accuracy'] < 80:
            issues[r['class_id']] = dict(class_id=r['class_id'], name=r['name'], accuracy=r['accuracy'],
                real_n=r['n'], wrong_ratings=0, action='Check the human label and cross-well name; remake the tile if the motif differs.',
                evidence=dict(version=folder.name, evaluation='eval.json#/fake_to_real/matcher/per_class',
                              source=c.get('source'), method=c.get('method')))
    for cid, count in e.get('owner_audit', {}).get('wrong_classes', {}).items():
        c = classes.get(cid, {})
        issue = issues.setdefault(cid, dict(class_id=cid, name=c.get('name'), accuracy=None, real_n=0,
                    action='Remake the tile following the owner Wrong ratings; check the label.',
                    evidence=dict(version=folder.name, evaluation='eval.json#/owner_audit',
                                  source=c.get('source'), method=c.get('method'))))
        issue['wrong_ratings'] = count
        issue['evidence']['owner_votes'] = e.get('owner_audit', {}).get('wrong_evidence', {}).get(cid, [])
        issue['evidence']['reviewer_note'] = c.get('reviewer_note', '')
    snap = read(folder / 'lab-snapshot.json', {})
    prefill = __import__('genlab.score', fromlist=['load_prefill']).load_prefill()
    estimated = Counter()
    from feedback import canonical
    for c in classes.values():
        if c.get('has_real_example'):
            continue
        for r in prefill.get(c['document'], []):
            if canonical(r.get('suggestion') or '') == c['name']:
                estimated[c['class_id']] += max(0, int(r['bottom_y']) - int(r['top_y']))
    for c in snap.get('cards', []):
        if c['key'] not in classes or classes[c['key']].get('has_real_example'):
            continue
        if c.get('source') == 'unlisted':
            doc = c['doc_id']
            # Proposal inventory's share estimates are not geological truth.
            from . import proposals
            p = next((p for p in proposals.inventory(doc) if p['legend_id'] == c['legend_id']), {})
            from .build import labstore
            d = labstore.Store().get(doc)
            column = d['boxes']['interpreted_lithology_column']
            share_rows = (column[3] - column[1]) * float(p.get('share_of_column_rows', 0) or 0)
            estimated[c['key']] = max(estimated[c['key']], int(p.get('estimated_rows', share_rows) or 0))
    without = [dict(class_id=cid, name=classes[cid]['name'], doc=classes[cid]['doc'], estimated_column_rows=n,
                    evidence='machine prefill/fleet row estimates; not truth', action='Needs real examples: accept a representative column box.')
               for cid, n in estimated.most_common(60)]
    # Include unknown row weights explicitly so unlisted coverage cannot disappear.
    present = {r['class_id'] for r in without}
    for cid, c in classes.items():
        if not c.get('has_real_example') and cid not in present:
            without.append(dict(class_id=cid, name=c['name'], doc=c['doc'], estimated_column_rows=0,
                                evidence='row weight unavailable', action='Needs real examples.'))
    return dict(version=folder.name, at=time.time(), issues=sorted(issues.values(),
        key=lambda r: (-r.get('wrong_ratings', 0), r['accuracy'] if r['accuracy'] is not None else 101)),
        without_real_examples=without, agents_started=False)


def publish(root, folder):
    root, folder = Path(root), Path(folder)
    e = read(folder / 'eval.json', {})
    prior = [p for p in versions(root) if int(p.name[1:]) < int(folder.name[1:])]
    prev = read(prior[-1] / 'eval.json', {}) if prior else {}
    m, pm = metrics(e), metrics(prev)
    delta = {k: round(m[k] - pm[k], 3) if m[k] is not None and pm[k] is not None else None for k in m}
    flags = []
    if delta['macro'] is not None and delta['macro'] < -2:
        flags.append('REGRESSION: macro fake→real fell more than 2 points')
    if delta['discriminator'] is not None and delta['discriminator'] > 5:
        flags.append('REGRESSION: discriminator rose more than 5 points')
    # Changed truth sets make deltas descriptive, not paired estimates.
    if prev and e.get('truth_wells') != prev.get('truth_wells'):
        flags.append('Validation well set changed; metrics are not directly paired')
    elif prev and e.get('truth_fingerprint') != prev.get('truth_fingerprint'):
        flags.append('Human validation labels changed; metrics are not directly paired')
    regression = any(s.startswith('REGRESSION') for s in flags)
    old_stats = read(prior[-1] / 'stats.json', {}) if prior else {}
    stats = read(folder / 'stats.json', {})
    a = {c['class_id']: c for c in old_stats.get('classes', [])}
    b = {c['class_id']: c for c in stats.get('classes', [])}
    changes = dict(added=sorted(set(b) - set(a)), removed=sorted(set(a) - set(b)),
                   changed=sorted(k for k in set(a) & set(b) if digest(a[k]) != digest(b[k])))
    report = dict(version=folder.name, previous=prior[-1].name if prior else None, metrics=m,
                  deltas=delta, flags=flags, regression=regression, changes=changes,
                  eligible_latest=e.get('status') == 'passed' and not regression)
    _write(folder / 'publication.json', report)
    if report['eligible_latest']:
        # A portable atomic pointer (Windows symlinks require privileges).
        _write(root / 'latest.json', dict(version=folder.name))
        (root / 'latest').write_text(folder.name + '\n', encoding='utf-8')
    _write(root / 'feedback.json', feedback(folder, e))
    history = {r['version']: r for r in read(root / 'history.json', [])}
    history[folder.name] = dict(version=folder.name, patterns=stats.get('patterns'), samples=stats.get('samples'),
        created_at=stats.get('created_at'), counts_by_source=stats.get('counts_by_source'),
        counts_by_split=stats.get('counts_by_split'), metrics=m, truth_n=e.get('truth_n'),
        status=e.get('status'), publication=report, sheets=stats.get('sheets', []))
    history = sorted(history.values(), key=lambda r: int(r['version'][1:]))
    _write(root / 'history.json', history)
    lines = ['# Dataset changelog', '', 'Deltas are percentage points against the preceding version. Every score and n is in its eval.json.', '']
    for item in history:
        r = item['publication']
        lines += [f"## {item['version']}", '',
            f"{item['patterns']} patterns / {item['samples']} samples; real eval n={item['truth_n']}. Status: {item['status']}.",
            f"Changes: {json.dumps(r.get('changes', {}))}",
            f"Metrics: {json.dumps(r.get('metrics', {}))}; deltas: {json.dumps(r.get('deltas', {}))}.",
            '; '.join(r.get('flags', [])) or 'No regression detected.', '']
    (root / 'CHANGELOG.md').write_text('\n'.join(lines), encoding='utf-8')
    # Precompute a streaming-download ZIP without holding it in server RAM.
    zip_tmp = root / (folder.name + '.zip.tmp')
    with zipfile.ZipFile(zip_tmp, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file():
                z.write(p, (Path(folder.name) / p.relative_to(folder)).as_posix())
    zip_tmp.replace(root / (folder.name + '.zip'))
    retain(root)
    return report


def retain(root):
    root = Path(root).resolve()
    vv = versions(root)
    latest = read(root / 'latest.json', {}).get('version')
    for p in vv[:-5]:
        if p.name == latest or (p / 'keep').exists() or read(p / 'stats.json', {}).get('keep'):
            continue
        target = p.resolve()
        if target.parent != root or not target.name[1:].isdigit():
            raise ValueError('Unsafe dataset retention target')
        shutil.rmtree(target)
        (root / (p.name + '.zip')).unlink(missing_ok=True)


def api_index(root):
    root = Path(root)
    vv = []
    for p in versions(root):
        s, e = read(p / 'stats.json', {}), read(p / 'eval.json', {})
        vv.append(dict(version=p.name, created_at=s.get('created_at'), patterns=s.get('patterns'),
            samples=s.get('samples'), counts_by_source=s.get('counts_by_source'), counts_by_split=s.get('counts_by_split'),
            metrics=metrics(e), truth_n=e.get('truth_n'), status=e.get('status', 'pending'),
            publication=read(p / 'publication.json', {}), sheets=s.get('sheets', []),
            download_ready=(root / (p.name + '.zip')).exists()))
    history = {r['version']: dict(r, retained=False) for r in read(root / 'history.json', [])}
    for r in vv:
        history[r['version']] = dict(r, retained=True)
    all_versions = sorted(history.values(), key=lambda r: int(r['version'][1:]))
    return dict(versions=all_versions, latest=read(root / 'latest.json', {}).get('version'),
                evaluation=read(versions(root)[-1] / 'eval.json', {}) if vv else {},
                feedback=read(root / 'feedback.json', {}), loop=read(root / 'loop-state.json', {}))


def loop(root, once=False):
    with writer_lock(root, '.loop.lock'):
        _run_loop(root, once)


def _run_loop(root, once=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    saved = read(root / 'loop-state.json', {})
    vv = versions(root)
    stats = read(vv[-1] / 'stats.json', {}) if vv else {}
    completed_after_state = stats.get('created_at', 0) > saved.get('last_build', 0)
    baseline = ((stats.get('watch_fingerprint') if completed_after_state else saved.get('baseline'))
                or stats.get('watch_fingerprint') or snapshot())
    last_build = ((stats.get('created_at') if completed_after_state else saved.get('last_build'))
                  or stats.get('created_at') or time.time())
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'LOOP start pid', os.getpid(), 'poll=120s threshold=10 timeout=1800s', flush=True)
    _write(root / 'loop-state.json', dict(pid=os.getpid(), checked_at=time.time(), phase='starting',
        baseline=baseline, last_build=last_build, error=None))
    while True:
        try:
            now = time.time()
            current = snapshot()
            gate = trigger(baseline, current, last_build, now)
            if not vv or gate['due']:
                from .dataset_eval import evaluate
                _write(root / 'loop-state.json', dict(pid=os.getpid(), checked_at=time.time(), phase='building',
                    baseline=baseline, last_build=last_build, gate=gate, error=None))
                print(time.strftime('%H:%M:%S'), 'LOOP trigger', gate['reason'], 'changed', gate['pattern_count'], flush=True)
                with writer_lock(root):
                    version = DatasetBuilder(root).build()
                    _write(root / 'loop-state.json', dict(pid=os.getpid(), checked_at=time.time(), phase='evaluating',
                        version=version.name, baseline=baseline, last_build=last_build, gate=gate, error=None))
                    evaluate(version)
                    publication = publish(root, version)
                baseline = read(version / 'stats.json')['watch_fingerprint']
                last_build = time.time()
                vv = versions(root)
                print(time.strftime('%H:%M:%S'), 'LOOP built', version.name, publication['flags'], flush=True)
            _write(root / 'loop-state.json', dict(pid=os.getpid(), checked_at=time.time(), baseline=baseline,
                last_build=last_build, gate=gate, poll_seconds=120, phase='idle', error=None))
            print(time.strftime('%H:%M:%S'), 'LOOP heartbeat', gate['reason'], 'changed', gate['pattern_count'], flush=True)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
            _write(root / 'loop-state.json', dict(pid=os.getpid(), checked_at=time.time(), baseline=baseline,
                                                 last_build=last_build, phase='error', error=error[-3000:]))
        if once:
            break
        # Short sleeps permit normal termination; process is detached by WMI.
        for _ in range(12):
            time.sleep(10)
