"""Stratified reference sample. No model calls unless --label or --bench is set."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'classify' / 'out'
QUEUE = ROOT / 'labeltool' / 'data' / 'queue.jsonl'
SAMPLE = OUT / 'sample2.jsonl'
REFERENCE = OUT / 'sample2_reference.jsonl'
LEDGER = ROOT / 'labelling' / 'out' / 'calls.json'
SEED = 5822026
MODELS = {'S': ('gpt-6.1-sol', 'high'), 'T': ('gpt-6.1-terra', 'medium')}
TIERS = ('low', 'medium', 'high')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def column_boxes(rows):
    from harness.io import DATASET
    return {log: read(DATASET / 'bbox_annotation' / (log + '.json'))['bounding_boxes']
            ['interpreted_lithology_column'] for log in sorted({r['log'] for r in rows})}


def _allocation(buckets, target):
    """Integral min-cost flow: log coverage, tier balance, log balance, height balance.

    Convex slot costs redistribute unavailable tiers without inventing candidates.
    Every used log has capacity 15. The first slot strongly rewards covering it.
    """
    graph = defaultdict(list)

    def edge(a, b, capacity, cost):
        forward = [b, len(graph[b]), capacity, cost]
        backward = [a, len(graph[a]), 0, -cost]
        graph[a].append(forward)
        graph[b].append(backward)
        return forward

    logs = sorted({k[0] for k in buckets})
    for tier in TIERS:
        for k in range(1, target + 1):
            edge('source', ('tier', tier), 1, 100000 * (2*k-1))
        for height in range(3):
            for k in range(1, target + 1):
                edge(('tier', tier), ('height', tier, height), 1,
                     10*(2*k-1) + (height-TIERS.index(tier)) % 3)
    tracked = {}
    for (log, tier, height), candidates in sorted(buckets.items()):
        tracked[(log, tier, height)] = edge(('height', tier, height), ('log', log), len(candidates), 0)
    for log in logs:
        for k in range(1, min(15, sum(len(v) for (l, _, _), v in buckets.items() if l == log)) + 1):
            edge(('log', log), 'sink', 1, -1000000000 if k == 1 else 100*(2*k-1))
    for _ in range(target):
        distances = {'source': 0}
        previous = {}
        pending = deque(['source'])
        queued = {'source'}
        while pending:
            node = pending.popleft()
            queued.remove(node)
            for i, (other, _, capacity, cost) in enumerate(graph[node]):
                candidate = distances[node] + cost
                if capacity and candidate < distances.get(other, float('inf')):
                    distances[other] = candidate
                    previous[other] = (node, i)
                    if other not in queued:
                        pending.append(other)
                        queued.add(other)
        if 'sink' not in previous:
            break
        node = 'sink'
        while node != 'source':
            parent, index = previous[node]
            forward = graph[parent][index]
            forward[2] -= 1
            graph[node][forward[1]][2] += 1
            node = parent
    return {key: len(buckets[key]) - item[2] for key, item in tracked.items()}


def select(rows, boxes, seed=SEED):
    families = defaultdict(list)
    for row in rows:
        families[row['family']]  # Keep even a wholly ineligible family in the audit.
        a, b = int(row['top_y']), int(row['bottom_y'])
        box = boxes[row['log']]
        if ('log06' in row['log'].lower() or b-a < 16 or a < box[1]+30 or b > box[3]-30):
            continue
        if row['tier'] not in TIERS:
            raise ValueError('Unexpected queue tier')
        families[row['family']].append(dict(id=row['id'], patch_id='sample2_' + row['id'],
            log=row['log'], family=row['family'], tier=row['tier'], top_y=a, bottom_y=b, height=b-a))
    chosen, audit = [], []
    for family, candidates in sorted(families.items()):
        buckets = defaultdict(list)
        for i, row in enumerate(sorted(candidates, key=lambda r: (r['height'], r['id']))):
            row['height_stratum'] = min(2, i*3//len(candidates))
            buckets[(row['log'], row['tier'], row['height_stratum'])].append(row)
        counts = Counter(r['log'] for r in candidates)
        target = min(60, sum(min(15, n) for n in counts.values()))
        allocation = _allocation(buckets, target)
        selected = []
        rng = random.Random(f'{seed}:{family}')
        for key, values in sorted(buckets.items()):
            values.sort(key=lambda r: r['id'])
            rng.shuffle(values)
            selected.extend(values[:allocation[key]])
        selected.sort(key=lambda r: (r['log'], r['top_y'], r['id']))
        chosen.extend(selected)
        audit.append(dict(family=family, n=len(selected), requested=60, shortfall=60-len(selected),
            logs_covered=len({r['log'] for r in selected}), logs_available=len(counts),
            tiers=dict(Counter(r['tier'] for r in selected)),
            heights=dict(Counter(r['height_stratum'] for r in selected)),
            per_log=dict(sorted(Counter(r['log'] for r in selected).items()))))
    return chosen, audit


def sheets(rows):
    by_family = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_family[row['family']][row['log']].append(row)
    # Cover every log's first sheet before later sheets; interleave families so
    # reaching the shared cap does not consistently exclude the last family.
    result = []
    for offset in range(0, 15, 8):
        queues = [deque(group[offset:offset+8] for _, group in sorted(logs.items()) if group[offset:offset+8])
                  for _, logs in sorted(by_family.items())]
        while any(queues):
            for queue in queues:
                if queue:
                    result.append(queue.popleft())
    return result


class CallCap(RuntimeError):
    pass


class ReaderFailure(RuntimeError):
    def __init__(self, message, record=None, cache_hit=False):
        super().__init__(message)
        self.record = record
        self.cache_hit = cache_hit


@contextmanager
def ledger_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lockpath = path.with_suffix('.sample2.lock')
    deadline = time.monotonic() + 15
    while True:
        try:
            descriptor = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except (FileExistsError, PermissionError):
            if time.monotonic() >= deadline:
                raise CallCap('Shared ledger busy; no call launched')
            time.sleep(.02)
    try:
        yield
    finally:
        os.close(descriptor)
        lockpath.unlink()


class CallBudget:
    """Reserve before launch; failed launches consume quota. Lock across our processes."""
    def __init__(self, path=LEDGER, max_calls=None):
        self.path = Path(path)
        self.remaining = max_calls
        self.lock = threading.Lock()

    def reserve(self, key, model, effort, digest):
        with self.lock, ledger_lock(self.path):
            ledger = read(self.path) if self.path.exists() else {'limit': 220, 'calls': []}
            limit = min(220, int(ledger.get('limit', 220)))
            if len(ledger['calls']) >= limit or self.remaining is not None and self.remaining <= 0:
                raise CallCap('Shared 220-call cap or --max-calls reached')
            call = dict(number=len(ledger['calls'])+1, key=key, model=model, effort=effort,
                        hash=digest, started=time.time())
            ledger['calls'].append(call)
            ledger['limit'] = limit
            dump(self.path, ledger)
            if self.remaining is not None:
                self.remaining -= 1
            return call

    def finish(self, call, ok, seconds):
        with self.lock, ledger_lock(self.path):
            ledger = read(self.path)
            ledger['calls'][call['number']-1].update(ok=ok, seconds=seconds)
            dump(self.path, ledger)


class Reader:
    def __init__(self, budget, cache=None, runner=None):
        self.budget = budget
        self.cache = Path(cache) if cache is not None else OUT / 'sample2_responses'
        self.runner = runner or subprocess.run
        self.stopped = threading.Event()

    def invoke(self, png, prompt, model, effort, key):
        digest = hashlib.sha256(png.read_bytes() + prompt.encode() + model.encode() + effort.encode()).hexdigest()
        path = self.cache / (digest + '.json')
        if path.exists():
            cached = read(path)
            if cached.get('ok'):
                return cached, True
            self.stopped.set()
            raise ReaderFailure('Cached failed response; no retry', cached, True)
        if self.stopped.is_set():
            raise ReaderFailure('Reader stopped after failure; no call launched')
        call = self.budget.reserve(key, model, effort, digest)
        runtime = OUT / 'sample2_runtime'
        for sub in ('tmp', 'state', 'logs'):
            (runtime / sub).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(CODEX_HOME='C:/Users/hssil/.codex-accounts/kira181204', PYTHONUTF8='1',
                   CODEX_SQLITE_HOME=str(runtime / 'state'), TEMP=str(runtime / 'tmp'), TMP=str(runtime / 'tmp'))
        script = Path(env.get('APPDATA', '')) / 'npm/node_modules/@openai/codex/bin/codex.js'
        cmd = [shutil.which('node') or 'node', str(script), 'exec', '--skip-git-repo-check',
               '--ephemeral', '--ignore-user-config', '--sandbox', 'read-only', '-m', model,
               '-c', f'model_reasoning_effort="{effort}"', '-c', 'mcp_servers={}',
               '-c', 'features.shell_tool=false', '-c', 'history.persistence="none"',
               '-c', 'web_search="disabled"', '-c', f'log_dir="{(runtime / "logs").as_posix()}"',
               '-c', f'sqlite_home="{(runtime / "state").as_posix()}"', '-i', str(png.resolve()), '-']
        started = time.perf_counter()
        record = dict(ok=False, call=call)
        try:
            proc = self.runner(cmd, input=prompt, cwd=ROOT / 'classify', env=env,
                               capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
            if proc.returncode:
                raise ReaderFailure(f'Codex exit {proc.returncode}')
            # Keep only the JSON answer. CLI diagnostics may contain credentials.
            answer = None
            for match in re.finditer(r'\{', proc.stdout):
                try:
                    answer, _ = json.JSONDecoder().raw_decode(proc.stdout[match.start():])
                    break
                except ValueError:
                    pass
            if not isinstance(answer, dict):
                raise ReaderFailure('Expected a JSON object')
            record.update(ok=True, answer=answer)
        except Exception as exc:
            record['error'] = type(exc).__name__
            self.stopped.set()
        record['seconds'] = time.perf_counter()-started
        dump(path, record)
        self.budget.finish(call, record['ok'], record['seconds'])
        if not record['ok']:
            raise ReaderFailure('Reader failed; response cached, no retry', record)
        return record, False


def legend_entries(log):
    """Use v2, keeping parser writes and copied swatches inside classify/."""
    from harness import legend_v2
    target = OUT / '_cache' / 'legend_v2'
    target.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'out' / '_cache' / 'legend_v2'
    cached = source / (log.name + '.json')
    if cached.exists() and not (target / cached.name).exists():
        for entry in read(cached)['entries']:
            shutil.copy2(source / entry['swatch_file'], target / entry['swatch_file'])
        shutil.copy2(cached, target / cached.name)
    old = legend_v2.CACHE
    try:
        legend_v2.CACHE = target
        return [e for e in legend_v2.parse(log) if e.is_lithology and e.swatch.size]
    finally:
        legend_v2.CACHE = old


def prepare_sheet(rows):
    """Eight or fewer native crops from one log, brackets outside the source pixels."""
    from PIL import Image, ImageDraw
    from classify.data import font
    from harness import io
    assert 1 <= len(rows) <= 8 and len({r['log'] for r in rows}) == 1
    log = io.open_log(rows[0]['log'])
    try:
        entries = legend_entries(log)
        vocab = list(dict.fromkeys(e.label for e in entries))
        if not vocab:
            raise ValueError('No v2 rock vocabulary for ' + log.name)
        box = log.boxes['interpreted_lithology_column']
        legend = Image.fromarray(io.read_rgb(log, 'lithology_legend'))
        crops = []
        for row in rows:
            a, b = max(box[1], row['top_y']-40), min(box[3], row['bottom_y']+40)
            patch = Image.fromarray(io.read_rgb(log, box, a, b))
            cell = Image.new('RGB', (max(patch.width+40, 260), patch.height+55), 'white')
            cell.paste(patch, (24, 50))
            draw = ImageDraw.Draw(cell)
            draw.text((3, 3), row['patch_id'], font=font(13), fill='black')
            draw.text((3, 23), f"rows [{row['top_y']}, {row['bottom_y']})", font=font(13), fill='black')
            y0, y1 = row['top_y']-a+50, row['bottom_y']-a+50-1
            draw.line([(20, y0), (9, y0), (9, y1), (20, y1)], fill='red', width=2)
            crops.append(cell)
        width = max(im.width for im in crops)
        heights = [max(im.height for im in crops[i:i+4]) for i in range(0, len(crops), 4)]
        columns = min(4, len(crops))
        canvas = Image.new('RGB', (columns*width + legend.width+20, max(sum(heights), legend.height+25)), 'white')
        for i, cell in enumerate(crops):
            canvas.paste(cell, ((i % 4)*width, sum(heights[:i//4])))
        canvas.paste(legend, (columns*width+20, 25))
        digest = hashlib.sha256(canvas.tobytes() + json.dumps(vocab).encode()).hexdigest()
        path = OUT / 'sample2_sheets' / (digest + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            canvas.save(path)
        prompt = ('Independently identify the dominant lithology inside each RED BRACKET. '
                  'Each patch is at native scale with up to 40 rows of context above and below; '
                  'ignore that context when choosing the label. All patches and the original legend at right '
                  'come from ONE log. Compare hatch strokes, colour, spacing and shape against that legend. '
                  'Use ONLY one of these exact legend names: ' + json.dumps(vocab) + '. '
                  'Use "unknown" if uncertain. Return ONLY a JSON object mapping these exact patch IDs '
                  'to labels: ' + json.dumps([r['patch_id'] for r in rows]) + '. '
                  'Do not use tools, inspect files, or infer labels from geological expectations.')
        return path, prompt, vocab
    finally:
        log.close()


def agreement_filter(rows, answers, vocabularies):
    accepted, audit = [], []
    for row in rows:
        pid = row['patch_id']
        pair = answers.get(pid, {})
        allowed = set(vocabularies.get(row['log'], []))
        s, t = pair.get('S'), pair.get('T')
        agreed = (isinstance(s, str) and s == t and s in allowed and s.casefold() != 'unknown')
        audit.append(dict(patch_id=pid, family=row['family'], log=row['log'],
                          paired='S' in pair and 'T' in pair, agreed=agreed, S=s, T=t))
        if agreed:
            accepted.append(dict(row, lithology=s, original_truth=s))
    stats = []
    for family in sorted({r['family'] for r in rows}):
        group = [r for r in audit if r['family'] == family]
        n, paired, kept = len(group), sum(r['paired'] for r in group), sum(r['agreed'] for r in group)
        stats.append(dict(family=family, n_planned=n, n_paired=paired, n_agreed=kept,
            agreement_rate=kept/paired if paired else None, retained_share=kept/n if n else None,
            logs_covered=len({r['log'] for r in group if r['agreed']})))
    return accepted, audit, stats


def label(rows, reader, workers=6, make_sheet=prepare_sheet):
    answers, vocabularies = {}, {}
    status = 'complete'

    def work(group):
        try:
            png, prompt, vocab = make_sheet(group)
        except Exception as exc:
            return group, [], {}, 'Sheet preparation failed: ' + type(exc).__name__
        results = {}
        error = None
        for who, (model, effort) in MODELS.items():
            try:
                record, _ = reader.invoke(png, prompt, model, effort, 'sample2:reference:' + group[0]['patch_id'])
                # Whitelist IDs and labels before retaining any model-supplied strings.
                results[who] = {r['patch_id']: record['answer'].get(r['patch_id'])
                               if record['answer'].get(r['patch_id']) in vocab else 'unknown' for r in group}
            except (CallCap, ReaderFailure) as exc:
                error = str(exc)
                break
        return group, vocab, results, error

    def save():
        accepted, audit, stats = agreement_filter(rows, answers, vocabularies)
        write_jsonl(REFERENCE, accepted)
        dump(OUT / 'sample2_agreement.json', dict(families=stats, patches=audit))
        dump(OUT / 'sample2_status.json', dict(status=status, references=len(accepted)))
        return stats

    # Bounded submission stops new work on cap/failure, while retaining in-flight answers.
    iterator = iter(sheets(rows))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(work, group) for group in [next(iterator, None) for _ in range(workers)] if group}
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                group, vocab, results, error = future.result()
                vocabularies[group[0]['log']] = vocab
                for who, result in results.items():
                    for pid, value in result.items():
                        answers.setdefault(pid, {})[who] = value
                if error:
                    status = error
            stats = save()
            if status == 'complete':
                for _ in completed:
                    group = next(iterator, None)
                    if group:
                        pending.add(pool.submit(work, group))
    return save()


def reference_cases():
    """Benchmark every retained interval, without the old 40-patch/12-row sampler."""
    from PIL import Image
    from classify.data import label_key
    from harness import io
    if not REFERENCE.exists():
        return
    groups = defaultdict(list)
    for row in jsonl(REFERENCE):
        groups[row['log']].append(row)
    for name, rows in sorted(groups.items()):
        log = io.open_log(name)
        try:
            entries = legend_entries(log)
            vocab = list(dict.fromkeys(e.label for e in entries))
            mapping = {label_key(v): v for v in vocab}
            key = 'sample2_' + hashlib.sha256(name.encode()).hexdigest()[:16]
            folder = OUT / 'inputs' / key
            folder.mkdir(parents=True, exist_ok=True)
            patches = []
            for row in rows:
                rgb = io.read_rgb(log, 'interpreted_lithology_column', row['top_y'], row['bottom_y'])
                path = folder / (row['patch_id'] + '.png')
                Image.fromarray(rgb).save(path)
                patches.append(dict(row, lithology=mapping[label_key(row['lithology'])],
                    patch_y0=row['top_y'], patch_y1=row['bottom_y'], rgb=rgb, path=str(path)))
            info = dict(source='SAMPLE2', family=rows[0]['family'], log=name, key=key, mode=log.mode,
                vocabulary=vocab, n_selected=len(rows), n_accepted=len(rows), excluded=[],
                column_box=log.boxes['interpreted_lithology_column'], legend_box=log.boxes['lithology_legend'],
                truth_sha256=hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest())
            yield dict(info=info, entries=entries, vocab=vocab, patches=patches,
                       legend=Image.fromarray(io.read_rgb(log, 'lithology_legend')), folder=folder)
        finally:
            log.close()


def terra_subset(rows):
    """Fixed 25% per-family subset of retained references; independent of predictions."""
    result = set()
    groups = defaultdict(list)
    for row in rows:
        groups[row['family']].append(row)
    for family, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda r: hashlib.sha256(f'{SEED}:{r["patch_id"]}'.encode()).digest())
        result.update(r['patch_id'] for r in ordered[:len(group)//4])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--label', action='store_true')
    parser.add_argument('--bench', action='store_true')
    parser.add_argument('--max-calls', type=int, help='Maximum NEW Codex calls this invocation, also subject to shared 220 cap')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_calls is not None and args.max_calls < 0:
        parser.error('workers must be positive; max-calls must be nonnegative')
    queue = jsonl(QUEUE)
    rows, audit = select(queue, column_boxes(queue))
    for family in audit:
        print(json.dumps(family, ensure_ascii=False), flush=True)
    calls = 2*len(sheets(rows))
    used = len(read(LEDGER)['calls']) if LEDGER.exists() else 0
    print(f'PLAN {len(rows)} patches; {len(sheets(rows))} sheets; {calls} reader calls; shared ledger {used}/220', flush=True)
    if calls > 220-used:
        print(f'CAP SHORTFALL: plan exceeds remaining quota by {calls-(220-used)} reader calls; partial references will be saved.', flush=True)
    if args.dry_run:
        return
    if SAMPLE.exists() and jsonl(SAMPLE) != rows:
        raise ValueError('Existing sample differs; preserve its reference/cache before changing the queue or seed')
    write_jsonl(SAMPLE, rows)
    dump(OUT / 'sample2_plan.json', dict(seed=SEED, families=audit, reader_calls=calls,
        queue_sha256=hashlib.sha256(QUEUE.read_bytes()).hexdigest(), ledger_used=used,
        reader_call_shortfall=max(0, calls-(220-used)),
        height_strata='Empirical family height terciles; ties ordered by queue ID',
        selection='Maximise log coverage, then balance tiers, then log counts, then height terciles'))
    budget = CallBudget(max_calls=args.max_calls)
    if args.label:
        for stat in label(rows, Reader(budget), args.workers):
            print(json.dumps(stat), flush=True)
    if args.bench:
        from classify.bench import run_sample2
        run_sample2(budget)


if __name__ == '__main__':
    main()
