"""Pattern fleet driver: keeps up to MAX_LIVE Codex jobs busy on kira.

Each unit = (document, kind) with kind 'legend' or 'unlisted'. A unit runs
make -> review -> (remake the non-confident ones -> review) up to MAX_ROUNDS.
Restart-safe: state is in fleet/state.json and job ids are deterministic.

    python -X utf8 data/genlab/fleet/driver.py       (from pattern_lab)
"""
import ctypes, json, os, subprocess, sys, time, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROP = ROOT / 'data/genlab/proposals'
JOBS = HERE / 'jobs'
SOL = 'C:/Users/hssil/.claude/skills/sol-fleet/scripts/sol_jobs.py'
ACCOUNT = 'kira181204'
MAX_LIVE = int(os.environ.get('FLEET_MAX_LIVE', 16))
MODE = os.environ.get('FLEET_MODE', 'first')
MIN_FREE_GB = 6
LADDER = {'legend': [('luna', 'high'), ('terra', 'high'), ('sol', 'high')],
          'unlisted': [('sol', 'medium'), ('sol', 'high')]}
REVIEW = ('terra', 'medium')
TERMINAL = {'done', 'failed', 'cancelled', 'error'}
ENV = dict(os.environ, PYTHONIOENCODING='utf-8')


def log(*a):
    line = time.strftime('%H:%M:%S ') + ' '.join(map(str, a))
    print(line, flush=True)
    with open(HERE / 'driver.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def free_gb():
    class M(ctypes.Structure):
        _fields_ = [('l', ctypes.c_ulong), ('load', ctypes.c_ulong), ('tot', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                    ('a', ctypes.c_ulonglong), ('b', ctypes.c_ulonglong), ('c', ctypes.c_ulonglong), ('d', ctypes.c_ulonglong), ('e', ctypes.c_ulonglong)]
    m = M(); m.l = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.avail / 2**30


def statuses(prefix):
    out = subprocess.run([sys.executable, SOL, 'status', '--prefix', prefix], capture_output=True, text=True, env=ENV, timeout=120).stdout
    res = {}
    for line in out.splitlines():
        if line.startswith('SOLJOB '):
            try:
                d = json.loads(line[7:])
                res[d['job']] = d
            except Exception:  # noqa: BLE001
                pass
    return res


def other_live():
    """kira jobs outside the fleet that are still running (e.g. gen1108-*)."""
    n = 0
    for p in ('gen1108-',):
        n += sum(1 for d in statuses(p).values() if d.get('state') not in TERMINAL)
    return n


COMMON = (HERE / 'prompts/common.md').read_text(encoding='utf-8')


def fill(t, **kw):
    t = t.replace('{COMMON}', COMMON)
    for k, v in kw.items():
        t = t.replace('{' + k + '}', str(v))
    return t


def proposals(doc, kind, rnd=None):
    out = {}
    for f in glob.glob(str(PROP / doc / f'{kind}-*.json')):
        n = os.path.basename(f)
        if 'review' in n or 'scan' in n:
            continue
        try:
            d = json.load(open(f, encoding='utf-8'))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or ('legend_id' not in d and 'unlisted_id' not in d):
            continue
        key = str(d.get('legend_id') if kind == 'legend' else d.get('unlisted_id'))
        if rnd is None or d.get('round') == rnd:
            out[key] = d
    return out


def review(doc, kind, rnd):
    f = PROP / doc / f'{kind}-review-r{rnd}.json'
    try:
        return json.load(open(f, encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None


SWEEP_LADDER = [('sol', 'high'), ('sol', 'xhigh')]
SWEEP_FOCUS = ('\nSWEEP: every entry listed failed, was rejected/unsure, or was never started in an earlier pass. '
               'Read each entry\'s `board_state` and `last_note` (the previous reviewer or maker note) and fix that problem. '
               '`human_boxes` are column boxes the owner labelled by hand for this entry: they are real, confirmed examples; use them first. '
               'Earlier attempts and sheets may exist in data/genlab/proposals/{DOC}/ (other rounds); learn from them, and overwrite '
               'legend-<id>.json and its tile only for the entries listed here.\n')


class Unit:
    def __init__(self, doc, kind, name, n, wid=None, base=0):
        self.doc, self.kind, self.name, self.n = doc, kind, name, n
        self.wid, self.base = wid or doc, base
        self.round, self.phase, self.focus, self.note = base + 1, 'make', (SWEEP_FOCUS.replace('{DOC}', doc) if base else ''), ''

    @property
    def key(self):
        return f'{self.kind}:{self.wid}'

    def job(self):
        tag = ('l' if self.kind == 'legend' else 'u') + ('m' if self.phase == 'make' else 'v')
        return f'fx-{tag}-{self.doc[:10]}-r{self.round}'

    def prompt(self):
        if self.phase == 'review':
            return fill((HERE / 'prompts/reviewer.md').read_text(encoding='utf-8'), DOC=self.doc, NAME=self.name, ROUND=self.round, KIND=self.kind)
        t = (HERE / f'prompts/{self.kind}_maker.md').read_text(encoding='utf-8')
        t = t.replace('data/genlab/fleet/work/{DOC}.json', 'data/genlab/fleet/work/' + self.wid + '.json')
        return fill(t, DOC=self.doc, NAME=self.name, ROUND=self.round, N=self.n, FOCUS=self.focus)

    def model(self):
        if self.phase == 'review':
            return REVIEW
        lad = SWEEP_LADDER if self.base else LADDER[self.kind]
        return lad[min(self.round - self.base, len(lad)) - 1]

    def max_rounds(self):
        return self.base + len(SWEEP_LADDER if self.base else LADDER[self.kind])


def start(u):
    jid = u.job()
    JOBS.mkdir(exist_ok=True)
    pf = JOBS / f'{jid}.md'
    pf.write_text(u.prompt(), encoding='utf-8')
    model, mode = u.model()
    cmd = [sys.executable, SOL, 'start', '--job', jid, '--run', 'fx', '--account', ACCOUNT, '--model', model, '--mode', mode,
           '--sandbox', 'danger-full-access', '--cwd', str(ROOT), '--prompt-file', str(pf), '--deadline-s', '7200', '--wait', '0']
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=120)
    log('start', jid, model, mode, (r.stdout.strip()[-200:] or r.stderr.strip()[-300:]).replace('\n', ' '))


def advance(u, st):
    """Called when u's current job is terminal."""
    state = st.get('state')
    if u.phase == 'make':
        made = {k: v for k, v in proposals(u.doc, u.kind, u.round).items() if v.get('status') == 'made'}
        if state != 'done' and not made:
            u.note = f'maker {state}: {st.get("error")}'
            log('maker failed', u.job(), st.get('error'))
            if u.round < u.max_rounds():
                u.round += 1
                return
            u.phase = 'done'
            return
        if not made:
            u.phase, u.note = 'done', 'nothing made'
            return
        u.phase = 'review'
        return
    if u.phase == 'review':
        rv = review(u.doc, u.kind, u.round)
        if rv is None:
            u.note = f'review {state} without file: {st.get("error")}'
            log('review missing', u.job(), st.get('error'))
            u.phase = 'done'
            return
        allp = proposals(u.doc, u.kind)
        redo = []
        for k, p in allp.items():
            v = rv.get(k) if p.get('round') == u.round else None
            if p.get('status') == 'failed' or (p.get('round') == u.round and p.get('status') == 'made' and (not v or v.get('verdict') != 'confident')):
                redo.append((k, p, v))
        if redo and u.round < u.max_rounds():
            u.round += 1
            u.phase = 'make'
            lines = '\n'.join(f'- {k} ({p.get("label") or p.get("description", "")}): previous status {p.get("status")}; '
                              f'reviewer: {(v or {}).get("verdict", "-")}: {(v or {}).get("note", p.get("notes", ""))}' for k, p, v in redo)
            what = 'legend entries' if u.kind == 'legend' else 'unlisted groups'
            u.focus = (SWEEP_FOCUS.replace('{DOC}', u.doc) if u.base else '') + (f'\nTHIS ROUND: only remake these {what} from round {u.round - 1}; an independent reviewer did not accept them. '
                       f'Fix exactly what the note says (look at the earlier sheet `{u.kind}-sheet-r{u.round - 1}*.png`). Leave every other entry untouched.\n{lines}\n')
            return
        u.phase = 'done'


def sweep_units():
    idx = json.load(open(HERE / 'work/_sweep.json'))
    units = []
    for d in idx:
        w = json.load(open(HERE / f'work/{d["wid"]}.json', encoding='utf-8'))
        units.append(Unit(d['doc_id'], 'legend', w['name'], d['pending'], wid=d['wid'], base=10 * (d['chunk'] + 1)))
    units.sort(key=lambda u: (-sum(bool(e.get('human_boxes')) for e in json.load(open(HERE / f'work/{u.wid}.json', encoding='utf-8'))['entries']), u.wid))
    return units


def main():
    if MODE == 'sweep':
        return run(sweep_units(), HERE / 'state-sweep.json')
    idx = json.load(open(HERE / 'work/_index.json'))
    names = {d['doc_id']: json.load(open(HERE / f'work/{d["doc_id"]}.json', encoding='utf-8'))['name'] for d in idx}
    units = []
    order = sorted(idx, key=lambda d: -d['share'])
    for d in order:
        if d['pending']:
            units.append(Unit(d['doc_id'], 'legend', names[d['doc_id']], d['pending']))
    for d in order:
        units.append(Unit(d['doc_id'], 'unlisted', names[d['doc_id']], 0))
    return run(units, HERE / 'state.json')


def run(units, sf):
    saved = json.load(open(sf)) if sf.exists() else {}
    for u in units:
        s = saved.get(u.key)
        if s:
            u.round, u.phase, u.focus, u.note = s['round'], s['phase'], s.get('focus', ''), s.get('note', '')
    log('driver start', len(units), 'units, max live', MAX_LIVE)
    while True:
      try:
        st = statuses('fx-')
        for u in units:
            if u.phase == 'done':
                continue
            j = st.get(u.job())
            if j and j.get('state') in TERMINAL:
                before = (u.round, u.phase)
                advance(u, j)
                log('advance', u.key, before, '->', (u.round, u.phase), u.note)
        live = sum(1 for j in st.values() if j.get('state') not in TERMINAL) + other_live()
        waiting = [u for u in units if u.phase != 'done' and u.job() not in st]
        # finish started units first (review / next round), then fresh units in priority order
        waiting.sort(key=lambda u: (u.round == 1 and u.phase == 'make', units.index(u)))
        for u in waiting:
            if live >= MAX_LIVE:
                break
            if free_gb() < MIN_FREE_GB:
                log('low memory, waiting', round(free_gb(), 1), 'GB')
                break
            start(u)
            live += 1
            time.sleep(2)
        json.dump({u.key: dict(round=u.round, phase=u.phase, focus=u.focus, note=u.note, job=u.job()) for u in units},
                  open(sf, 'w'), indent=1)
        prog = dict(at=time.time(), live=live, units=len(units),
                    done=sum(u.phase == 'done' for u in units),
                    by_phase={p: sum(u.phase == p for u in units) for p in ('make', 'review', 'done')})
        json.dump(dict(prog, mode=MODE), open(HERE / ('progress-sweep.json' if MODE == 'sweep' else 'progress.json'), 'w'), indent=1)
        if all(u.phase == 'done' for u in units):
            log('all units done')
            break
      except Exception as ex:  # noqa: BLE001  keep the fleet alive
        import traceback
        log('tick error', repr(ex), traceback.format_exc()[-600:].replace(chr(10), ' | '))
      time.sleep(45)


if __name__ == '__main__':
    main()
