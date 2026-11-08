"""Prepared, bounded continuation for the saved Jev run.

Never run this alongside the prior owner. `--execute` requires an explicit
released-owner receipt and an environment-only credential; it refuses the
legacy .env fallback. A transport uncertainty stops the pass for inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import psutil

HERE = Path(__file__).resolve().parent
BAKE = HERE.parents[2]
LAB = BAKE.parent / 'pattern_lab'
OUT = LAB / 'data' / 'jev-e2e-20261108'
STATE = HERE / 'continuation_state.json'
DEADLINE = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
ALLOWED = {'viridien_eval', 'human_gold', 'human_qa', 'hadrian_unqa', 'ext_dev'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def select_remaining(cases, done):
    if len(cases) != 68 or len({c['case_id'] for c in cases}) != 68:
        raise ValueError('Expected 68 unique frozen cases')
    if not set(done) <= {c['case_id'] for c in cases}:
        raise ValueError('Saved result is outside the frozen case list')
    if any(c['suite'] not in ALLOWED for c in cases):
        raise ValueError('Frozen case list has an unsupported suite')
    order = {'viridien_eval': 0, 'human_gold': 1, 'human_qa': 2, 'hadrian_unqa': 3, 'ext_dev': 4}
    return sorted((c for c in cases if c['case_id'] not in set(done)),
                  key=lambda c: (order[c['suite']], c['case_id']))


def run_bounded(cases, done, ops, *, max_cases, deadline, now, checkpoint):
    """Use injected operations so transport and failures can be tested offline."""
    pending = select_remaining(cases, done)
    finished = list(done)
    state = {'status': 'started', 'planned_remaining': len(pending), 'completed_this_pass': [],
             'max_cases': max_cases, 'cutoff_utc': deadline.isoformat()}
    checkpoint(state)
    for case in pending[:max_cases]:
        if now() >= deadline:
            state.update(status='stopped', reason='cutoff reached before next case')
            checkpoint(state)
            return state
        state.update(current=case['case_id'], phase='case')
        checkpoint(state)
        try:
            ops.case(case)
        except Exception as exc:
            # Unknown transport/API outcome may still be running. No automatic
            # retry or next case until an owner inspects the saved receipts.
            state.update(status='inspection_required', reason=type(exc).__name__,
                         message=str(exc)[:500], uncertain_case=case['case_id'])
            checkpoint(state)
            return state
        finished.append(case['case_id'])
        state['completed_this_pass'].append(case['case_id'])
        state['phase'] = 'saved_and_verified'
        checkpoint(state)
    if now() >= deadline:
        state.update(status='stopped', reason='cutoff reached before paired replay')
    else:
        try:
            ops.compare_once()
            state.update(status='stopped', reason='bounded pass and one paired replay complete')
        except Exception as exc:
            state.update(status='inspection_required', phase='paired_replay',
                         reason=type(exc).__name__, message=str(exc)[:500])
    state['remaining_after_pass'] = len(cases) - len(set(finished))
    checkpoint(state)
    return state


def verify_handoff(receipt):
    """Require the manager's explicit release and exact durable snapshot."""
    if receipt.get('state') != 'released' or receipt.get('authorized_continuation') is not True:
        raise ValueError('Explicit single-owner release is required')
    if receipt.get('results_sha256') != sha(OUT / 'results.json'):
        raise ValueError('Released checkpoint hash is stale')
    queue = read(OUT / 'boat-queue.json')
    if receipt.get('source_manifest_sha256') != queue['source_manifest_sha256']:
        raise ValueError('Released source manifest does not match the frozen queue')
    if not receipt.get('confirmed_by'):
        raise ValueError('Release must identify its confirmer')
    if receipt.get('remote_stage_verified_idle') is not True or receipt.get('ready_for_api') is not True:
        raise ValueError('Release must confirm idle remote stages and approved API continuation')
    for owner in receipt.get('prior_owner_pids', []):
        pid = int(owner['pid'])
        if psutil.pid_exists(pid):
            proc = psutil.Process(pid)
            if abs(proc.create_time() - float(owner['creation_epoch'])) < 1:
                raise ValueError('A prior owner is still alive')
    if not receipt.get('prior_owner_pids'):
        raise ValueError('Release must identify the prior owners')
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(proc.info['cmdline'] or [])
            if proc.info['pid'] != os.getpid() and ('jev_boat_control.py run' in cmd or 'run_jev_boat_repair.ps1' in cmd):
                raise ValueError('An existing Jev owner is still running')
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not os.environ.get('JEV_API_KEY'):
        raise ValueError('Environment-only Jev credential is unavailable')
    if datetime.now(timezone.utc) >= DEADLINE:
        raise ValueError('The run cutoff has passed')
    return queue


class RealOps:
    """Existing pinned worker/receipt functions, with bounded local transport."""
    def __init__(self, queue):
        sys.path.insert(0, str(LAB))
        sys.path.insert(0, str(BAKE))
        self.ctrl = importlib.import_module('jev_boat_control')
        self.api = importlib.import_module('jev_boat_api')
        # Explicitly disable the legacy file fallback in this process. The
        # frozen file is not edited and the secret is never logged or sent to VM.
        key_client = importlib.import_module('strategies._jev')
        key_client._key = lambda: os.environ.get('JEV_API_KEY')
        if sha(Path(self.ctrl.__file__)) != queue['local_control_sha256'] or sha(Path(self.api.__file__)) != queue['api_control_sha256']:
            raise ValueError('Pinned controller or API source changed')
        if sha(OUT / 'boat-packet' / 'source-manifest.json') != queue['source_manifest_sha256']:
            raise ValueError('Local packet source manifest changed')
        experiment = read(OUT / 'manifest.json')
        packet = read(OUT / 'boat-packet' / 'source-manifest.json')
        for filename, expected in experiment['source_hashes'].items():
            current = sha(filename)
            if current != expected and not (Path(filename).name == 'grading_policy.py' and
                                             current == packet['source/strategies/grading_policy.py']):
                raise ValueError('Frozen experiment source changed: ' + Path(filename).name)
        source = self.ctrl.remote('cd ' + shlex.quote(self.ctrl.REMOTE) + ' && sha256sum source-manifest.json').split()[0]
        if source != queue['source_manifest_sha256']:
            raise ValueError('Remote packet source hash differs from the frozen queue')
        if sha(OUT / 'manifest.json') != read(OUT / 'lock.json')['manifest_sha256']:
            raise ValueError('Experiment lock mismatch')
        self.original_command = self.ctrl.command
        def bounded_command(args, timeout=2800):
            remaining = (DEADLINE - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 1:
                raise TimeoutError('Cutoff reached; transport was not started')
            return self.original_command(args, timeout=min(timeout, max(1, remaining - 1)))
        self.ctrl.command = bounded_command

    def case(self, case):
        c = self.ctrl
        ident = case['case_id']
        c.stage_case(case)  # verifies image, annotation, truth and source hashes
        root = OUT / 'boat-results' / 'exports' / ident
        decision_path = OUT / 'boat-decisions' / (ident + '.json')
        remote_dir = c.REMOTE + '/exports/' + ident
        remote_export = c.remote('test -s ' + shlex.quote(remote_dir + '/export.json') +
                                 ' && sha256sum ' + shlex.quote(remote_dir + '/export.json') + ' || true')
        if not remote_export:
            c.stage('export', ident)
            remote_export = c.remote('sha256sum ' + shlex.quote(remote_dir + '/export.json'))
        if not (root / 'receipt.json').exists():
            archive = c.REMOTE + '/export-' + ident + '.tar.gz'
            command = 'cd ' + shlex.quote(c.REMOTE) + ' && tar -czf ' + shlex.quote(archive) + ' exports/' + ident + ' && sha256sum ' + shlex.quote(archive)
            digest = c.remote(command).split()[0]
            local = OUT / ('export-' + ident + '.tar.gz')
            c.get(archive, local, digest)
            self.api.extract(local, digest)
        if remote_export.split()[0] != sha(root / 'export.json'):
            raise ValueError('Local and remote frozen export hashes differ')
        exported = read(root / 'export.json')
        required_pixels = {q['pixel_sha256'] for q in exported['queries']}
        if decision_path.exists():
            decision = read(decision_path)
            if decision.get('export_sha256') != sha(root / 'export.json'):
                raise ValueError('Saved decision belongs to a different export')
        else:
            decision = None
        if decision is None or any(value is None for value in decision['by_pixel'].values()):
            # Existing exact-pixel/request caches are reused by classify. One
            # attempt only; an uncertain response requires owner inspection.
            remaining = (DEADLINE - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 1:
                raise TimeoutError('Cutoff reached before the API pass')
            # A timeout may leave an upstream request outcome unknown. The
            # caller stops for inspection and does not issue a blind retry.
            subprocess.run([sys.executable, '-X', 'utf8', str(HERE / 'api_envonly_worker.py'), ident],
                           check=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=min(900, max(1, remaining - 1)))
            decision = read(decision_path)
        if decision.get('export_sha256') != sha(root / 'export.json'):
            raise ValueError('Decision belongs to a different export')
        if any(value is None for value in decision['by_pixel'].values()):
            raise ValueError('Incomplete Jev decisions; saved for inspection')
        if set(decision['by_pixel']) != required_pixels:
            raise ValueError('Decision pixels do not match the frozen query set')
        c.put(decision_path, c.REMOTE + '/decisions/' + decision_path.name)
        # Earlier evaluation receipts do not bind the decision-file hash. For
        # unfinished local cases, re-score the verified cached decisions once.
        c.stage('evaluate', ident)
        c.stage('report', ident)
        c.collect(ident)
        c.import_case(case)
        matches = [r for r in read(OUT / 'results.json')['records'] if r['case_id'] == ident]
        if len(matches) != 1 or sha(matches[0]['prediction_csv']) != matches[0]['prediction_sha256']:
            raise ValueError('Saved case checkpoint failed its prediction hash check')

    def compare_once(self):
        self.ctrl.compare()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--handoff', type=Path, help='Explicit manager release receipt')
    parser.add_argument('--max-cases', type=int, default=41)
    args = parser.parse_args()
    manifest = read(OUT / 'manifest.json')
    saved = read(OUT / 'results.json')['records']
    done = [r['case_id'] for r in saved]
    if len(done) != len(set(done)):
        raise ValueError('Duplicate saved results')
    pending = select_remaining(manifest['cases'], done)
    if not args.execute:
        print(json.dumps({'mode': 'dry-run', 'saved': len(done), 'remaining': len(pending),
                          'cutoff_utc': DEADLINE.isoformat(), 'api_calls': 0,
                          'requires': 'owner release receipt and environment-only credential'}))
        return
    if args.handoff is None or args.max_cases < 1 or args.max_cases > 68:
        raise ValueError('Execution needs a handoff and max-cases in 1..68')
    queue = verify_handoff(read(args.handoff))
    reference = read(BAKE / 'report' / 'scoreboard_1108' / 'manifest_1108.json')
    digest = hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    if digest != '1da7b757c9846b7cf1b7d79a1f01519195d3011a6c4041ac02c05f6798795301':
        raise ValueError('Current scoreboard manifest differs from the approved freeze')
    reference_cases = {c['case_id']: c for c in reference['cases']}
    fields = ('suite', 'log', 'window', 'image_sha256', 'annotation_sha256', 'truth_sha256')
    if len(reference_cases) != 68 or any(c['case_id'] not in reference_cases or any(
            c.get(field) != reference_cases[c['case_id']].get(field) for field in fields) for c in manifest['cases']):
        raise ValueError('Experiment and scoreboard case identities differ')
    ops = RealOps(queue)
    state = run_bounded(manifest['cases'], done, ops, max_cases=args.max_cases,
                        deadline=DEADLINE, now=lambda: datetime.now(timezone.utc),
                        checkpoint=lambda value: write(STATE, value))
    print(json.dumps({'status': state['status'], 'reason': state.get('reason'),
                      'completed_this_pass': len(state['completed_this_pass'])}))
    if state['status'] == 'inspection_required':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
