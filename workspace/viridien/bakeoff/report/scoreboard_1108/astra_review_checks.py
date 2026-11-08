"""Review checks. Use synthetic pixels and declared input paths only."""
from pathlib import Path
import collections
import hashlib
import importlib.util
import json
import re
import sys

B = Path(__file__).resolve().parents[2]
R = Path(__file__).resolve().parent
sys.path.insert(0, str(B))
sys.dont_write_bytecode = True

def read(p):
    return Path(p).read_text(encoding='utf-8')

def digest(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def main():
    runner = module('astra_runner', R/'score_all.py')
    report = module('astra_report', R/'build_report.py')
    manifest = json.loads(read(R/'manifest_1108.json'))
    dev = json.loads(read(B.parent/'external_tests/eval_set_dev.json'))
    allowed = {(d['log'], d['y0'], d['y1']) for d in dev}
    assert all((c['log'], c['y0'], c['y1']) in allowed for c in manifest['cases'] if c['suite']=='ext_dev')
    checks = {'declared_cases': len(manifest['cases']), 'input_hash_mismatches': [], 'inputs_checked': 0}
    seen = set()
    for case in manifest['cases']:
        paths = [(case[k], case[h]) for k,h in [('image','image_sha256'),('annotation','annotation_sha256'),('truth_csv','truth_sha256')]]
        paths += [(p['path'],p['sha256']) for p in case['truth_sources']]
        for path, expected in paths:
            if (path,expected) in seen:
                continue
            seen.add((path,expected))
            assert 'external_vault' not in path.lower()
            actual = digest(path)
            checks['inputs_checked'] += 1
            if actual != expected:
                checks['input_hash_mismatches'].append(path)
    checks['source_drift'] = [p for p,h in manifest['source_hashes'].items() if digest(B/p) != h]
    checks['unfrozen_executed_dependency'] = 'strategies/s02_features_pelt.py'
    checks['dependency_in_manifest'] = any(Path(p).as_posix() == checks['unfrozen_executed_dependency'] for p in manifest['source_hashes'])
    dev_cases = [c for c in manifest['cases'] if c['suite']=='ext_dev']
    checks['dev_stored_wells'] = len({c['well'] for c in dev_cases})
    checks['dev_provisional_page_groups'] = dict(collections.Counter(re.sub(r'_p\d+$','',c['log']) for c in dev_cases))

    def rec(case, strategy, score, start):
        return dict(case_id=case,strategy=strategy,score=score,row_acc=score,boundary_f1=1.,cluster_ari=0.,
                    suite='ext_dev',log='same-page',well='same-well',family='x',y0=start,y1=start+100,truth_sha256='t')
    disjoint = [rec('A','a',1.,0),rec('B','b',0.,100)]
    pair = report.pairs(report.log_means(disjoint))[0]
    checks['disjoint_window_pair_accepted'] = {'mean_delta':pair['mean_delta'],'logs':pair['logs'],'ci95':pair['ci95']}
    assert pair['mean_delta'] == 1. and pair['logs'] == 1
    summary, paired = runner.summarize(disjoint)
    checks['runner_disjoint_matched_logs'] = next(p['metrics']['score']['log_ci']['n'] for p in paired if p['suite']=='ext_dev')
    from strategies import scoreboard_v4 as adapter
    native = adapter._native()
    checks['dev_native_cache_entries'] = sum(c['log'] in native for c in dev_cases)
    checks['native_cache_logs'] = len(native)
    from harness import score, io
    def row(a,b,label):
        return dict(top_y=a,bottom_y=b,lithology=label,cluster_id=0)
    checks['strict_slash_accuracy'] = score.score([row(0,100,'Clay / Shale')],[row(0,100,'Shale')],(0,100))['row_acc']
    checks['shared_alias_accuracy'] = score.score([row(0,100,'Sand and Sandstone')],[row(0,100,'Sandstone')],(0,100))['row_acc']
    assert checks['strict_slash_accuracy'] == 0 and checks['shared_alias_accuracy'] == 1
    from PIL import Image
    png = R/'astra_loader_probe.png'
    Image.new('RGB',(20,100),'white').save(png)
    log = io.Log(Image.open(png),{'depth_column':[0,0,2,100]},'astra_synthetic_png')
    try:
        io.read_rgb(log,'depth_column')
        checks['png_loader'] = 'accepted'
    except Exception as ex:
        checks['png_loader'] = f'{type(ex).__name__}: {ex}'
    finally:
        log.close()
    checks['image_extensions'] = dict(collections.Counter(Path(c['image']).suffix.lower() for c in manifest['cases']))
    (R/'astra_review_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)
    # The one legacy CSV test writes under io.ROOT. Keep it in the review area.
    io.ROOT = R/'astra_test_root'
    (io.ROOT/'out/_cache').mkdir(parents=True,exist_ok=True)
    import pytest
    result = pytest.main([str(R/'test_scoreboard.py'),str(R/'test_report.py'),
                          str(B/'strategies/test_scoreboard_v4.py'),str(B/'harness/test_score.py'),
                          '-q','--tb=short','-p','no:cacheprovider','--basetemp='+str(R/'astra_pytest_tmp')])
    print('ASTRA CHECK EXIT',result,flush=True)
    return int(result)

if __name__ == '__main__':
    raise SystemExit(main())
