"""Run bounded, deterministic project stress checks without network access.

Examples:
  python stress_testing/run_stress.py --profile quick
  python stress_testing/run_stress.py --profile extended --suite archives --suite workers
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import importlib.metadata
import importlib.util
import json
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
SUITES={'data':'test_contracts.py','metrics':'test_metrics.py','results':'test_results.py',
        'workers':'test_workers.py','archives':'test_archives.py','security':'test_security_boundaries.py',
        'notebook':'test_notebook.py','harness':'test_harness.py'}
DEMO_SUITES={'demo-config':'test_demo_config.py','demo-pipeline':'test_demo_pipeline.py','demo-services':'test_demo_services.py',
             'demo-security':'test_demo_security.py'}
PROFILES={'quick':{'iterations':50,'timeout':180},'extended':{'iterations':500,'timeout':600},
          'soak':{'iterations':3000,'timeout':1200}}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=PROFILES,default='quick')
    p.add_argument('--suite',action='append',choices=['all',*SUITES,*DEMO_SUITES],help='Repeat to combine suites; default all core suites')
    p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--iterations',type=int,help='Override seeded loop count, 1..10000')
    p.add_argument('--timeout',type=float,help='Whole pytest subprocess deadline in seconds, 1..1800')
    p.add_argument('--max-rss-mb',type=float,default=1536,help='Sampled pytest process-tree MiB ceiling, 64..8192')
    p.add_argument('--output',type=Path,default=HERE/'.runs')
    p.add_argument('--include-regressions',action='store_true',help='Also execute the normal project tests')
    p.add_argument('--include-demo',action='store_true',help='Add demo backend regressions and adversarial suites; excludes real-OCR model tests')
    p.add_argument('--include-real-model',action='store_true',help='With --include-demo, also run the local real-OCR crop test (may skip if unavailable)')
    p.add_argument('--fail-fast',action='store_true')
    return p


def terminate(process,observed):
    """Only stop descendants belonging to this runner, never unrelated Python."""
    import psutil
    errors=[]
    try:observed.update({p.pid:p for p in psutil.Process(process.pid).children(recursive=True)})
    except psutil.Error:pass
    for child in observed.values():
        try:
            if child.is_running():child.kill()
        except psutil.NoSuchProcess:pass
        except psutil.Error as error:errors.append(f'descendant {child.pid}: {type(error).__name__}')
    if process.poll() is None:process.kill()
    process.wait(timeout=10)
    if observed:
        try:
            _,alive=psutil.wait_procs(list(observed.values()),timeout=3)
            errors.extend(f'descendant {child.pid} still running' for child in alive)
        except psutil.Error:
            # Windows may refuse waiting on a short-lived, already-gone child.
            # Verify liveness separately rather than losing the entire report.
            for child in observed.values():
                try:
                    if child.is_running():errors.append(f'descendant {child.pid} cleanup unconfirmed')
                except psutil.NoSuchProcess:pass
                except psutil.Error:errors.append(f'descendant {child.pid} cleanup unconfirmed')
    return errors


def junit_counts(path):
    if not path.is_file():return None
    try:
        root=ET.parse(path).getroot()
        if root.tag not in ('testsuite','testsuites'):return None
        suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
        if not suites:return None
        result={key:sum(int(s.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    except (ET.ParseError,ValueError,OSError):return None
    result['passed']=result['tests']-result['failures']-result['errors']-result['skipped']
    if any(value<0 for value in result.values()):return None
    return result


def source_provenance():
    paths=[*HERE.glob('*.py'),*HERE.glob('*.cjs'),*HERE.glob('checks/*.py'),*HERE.glob('requirements*.lock'),*ROOT.glob('src/viridian402/**/*.py'),
           *ROOT.glob('scripts/*.py'),*ROOT.glob('tests/*.py'),*ROOT.glob('requirements*.lock'),
           *ROOT.glob('configs/*.json'),*ROOT.glob('demo/backend/**/*.py'),
           *ROOT.glob('demo/frontend/src/**/*.ts'),*ROOT.glob('demo/frontend/src/**/*.tsx'),
           *ROOT.glob('demo/frontend/src/**/*.css'),ROOT/'demo/frontend/index.html',ROOT/'demo/frontend/vite.config.ts',
           *ROOT.glob('demo/frontend/package*.json'),ROOT/'demo/backend/requirements.txt',
           ROOT/'notebooks/main_pipeline.ipynb',ROOT/'pyproject.toml']
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}
    def git(*args):
        try:
            result=subprocess.run(['git',*args],cwd=ROOT,text=True,capture_output=True,timeout=5)
            return result.stdout.strip() if result.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired):return None
    toplevel=git('rev-parse','--show-toplevel')
    own_git=bool(toplevel and Path(toplevel).resolve()==ROOT)
    status=git('status','--porcelain','--untracked-files=normal') if own_git else None
    return {'git_commit':git('rev-parse','HEAD') if own_git else None,'git_dirty':bool(status) if status is not None else None,
            'file_sha256':hashes}


def run(args):
    import math
    settings=PROFILES[args.profile]
    iterations=args.iterations if args.iterations is not None else settings['iterations']
    deadline=args.timeout if args.timeout is not None else settings['timeout']
    if not 0<=args.seed<=2**32-1:raise ValueError('seed must be between 0 and 4294967295')
    if args.include_real_model and not args.include_demo:raise ValueError('--include-real-model requires --include-demo')
    if not 1<=iterations<=10000:raise ValueError('iterations must be between 1 and 10000')
    if not math.isfinite(deadline) or not 1<=deadline<=1800:raise ValueError('timeout must be between 1 and 1800 seconds')
    if not math.isfinite(args.max_rss_mb) or not 64<=args.max_rss_mb<=8192:raise ValueError('max-rss-mb must be between 64 and 8192 MiB')
    for module in ('pytest','numpy','PIL','psutil'):
        if importlib.util.find_spec(module) is None:
            raise ValueError('Missing '+module+'; run python -m pip install -c requirements.lock -e ".[test,review]"')
    import psutil
    selected=list(SUITES) if not args.suite or 'all' in args.suite else list(dict.fromkeys(args.suite))
    if args.include_demo:selected=list(dict.fromkeys([*selected,*DEMO_SUITES]))
    uses_demo=args.include_demo or any(s in DEMO_SUITES for s in selected)
    if uses_demo:
        for module in ('fastapi','pandas','cv2','scipy','httpx','multipart','tifffile'):
            if importlib.util.find_spec(module) is None:
                raise ValueError('Missing demo dependency '+module+'; use a Python 3.12 environment with demo/backend/requirements.txt installed')
    paths=[HERE/'checks'/{**SUITES,**DEMO_SUITES}[s] for s in selected]
    for path in paths:
        if not path.is_file():raise ValueError('Missing stress suite: '+str(path))
    output=args.output.resolve()/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    temporary=output/'work';temporary.mkdir()
    env=os.environ.copy()
    env['VIRIDIAN_STRESS_SEED']=str(args.seed)
    env['VIRIDIAN_STRESS_ITERATIONS']=str(iterations)
    env['PYTHONIOENCODING']='utf-8'
    env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    if uses_demo:env['PYTHONPATH']=str(ROOT/'demo/backend')+os.pathsep+env['PYTHONPATH']
    for key in ('TMP','TEMP','TMPDIR'):env[key]=str(temporary)
    junit=output/'junit.xml';log=output/'pytest.log'
    command=[sys.executable,'-m','pytest','-q','--import-mode=importlib',
             '--basetemp',str(temporary/'pytest'),'--junitxml',str(junit),*map(str,paths)]
    if args.include_regressions:command.append(str(ROOT/'tests'))
    if args.include_demo:
        command.append(str(ROOT/'demo/backend/tests'))
        if not args.include_real_model:command.extend(['-m','not model'])
    if args.fail_fast:command.append('-x')
    provenance=source_provenance()
    started=time.perf_counter();peak=0.;observed={};status='running';process=None;return_code=None
    cleanup_errors=[]
    try:
        with log.open('wb') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            while process.poll() is None:
                if time.perf_counter()-started>deadline:
                    status='timeout';terminate(process,observed);break
                if log.stat().st_size>16*1024**2:
                    status='log_budget_exceeded';terminate(process,observed);break
                try:
                    parent=psutil.Process(process.pid);children=parent.children(recursive=True)
                    observed.update({p.pid:p for p in children})
                    peak=max(peak,sum(p.memory_info().rss for p in [parent,*children] if p.is_running())/1024**2)
                    if peak>args.max_rss_mb:
                        status='memory_budget_exceeded';terminate(process,observed);break
                except psutil.Error:pass
                time.sleep(.05)
            return_code=process.wait(timeout=10)
            if status=='running':status='passed' if return_code==0 else 'failed'
            if log.stat().st_size>16*1024**2:status='log_budget_exceeded'
    except KeyboardInterrupt:
        status='interrupted'
    finally:
        if process is not None:cleanup_errors=terminate(process,observed) or []
    counts=junit_counts(junit)
    if status=='passed' and (not counts or counts['passed']==0):status='no_completed_tests'
    if status=='passed' and (counts['failures'] or counts['errors']):status='failed'
    if status=='passed' and cleanup_errors:status='cleanup_incomplete'
    final_provenance=source_provenance()
    source_changed=provenance['file_sha256']!=final_provenance['file_sha256']
    if source_changed:status='source_changed'
    provenance['final_file_sha256']=final_provenance['file_sha256']
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    versions={name:importlib.metadata.version(name) for name in ('pytest','numpy','Pillow','psutil')}
    if uses_demo:
        for name in ('fastapi','starlette','pydantic','httpx','pandas','scipy','rapidocr-onnxruntime','onnxruntime','tifffile'):
            versions[name]=importlib.metadata.version(name)
    result={'schema_version':1,'status':status,'passed':status=='passed','profile':args.profile,'suites':selected,
            'seed':args.seed,'iterations':iterations,'include_regressions':args.include_regressions,'counts':counts,
            'include_demo':args.include_demo,'real_ocr_tests_included':args.include_real_model,
            'return_code':return_code,'wall_seconds':time.perf_counter()-started,'sampled_peak_tree_mib':peak,
            'limits':{'timeout_seconds':deadline,'sampled_tree_rss_mib':args.max_rss_mb,'pytest_log_bytes':16*1024**2},
            'python':sys.version,'platform':platform.platform(),'packages':versions,
            'git_commit':provenance['git_commit'],'git_dirty':provenance['git_dirty'],
            'source_changed_during_run':source_changed,'cleanup_errors':cleanup_errors,
            'report_directory':str(output),'network_requested':False,
            'note':'Resource checks sample process usage; this is not an OS sandbox or hard allocator limit. Skipped checks are reported separately.'}
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['# Stress-test run','',f'Status: **{status}**',f'Profile: {args.profile}; seed: {args.seed}; iterations: {iterations}.','',
           f'Test counts: {json.dumps(counts)}',f'Wall seconds: {result["wall_seconds"]:.2f}; sampled peak process-tree MiB: {peak:.1f}.','',
           'Detailed test output: `pytest.log`; machine-readable cases: `junit.xml`; execution metadata: `summary.json`.','',result['note']]
    (output/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 130 if status=='interrupted' else 124 if status=='timeout' else 1


def main(argv=None):
    args=parser().parse_args(argv)
    try:return run(args)
    except (ValueError,OSError,ImportError) as error:
        print('error: '+str(error),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
