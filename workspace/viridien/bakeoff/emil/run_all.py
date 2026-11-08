"""Portable resume driver; retain original shared preprocessing, never open removed wells."""
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def call(label, arguments, isolated=True, allow_failure=False):
    env=os.environ.copy()
    env.pop('LITHOLOGY_DATASET',None)
    env.setdefault('MPLCONFIGDIR',str(HERE/'cache/matplotlib'))
    command=[sys.executable,*(['-I'] if isolated else []),'-u',*map(str,arguments)]
    with (HERE/(label+'.log')).open('w',encoding='utf-8') as stream:
        result=subprocess.run(command,cwd=HERE.parents[1],env=env,stdout=stream,stderr=subprocess.STDOUT)
    print(label,'exit',result.returncode,flush=True)
    if result.returncode and not allow_failure:raise RuntimeError(f'{label} failed; inspect {label}.log')


def main():
    call('vm_port',[HERE/'port_notebook.py'])
    call('vm_validation',[HERE/'validate.py'])
    call('vm_viridien',[HERE/'evaluate.py','--domain','viridien'])
    call('vm_external',[HERE/'evaluate.py','--domain','external'])
    call('vm_incomplete',[HERE/'evaluate.py','--domain','incomplete','--modes','detect','labels'])
    if (HERE/'boundary_net.pt').exists():
        call('vm_retrained',[HERE/'evaluate.py','--domain','viridien','--modes','detect','labels',
             '--tag','retrained_live','--model',HERE/'boundary_net.pt'])
    call('vm_smoke',[HERE/'smoke_live.py'])
    call('vm_benchmark',[HERE/'benchmark_vm.py'])
    call('tests_live',['-m','pytest','bakeoff/harness','bakeoff/external','-q'],isolated=False,allow_failure=True)
    call('tests_harness_vm',['-m','pytest','bakeoff/harness','-q'],isolated=False)
    call('vm_acceptance',[HERE/'audit_results.py'])
    call('vm_report',[HERE/'live_report.py'])


if __name__=='__main__':main()
