"""Prepare a private draft; freeze only after non-network checks pass."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'data/jev-e2e-20261108'
BUNDLE=OUT/'env-only-resume-v1'


def main():
    BUNDLE.mkdir(exist_ok=True)
    if (BUNDLE/'FROZEN.json').exists():raise RuntimeError('Immutable bundle already frozen')
    source=(HERE.parent/'bakeoff/strategies/jev_sourcecards.py').read_text(encoding='utf-8');lines=source.splitlines(keepends=True);tree=ast.parse(source)
    names={'digest','write','image_hash','card','ranks','payload','parse','classify'}
    helpers='\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
    header='''from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,io,json,time,uuid
from pathlib import Path
import numpy as np
from PIL import Image
import requests
MODEL='jev-1.14.0'
VERSION='sourcecards-e2e-v1'
BATCH=4
WORKERS=4
'''
    client=header+'\n'+helpers+'\n'+(HERE/'env_request_template.py').read_text(encoding='utf-8')
    assert 'strategies._jev' not in client and '_key(' not in client
    (BUNDLE/'env_client.py').write_text(client,encoding='utf-8')
    control=(HERE/'jev_boat_control.py').read_text(encoding='utf-8').split('\ndef compare():')[0]
    control=control.replace('HERE=Path(__file__).resolve().parent',"HERE=Path("+repr(str(HERE))+')').replace('from strategies.jev_sourcecards import image_hash','from env_client import image_hash')
    (BUNDLE/'control.py').write_text(control,encoding='utf-8')
    api=(HERE/'jev_boat_api.py').read_text(encoding='utf-8').split("\nif __name__=='__main__':")[0]
    api=api.replace("HERE=Path(__file__).resolve().parent;OUT=HERE/'data/jev-e2e-20261108';sys.path.insert(0,str(HERE.parent/'bakeoff'))",'HERE=Path('+repr(str(HERE))+ ");OUT=HERE/'data/jev-e2e-20261108'")
    api=api.replace('from strategies import jev_sourcecards as api','import env_client as api\nALLOWED_SOURCE_IDS=set()')
    api=api.replace("ledger=read(ledger_path) if ledger_path.exists() else {}","ledger=read(ledger_path) if ledger_path.exists() else {}\n    private_ledger=OUT/'env-only-semantic-decisions.json'\n    if private_ledger.exists():ledger.update(read(private_ledger))\n    blocked_keys={k for k,v in ledger.items() if v.get('source') not in ALLOWED_SOURCE_IDS}\n    ledger={k:v for k,v in ledger.items() if v.get('source') in ALLOWED_SOURCE_IDS}")
    api=api.replace("        path=record.get('diagnostics')","        if record['case_id'] not in ALLOWED_SOURCE_IDS:continue\n        path=record.get('diagnostics')")
    api=api.replace("    reused={};missing=[]","    reused={};missing=[]\n    if any(query_key(book,q['pixel_sha256']) in blocked_keys for q in queries):raise api.ClientHold('Cached crop lineage is unresolved or excluded; no duplicate call is permitted')")
    api=api.replace("OUT/'cache/jev-api'","OUT/'cache/jev-api-env-v1'").replace('api.write(ledger_path,ledger)','api.write(private_ledger,ledger)')
    start=api.index("    calibration=preprocessing.get('calibration');reference=")
    end=api.index("    decision_path=",start)
    api=api[:start]+"    calibration=preprocessing.get('calibration')\n"+api[end:]
    (BUNDLE/'api_control.py').write_text(api,encoding='utf-8')
    print('ENV_ONLY_DRAFT',str(BUNDLE))


if __name__=='__main__':main()
