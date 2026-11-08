"""Copied into the private frozen client. No credential-file reader exists."""
import os
from datetime import datetime,timezone

AUTHORIZATION=None
INJECTION_METHOD='manager-secret-to-process-env-v1'


class ClientHold(RuntimeError):
    pass


def credential_status():
    present=bool(os.environ.get('JEV_API_KEY'))
    method=INJECTION_METHOD if os.environ.get('JEV_CREDENTIAL_INJECTION')==INJECTION_METHOD else 'none-approved'
    return dict(credentialPresent=present,injectionMethod=method)


def require_environment():
    status=credential_status()
    if not status['credentialPresent'] or status['injectionMethod']!=INJECTION_METHOD:
        raise ClientHold('Approved process-environment credential is absent. New API calls are held.')
    return os.environ['JEV_API_KEY']


def authorize_case(value):
    global AUTHORIZATION
    AUTHORIZATION=value


def request(entries,queries,cache):
    body,files,key=payload(entries,queries)
    auth=AUTHORIZATION
    if not auth:raise ClientHold('No registry-bound case authorization')
    path=Path(cache)/(key+'.json');old=Path(auth['legacyCache'])/(key+'.json')
    for candidate in [path,old]:
        if candidate.exists():
            if key not in auth['allowedPacketHashes']:raise ClientHold('An identical cached packet has unresolved or excluded lineage. No duplicate call is permitted.')
            cached=json.loads(candidate.read_text(encoding='utf-8'))
            if cached.get('packet_sha256')!=key:raise ClientHold('Packet cache hash mismatch')
            return dict(cells=parse(cached['response'],queries,entries),cached=True,seconds=0,observed_fresh_seconds=cached['seconds'],attempts=0,usage=cached['response'].get('usage',{}),packet_sha256=key)
    secret=require_environment()
    if hashlib.sha256(Path(auth['registryPath']).read_bytes()).hexdigest()!=auth['registrySha256']:raise ClientHold('Registry changed; new API calls held')
    if hashlib.sha256(Path(auth['bindingPath']).read_bytes()).hexdigest()!=auth['bindingSha256']:raise ClientHold('Case binding changed; new API calls held')
    headers={'Authorization':'Bearer '+secret,'TypeSafe-Danger-Zone-Images':'true'}
    files={'request':(None,json.dumps(body),'application/json'),**files};start=time.perf_counter()
    for attempt in range(3):
        if Path(auth['yieldPath']).exists():raise ClientHold('Owner requested a checkpoint yield; no new API request may start')
        if datetime.now(timezone.utc)>=datetime.fromisoformat(auth['deadline']):raise ClientHold('Deadline reached; no new API request may start')
        try:response=requests.post('https://api.typesafe.ai/v1/systemone',headers=headers,files=files,timeout=120)
        except Exception:raise ClientHold('API transport failed; completed packets remain saved') from None
        if response.status_code not in [429,500,502,503] or attempt==2:break
        try:delay=min(10,max(.25,float(response.headers.get('retry-after',1))))
        except ValueError:delay=1
        time.sleep(delay)
    elapsed=time.perf_counter()-start
    if not response.ok:raise ClientHold('Jev HTTP '+str(response.status_code)+'; response body omitted')
    raw=response.json()
    # Never persist credential material even if an upstream error echoes headers.
    encoded=json.dumps(raw,ensure_ascii=False)
    if secret in encoded:raw=json.loads(encoded.replace(secret,'[REDACTED]'))
    cells=parse(raw,queries,entries)
    write(path,dict(packet_sha256=key,seconds=elapsed,response=raw,attempts=attempt+1,model=MODEL))
    auth['allowedPacketHashes'].add(key)
    write(Path(cache)/'usage'/auth['caseId']/(key+'.json'),dict(caseId=auth['caseId'],imageSha256=auth['imageSha256'],registrySha256=auth['registrySha256'],packetSha256=key))
    return dict(cells=cells,cached=False,seconds=elapsed,attempts=attempt+1,usage=raw.get('usage',{}),packet_sha256=key)
