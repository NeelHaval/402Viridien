import asyncio,json,re
from dataclasses import replace
from pathlib import Path
import httpx
import pytest
from app import providers
from app.descriptions import build_facts,complete_note,validate_selection,FactError
from tests.test_providers import config
from tests.test_descriptions import saved,make_client

def test_grounding_and_partial(saved):
    row,folder,path=saved
    facts,_=build_facts(row,folder)
    note=validate_selection('{"highlight_id":"dominant","review_id":"scan_check"}',facts)
    assert set(re.findall(r'\d+(?:\.\d+)?',note)) <= {'100','200','60.0'}
    assert len(re.split(r'(?<=[a-z%.])\. ',note)) >= 3
    with pytest.raises(FactError):validate_selection('{"highlight_id":"range","review_id":"scan_check","number":999}',facts)
    row['status']='partial'
    path.write_text('top_depth,base_depth,lithology\n,,Shale\n',encoding='utf-8')
    facts,_=build_facts(row,folder)
    async def invalid(*a,**k):return {'answer':'999 metres'}
    note,fallback,model=asyncio.run(complete_note(invalid,config(),'prompt',facts))
    assert fallback and model is None and 'insufficient evidence' in note and '999' not in note

def test_timeout_uses_15_seconds_and_fallback(saved,monkeypatch):
    import app.descriptions as d
    facts,_=build_facts(saved[0],saved[1])
    seen=[]
    async def timed(awaitable,timeout):
        seen.append(timeout);awaitable.close();raise asyncio.TimeoutError()
    monkeypatch.setattr(d.asyncio,'wait_for',timed)
    async def fake(*a,**k):return {'answer':'ignored'}
    note,fallback,model=asyncio.run(d.complete_note(fake,config(),'prompt',facts))
    assert seen==[15] and fallback and '100 to 200' in note

def test_public_endpoint_cache_and_fallback(saved,tmp_path,monkeypatch):
    client,calls=make_client(saved,tmp_path,monkeypatch,signed=False)
    r=client.post('/api/wells/well-test/description')
    assert r.status_code==200 and not r.json()['fallback']
    assert client.post('/api/wells/well-test/description').json()['cached'] and len(calls)==1

def test_fallback_not_cached(saved,tmp_path,monkeypatch):
    client,calls=make_client(saved,tmp_path,monkeypatch,output='invented 999')
    for _ in range(2):
        r=client.post('/api/wells/well-test/description')
        assert r.status_code==200 and r.json()['fallback'] and '999' not in r.json()['note']
    assert len(calls)==2

def test_model_discovery_and_server_key():
    requests=[]
    def handle(request):
        requests.append(request)
        assert request.headers['x-goog-api-key']=='test-gemini-key'
        if request.method=='GET':return httpx.Response(200,json={'models':[{'name':'models/gemini-3.8-flash','supportedGenerationMethods':['generateContent']}]})
        return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':'{"highlight_id":"range","review_id":"scan_check"}'}]}}]})
    result=asyncio.run(providers.ask_gemini(config(), 'facts', transport=httpx.MockTransport(handle)))
    assert result['model']=='gemini-3.8-flash' and len(requests)==2
    assert str(requests[1].url).endswith(':generateContent') and 'test-gemini-key' not in str(requests[1].url)


def test_partial_pixel_shares_area_and_no_raw_members(saved,tmp_path,monkeypatch):
    row,folder,path=saved
    row['status']='partial'
    path.write_text('top_depth,base_depth,top_y,bottom_y,lithology\n,,0,60,Sandstone\n,,60,100,Shale\n',encoding='utf-8')
    facts,_=build_facts(row,folder)
    assert facts['pixel_lithology_shares'][0]['percent']==60
    client,calls=make_client(saved,tmp_path,monkeypatch,output='{"highlight_id":"pixels","review_id":"depth_check"}')
    with client.app.state.db.connect() as conn:conn.execute("UPDATE wells SET lat=0,lon=0,status='partial'")
    r=client.post('/api/report/description',json={'polygon':[[-1,-1],[1,-1],[1,1],[-1,1]]})
    assert r.status_code==200 and 'members' not in r.json()
    assert '60.0%' in r.json()['note'] and 'not a depth share' in r.json()['note']


def test_alternate_model_does_not_poison_configured_cache(saved,tmp_path,monkeypatch):
    client,calls=make_client(saved,tmp_path,monkeypatch)
    # The registered closure holds fake(), which uses this supplied envelope only through generate.
    import inspect
    from app.descriptions import complete_note
    facts,_=build_facts(saved[0],saved[1])
    async def alternate(*a,**k):return {'answer':'{"highlight_id":"range","review_id":"scan_check"}', 'model':'gemini-other-flash'}
    note,fallback,model=asyncio.run(complete_note(alternate,config(),'facts',facts))
    assert not fallback and model=='gemini-other-flash'
