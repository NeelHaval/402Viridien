import asyncio
import json
import httpx
import pytest
from app.providers import ask_gemini
from tests.test_providers import config


@pytest.mark.parametrize('status',[400,403,404])
def test_listed_retired_model_retries_once(status,caplog):
    calls=[]
    def handle(request):
        if request.method=='GET':return httpx.Response(200,json={'models':[{'name':'models/'+n,'supportedGenerationMethods':['generateContent']} for n in ['gemini-3.8-flash','gemini-flash-latest','gemini-2.5-flash']]})
        calls.append(request.url.path)
        if len(calls)==1:return httpx.Response(status,json={'error':{'message':'fixture-only'}})
        return httpx.Response(200,json={'candidates':[{'content':{'parts':[{'text':'{"highlight_id":"dominant","review_id":"scan_check"}'}]}}]})
    result=asyncio.run(ask_gemini(config(), 'facts',transport=httpx.MockTransport(handle)))
    assert len(calls)==2 and calls[1].endswith('gemini-2.5-flash:generateContent')
    assert result['model']=='gemini-2.5-flash'
    assert 'provider_status='+str(status) in caplog.text
    assert config().gemini_key not in caplog.text and '?key=' not in caplog.text


def test_quota_does_not_trigger_extra_generation():
    calls=[]
    def handle(request):
        if request.method=='GET':return httpx.Response(200,json={'models':[{'name':'models/'+n,'supportedGenerationMethods':['generateContent']} for n in ['gemini-3.8-flash','gemini-2.5-flash']]})
        calls.append(request.url.path);return httpx.Response(429)
    with pytest.raises(httpx.HTTPStatusError):asyncio.run(ask_gemini(config(),'facts',transport=httpx.MockTransport(handle)))
    assert len(calls)==1
