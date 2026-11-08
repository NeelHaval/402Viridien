"""Provider checks use public text and signed test sessions, never private well data."""
import asyncio
import base64
import json
import time

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from itsdangerous import TimestampSigner

from app import providers

SECRET = 'test-session-secret-with-more-than-thirty-two-characters'


def config(**changes):
    values = dict(origin='https://app.example.test', domain='tenant.eu.auth0.com',
                  client_id='test-client', client_secret='test-secret', session_secret=SECRET,
                  gemini_key='test-gemini-key', model='gemini-3.8-flash', required=True)
    values.update(changes)
    return providers.ProviderConfig(**values)


def app_client(**changes):
    app = FastAPI()
    providers.install(app, config(**changes))
    for path in ('/api/wells', '/wells', '/api/wells/test/csv', '/api/events', '/events', '/api/health'):
        app.add_api_route(path, lambda: {'ok': True})
    return TestClient(app, base_url='https://app.example.test')


def cookie(expires=None, secret=SECRET):
    data = {'user': {'sub': 'auth0|public-test', 'name': 'Test User', 'email': 'test@example.test',
                     'expires_at': time.time() + 3600 if expires is None else expires}}
    return TimestampSigner(secret).sign(base64.b64encode(json.dumps(data).encode())).decode()


def signed_client(**changes):
    client = app_client(**changes)
    client.cookies.set('landmarkr_session', cookie())
    return client


@pytest.mark.parametrize('path', ['/api/wells', '/wells', '/api/wells/test/csv', '/api/events', '/events', '/docs', '/openapi.json'])
def test_anonymous_access_is_denied(path):
    assert app_client().get(path).status_code == 401


def test_signed_session_and_public_status():
    client = signed_client()
    assert client.get('/api/wells').status_code == 200
    data = client.get('/api/session').json()
    assert data['user']['name'] == 'Test User'
    assert 'sub' not in data['user']
    assert 'test-secret' not in json.dumps(data)
    assert app_client().get('/api/health').status_code == 200


@pytest.mark.parametrize('session', [cookie(expires=0), cookie(secret='wrong-secret'), 'invalid'])
def test_bad_or_expired_session_is_denied(session):
    client = app_client()
    client.cookies.set('landmarkr_session', session)
    assert client.get('/api/wells').status_code == 401


def test_cross_site_request_is_denied():
    client = signed_client()
    assert client.post('/api/assistant', json={'question': 'sandstone'}, headers={'Origin': 'https://evil.test'}).status_code == 403
    assert client.post('/auth/logout', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403


def test_logout_clears_cookie():
    client = signed_client()
    result = client.post('/auth/logout', follow_redirects=False)
    assert result.status_code == 303
    assert result.headers['location'].startswith('https://tenant.eu.auth0.com/v2/logout?')
    header = result.headers['set-cookie'].lower()
    assert 'expires=thu, 01 jan 1970' in header
    assert 'httponly' in header and 'secure' in header


def test_missing_settings_fail_closed():
    with pytest.raises(ValueError, match='complete Auth0'):
        app_client(client_secret='')
    with pytest.raises(ValueError, match='HTTPS'):
        app_client(origin='http://app.example.test')
    client = app_client(required=False, client_secret='')
    assert client.get('/auth/login').status_code == 503
    assert client.get('/api/session').json()['auth_configured'] is False


def test_callback_rejects_missing_state():
    # Authlib checks state before token exchange. This must not need the provider network.
    assert app_client().get('/auth/callback?code=forged&state=wrong', follow_redirects=False).status_code == 400






def test_gemini_request_and_response():
    from tests.test_ux2_gemini import test_model_discovery_and_server_key
    test_model_discovery_and_server_key()



@pytest.mark.parametrize('status', [401, 429, 500])
def test_upstream_http_failures(status):
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(providers.ask_gemini(config(), 'Public test', httpx.MockTransport(
            lambda request: httpx.Response(status, json={'error': 'private upstream detail'}))))


def test_blocked_answer_is_not_success():
    with pytest.raises(ValueError):
        asyncio.run(providers.ask_gemini(config(), 'Public test', httpx.MockTransport(
            lambda request: httpx.Response(200, json={'status': 'failed', 'steps': []}))))


def test_oversized_provider_response_is_rejected():
    with pytest.raises(ValueError, match='size limit'):
        asyncio.run(providers.ask_gemini(config(), 'Public test', httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'x' * 65537))))


@pytest.mark.parametrize('payload', [None, [], {'status': 'completed', 'steps': None},
    {'status': 'completed', 'steps': [None]},
    {'status': 'completed', 'steps': [{'type': 'model_output', 'content': [{'type': 'text', 'text': 42}]}]}])
def test_malformed_gemini_response_is_rejected(payload):
    with pytest.raises(ValueError):
        asyncio.run(providers.ask_gemini(config(), 'Public test', httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))))




