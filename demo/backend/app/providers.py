"""Auth0 browser sessions and the server-side Gemini assistant."""
from __future__ import annotations

import os
import json
import time
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from urllib.parse import urlencode, urlsplit

import httpx
from authlib.integrations.base_client.errors import OAuthError
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware


@dataclass(frozen=True)
class ProviderConfig:
    origin: str
    domain: str
    client_id: str
    client_secret: str
    session_secret: str
    gemini_key: str
    model: str
    required: bool

    @classmethod
    def from_env(cls):
        return cls(
            os.getenv('APP_PUBLIC_URL', 'http://localhost:8000').rstrip('/'),
            os.getenv('AUTH0_DOMAIN', '').strip(),
            os.getenv('AUTH0_CLIENT_ID', '').strip(),
            os.getenv('AUTH0_CLIENT_SECRET', '').strip(),
            os.getenv('APP_SESSION_SECRET', ''),
            os.getenv('GEMINI_API_KEY', '').strip(),
            os.getenv('GEMINI_MODEL', 'gemini-3.8-flash'),
            os.getenv('AUTH_REQUIRED', 'false').lower() == 'true',
        )

    @property
    def auth_ready(self):
        return bool(self.domain and self.client_id and self.client_secret and len(self.session_secret) >= 32)

    def validate(self):
        url = urlsplit(self.origin)
        if url.scheme not in ('http', 'https') or not url.netloc or url.path:
            raise ValueError('APP_PUBLIC_URL must be an origin with no path')
        if self.domain and (urlsplit('https://' + self.domain).netloc != self.domain or '/' in self.domain):
            raise ValueError('AUTH0_DOMAIN must be a hostname')
        if self.required and (url.scheme != 'https' or not self.auth_ready):
            raise ValueError('Required authentication needs HTTPS and complete Auth0 settings')


def problem(status: int, code: str, message: str):
    return JSONResponse({'error': {'code': code, 'message': message}}, status_code=status,
                        headers={'Cache-Control': 'no-store'})


def current_user(session):
    user = session.get('user')
    if isinstance(user, dict) and user.get('sub') and user.get('expires_at', 0) > time.time():
        return user
    return None


class AccessGate:
    """Protect API aliases, downloads and event streams with the same session."""
    def __init__(self, app, config):
        self.app, self.config = app, config

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http':
            path = scope['path'].rstrip('/') or '/'
            protected = (path.startswith('/api/') or path == '/api' or
                         path.split('/')[1] in ('wells', 'upload', 'report', 'events', 'docs', 'redoc', 'openapi.json'))
            if path in ('/api/health', '/health', '/api/session'):
                protected = False
            user = current_user(scope.get('session', {}))
            if protected and self.config.required and not user:
                return await problem(401, 'login_required', 'Sign in to use the app.')(scope, receive, send)
            if scope['method'] not in ('GET', 'HEAD', 'OPTIONS'):
                headers = dict(scope['headers'])
                origin = headers.get(b'origin', b'').decode()
                cross_site = headers.get(b'sec-fetch-site') == b'cross-site'
                if (origin and origin != self.config.origin) or cross_site:
                    return await problem(403, 'origin_rejected', 'Use the app page to send this request.')(scope, receive, send)
        await self.app(scope, receive, send)




async def ask_gemini(config: ProviderConfig, question: str, transport=None, *, system=None):
    """Discover a Flash text model available to this key; keep all credentials server-side."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(14, connect=4), transport=transport) as client:
        headers = {'x-goog-api-key': config.gemini_key}
        async def bounded(method, url, **kwargs):
            async with client.stream(method, url, **kwargs) as response:
                response.raise_for_status()
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > 65536:
                        raise ValueError('Provider response exceeds the size limit')
                    raw.extend(chunk)
                return json.loads(raw)
        listing = await bounded('GET', 'https://generativelanguage.googleapis.com/v1beta/models', headers=headers)
        if not isinstance(listing, dict) or not isinstance(listing.get('models', []), list):
            raise ValueError('Invalid model list')
        models = [m['name'].removeprefix('models/') for m in listing.get('models', [])
                  if isinstance(m, dict) and isinstance(m.get('name'), str) and 'flash' in m.get('name', '') and 'generateContent' in m.get('supportedGenerationMethods', [])
                  and not any(k in m['name'] for k in ('image', 'audio', 'tts', 'live', 'robotics'))]
        if not models:
            raise ValueError('No Flash text model is available')
        preferred=list(dict.fromkeys(['gemini-2.5-flash','gemini-flash-latest',*models]))
        model = config.model if config.model in models else next(name for name in preferred if name in models)
        payload={'contents': [{'role': 'user', 'parts': [{'text': question}]}],
              'systemInstruction': {'parts': [{'text': system or 'Select grounded facts. Return JSON only.'}]},
              'generationConfig': {'maxOutputTokens': 512, 'responseMimeType': 'application/json'}}
        async def generate(model):
            return await bounded('POST', 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent', headers=headers,json=payload)
        try:
            data=await generate(model)
        except httpx.HTTPStatusError as error:
            status=error.response.status_code
            logging.getLogger(__name__).warning('Gemini generation failed: provider_status=%d',status)
            alternate=next((name for name in preferred if name in models and name!=model),None)
            if status not in (400,403,404) or alternate is None:raise
            model=alternate
            try:data=await generate(model)
            except httpx.HTTPStatusError as retry_error:
                logging.getLogger(__name__).warning('Gemini retry failed: provider_status=%d',retry_error.response.status_code)
                raise
        if not isinstance(data, dict) or not isinstance(data.get('candidates'), list):
            raise ValueError('Invalid generation response')
        texts = [part['text'] for candidate in data.get('candidates', [])
                 if isinstance(candidate, dict) and isinstance(candidate.get('content'), dict)
                 for part in candidate['content'].get('parts', []) if isinstance(part, dict) and isinstance(part.get('text'), str)]
        if not texts:
            raise ValueError('Gemini did not return text')
        return {'answer': '\n'.join(texts), 'model': model}


def install(app: FastAPI, config: ProviderConfig | None = None):
    config = config or ProviderConfig.from_env()
    config.validate()
    app.state.providers = config
    oauth = OAuth()
    if config.auth_ready:
        oauth.register('auth0', client_id=config.client_id, client_secret=config.client_secret,
                       server_metadata_url=f'https://{config.domain}/.well-known/openid-configuration',
                       client_kwargs={'scope': 'openid profile email', 'code_challenge_method': 'S256'})
    router = APIRouter()
    limits = defaultdict(deque)

    def reserve(subject):
        now = time.monotonic()
        for key in list(limits):
            while limits[key] and now - limits[key][0] >= 60:
                limits[key].popleft()
            if not limits[key]:
                del limits[key]
        bucket = limits[subject]
        if len(bucket) >= 10:
            return False
        bucket.append(now)
        return True

    @router.get('/api/session')
    async def session(request: Request):
        user = current_user(request.session)
        public_user = {k: user[k] for k in ('name', 'email') if k in user} if user else None
        return JSONResponse({'user': public_user, 'auth_required': config.required,
                             'auth_configured': config.auth_ready, 'gemini_configured': bool(config.gemini_key),
                             'gemini_model': config.model}, headers={'Cache-Control': 'no-store'})

    @router.get('/auth/login')
    async def login(request: Request):
        if not config.auth_ready:
            return problem(503, 'auth_not_configured', 'Sign-in setup is not complete.')
        return await oauth.auth0.authorize_redirect(request, config.origin + '/auth/callback')

    @router.get('/auth/callback')
    async def callback(request: Request):
        if not config.auth_ready:
            return problem(503, 'auth_not_configured', 'Sign-in setup is not complete.')
        try:
            token = await oauth.auth0.authorize_access_token(request)
            claims = token.get('userinfo')
            if not claims or not claims.get('sub'):
                raise ValueError('No verified identity')
        except (OAuthError, ValueError, httpx.HTTPError):
            request.session.clear()
            return problem(400, 'login_failed', 'Sign-in failed. Start sign-in again from the app.')
        request.session.clear()
        request.session['user'] = {
            'sub': claims['sub'], 'name': claims.get('name', 'Team member'),
            'email': claims.get('email', ''), 'expires_at': int(time.time()) + 3600,
        }
        return RedirectResponse(config.origin + '/', status_code=303, headers={'Cache-Control': 'no-store'})

    @router.post('/auth/logout')
    async def logout(request: Request):
        request.session.clear()
        target = config.origin + '/'
        if config.auth_ready:
            target = f'https://{config.domain}/v2/logout?' + urlencode({'client_id': config.client_id, 'returnTo': target})
        return RedirectResponse(target, status_code=303, headers={'Cache-Control': 'no-store'})


    from app.descriptions import register
    register(router, config, ask_gemini, current_user, problem, reserve)
    app.include_router(router)
    app.add_middleware(AccessGate, config=config)
    # Session must be outside AccessGate. Tokens and client secrets never enter the cookie.
    app.add_middleware(SessionMiddleware, secret_key=config.session_secret or os.urandom(32).hex(),
                       session_cookie='landmarkr_session', max_age=3600, same_site='lax',
                       https_only=config.origin.startswith('https://'))
