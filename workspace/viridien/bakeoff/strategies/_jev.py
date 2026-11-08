"""Small shared SystemOne image client; credentials are read only at runtime."""
import base64
import io
import os
from pathlib import Path
import time

import requests
from PIL import Image

ENV_PATH = Path('C:/Users/hssil/Downloads/03_Programming_Projects/misc/.env')
_proxy_unavailable = False


def _key():
    key = os.environ.get('JEV_API_KEY')
    if key:
        return key
    try:
        for line in ENV_PATH.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.strip().partition('=')
            if sep and name.strip().removeprefix('export ').strip() == 'JEV_API_KEY':
                return value.strip().strip('"\'')
    except OSError:
        pass
    return None


def image_asset(image):
    image = image.convert('RGB') if isinstance(image, Image.Image) else Image.fromarray(image).convert('RGB')
    # Fit inside the permitted canvas, then pad, preserving image proportions.
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.NEAREST)
    scale = min(1024 / max(image.size), max(1, 512 / min(image.size)))
    if scale > 1:
        image = image.resize(tuple(max(1, round(x * scale)) for x in image.size), Image.Resampling.NEAREST)
    canvas = Image.new('RGB', (max(512, image.width), max(512, image.height)), 'white')
    canvas.paste(image, ((canvas.width-image.width)//2, (canvas.height-image.height)//2))
    buffer = io.BytesIO()
    canvas.save(buffer, format='PNG')
    return {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(buffer.getvalue()).decode('ascii')}


class Client:
    def __init__(self, timeout=20, retries=2):
        self.timeout, self.retries, self.calls = timeout, retries, 0
        self.key = _key()

    def ask(self, state, assets, questions, deadline=None):
        global _proxy_unavailable
        if not self.key or _proxy_unavailable:
            return None
        payload = {'model':'jev-latest', 'state':state, 'assets':assets, 'questions':questions}
        headers = {'Authorization':'Bearer '+self.key, 'Content-Type':'application/json', 'TypeSafe-Danger-Zone-Images':'true'}
        proxy_failures = 0
        for _ in range(self.retries + 1):
            remaining = deadline - time.monotonic() if deadline is not None else self.timeout
            if remaining <= 0:
                break
            try:
                self.calls += 1
                response = requests.post('https://api.typesafe.ai/v1/systemone', headers=headers, json=payload, timeout=min(self.timeout, remaining))
                response.raise_for_status()
                return response.json()
            except requests.exceptions.ProxyError:
                proxy_failures += 1
            except (requests.RequestException, ValueError):
                continue
        if proxy_failures == self.retries + 1:
            _proxy_unavailable = True
        return None
