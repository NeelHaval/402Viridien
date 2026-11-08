"""Reversible process/network prohibition and capped optional response replay."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from .statistics import read_json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode()).hexdigest()


class Replay:
    """Optional paid methods only replay explicitly supplied offline responses.

    Each cache file is <method>_<request SHA256>.json, containing the provider's
    decoded response. No credentials are read and no external calls are made.
    The cap counts all replay lookups, including misses, across both conditions.
    """
    def __init__(self, directory, cap=20):
        self.directory = Path(directory)
        self.cap = max(0, cap)
        self.events = []

    def get(self, method, signature):
        key = digest(signature)
        if len(self.events) >= self.cap:
            return None
        response = read_json(self.directory / f'{method}_{key}.json')
        self.events.append(dict(method=method, key=key, hit=response is not None, external_calls=0))
        return response


@contextmanager
def offline_guard():
    import os
    import socket
    import subprocess
    def blocked(*args, **kwargs):
        raise RuntimeError('Synthetic benchmark is offline; network and child processes are disabled')
    with ExitStack() as stack:
        for obj, name in [(socket, 'create_connection'), (socket, 'getaddrinfo'),
                          (socket.socket, 'connect'), (socket.socket, 'connect_ex'),
                          (subprocess, 'Popen'), (os, 'system'), (os, 'startfile')]:
            if hasattr(obj, name):
                stack.enter_context(patch.object(obj, name, blocked))
        stack.enter_context(patch.dict(os.environ, dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_TELEMETRY='1', USE_TF='0', PYTHONDONTWRITEBYTECODE='1')))
        yield


@contextmanager
def paid_replay(replay):
    """Cover existing paid strategy clients without opening credentials."""
    from strategies import _jev, s09_vlm_terra
    def init(client, timeout=20, retries=0):
        client.timeout, client.retries, client.calls, client.key = timeout, 0, 0, None
    def ask(client, state, assets, questions, deadline=None):
        return replay.get('jev', dict(state=state, assets=assets, questions=questions))
    def terra(png_path, prompt, cache, deadline):
        return replay.get('terra', dict(image_sha256=hashlib.sha256(Path(png_path).read_bytes()).hexdigest(),
                                       prompt=prompt)) or {}
    with patch.object(_jev.Client, '__init__', init), patch.object(_jev.Client, 'ask', ask), \
         patch.object(s09_vlm_terra, '_invoke', terra):
        yield
