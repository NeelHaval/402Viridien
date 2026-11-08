"""Read-only tolerance for Windows file replacement by the active labeller."""
import json
import time


class ReadOnlyStore:
    def __init__(self, store):
        self.store = store

    def __getattr__(self, key):
        return getattr(self.store, key)

    def get(self, ident):
        for attempt in range(10):
            try:
                return self.store.get(ident)
            except (PermissionError, FileNotFoundError, json.JSONDecodeError):
                if attempt == 9:
                    raise
                time.sleep(.05 * (attempt + 1))
