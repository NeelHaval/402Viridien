"""Environment-driven settings. Every path and limit has a default under demo/backend/."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to the default."""
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    """Paths, limits and tunables for the app and the pipeline."""

    data_dir: Path = field(default_factory=lambda: _env_path("APP_DATA_DIR", BACKEND_DIR / "data"))
    max_upload_bytes: int = field(default_factory=lambda: int(os.environ.get("APP_MAX_UPLOAD_MB", "500")) * 1024 * 1024)
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://localhost:8000")
    frontend_dist: Path = field(default_factory=lambda: _env_path("APP_FRONTEND_DIST", BACKEND_DIR.parent / "frontend" / "dist"))
    # Layer-1 OCR time limits (seconds). They are safety valves that never bind in normal use: results must not depend on machine load.
    layer1_budget_s: float = field(default_factory=lambda: float(os.environ.get("APP_LAYER1_BUDGET_S", "1200")))
    locate_budget_s: float = field(default_factory=lambda: float(os.environ.get("APP_LOCATE_BUDGET_S", "600")))
    probe_budget_s: float = field(default_factory=lambda: float(os.environ.get("APP_PROBE_BUDGET_S", "300")))
    # Pipeline tunables, carried over from the research notebook.
    stripe_height: int = 8
    boundary_window: int = 3
    boundary_min_gap: int = 2
    boundary_ink_k: float = 2.0
    split_window: int = 16
    split_min_run: int = 3

    @property
    def db_path(self) -> Path:
        """SQLite file."""
        return self.data_dir / "app.db"

    @property
    def scans_dir(self) -> Path:
        """Uploaded scans."""
        return self.data_dir / "scans"

    @property
    def csv_dir(self) -> Path:
        """Output and raw CSVs."""
        return self.data_dir / "csv"

    @property
    def reference_dir(self) -> Path:
        """Reference data (bathymetry, transitions, ...)."""
        return self.data_dir / "reference"

    def ensure_dirs(self) -> None:
        """Create the runtime folders."""
        for folder in (self.data_dir, self.scans_dir, self.csv_dir, self.reference_dir):
            folder.mkdir(parents=True, exist_ok=True)
