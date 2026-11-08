import json
from pathlib import Path
from threading import RLock
import pandas as pd

ROOT = Path(__file__).parent / 'storage'
LOCK = RLock()


def read_wells() -> list[dict]:
    """Read the registry under the process lock."""
    with LOCK:
        return json.loads((ROOT / 'wells.json').read_text()) if (ROOT / 'wells.json').exists() else []


def write_wells(wells: list[dict]) -> None:
    """Atomically replace the registry under the process lock."""
    with LOCK:
        ROOT.mkdir(parents=True, exist_ok=True)
        temp = ROOT / 'wells.tmp'
        temp.write_text(json.dumps(wells))
        temp.replace(ROOT / 'wells.json')


def save_well(well: dict) -> None:
    """Insert or update one registry record without losing concurrent changes."""
    with LOCK:
        wells = read_wells()
        wells = [w for w in wells if w['id'] != well['id']]
        write_wells(wells + [well])


def write_csv(well_id: str, df: pd.DataFrame) -> None:
    """Persist the pipeline output for one well."""
    (ROOT / 'csv').mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / 'csv' / f'{well_id}.csv', index=False)


def read_csv(well_id: str) -> pd.DataFrame:
    """Load a cached pipeline output."""
    return pd.read_csv(ROOT / 'csv' / f'{well_id}.csv')
