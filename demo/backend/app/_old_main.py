from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path
import shutil
from uuid import uuid4
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from column import build_column
from pipeline import run_pipeline
from scoring import aggregate_scores, score_well
from seed import seed_wells
from storage import ROOT, read_csv, read_wells, save_well, write_csv


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the demo registry before serving requests."""
    seed_wells()
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:5173'], allow_methods=['*'], allow_headers=['*'])
api = APIRouter()


class Area(BaseModel):
    polygon: list[tuple[float, float]] = Field(min_length=3)


def inside(lon: float, lat: float, polygon: list[tuple[float, float]]) -> bool:
    """Test polygon containment with ray casting, including boundary points."""
    hit = False
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        cross = (lon - x1) * (y2 - y1) - (lat - y1) * (x2 - x1)
        if abs(cross) < 1e-10 and min(x1, x2) <= lon <= max(x1, x2) and min(y1, y2) <= lat <= max(y1, y2):
            return True
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            hit = not hit
    return hit


def process_upload(well: dict, path: Path) -> None:
    """Run the replaceable pipeline and persist a terminal status."""
    try:
        result = run_pipeline(str(path))
        if result['lat'] is None or result['lon'] is None:
            raise ValueError('location not found')
        df = result['df']
        write_csv(well['id'], df)
        well.update({key: result[key] for key in ('lat', 'lon')})
        well.update(name=result.get('name') or well['name'], block=result.get('block') or 'Unknown',
                    total_depth_ft=float(df.base_depth.max()),
                    scores=score_well(df, well['lat'], well['lon'], [w for w in read_wells() if w['id'] != well['id']]), status='ready')
    except Exception as exc:
        well.update(status='failed', error=str(exc), scores=None)
    save_well(well)


@api.get('/wells')
def wells() -> list[dict]:
    """Return lightweight cached map records."""
    keys = ('id', 'name', 'lat', 'lon', 'block', 'status', 'scores')
    return [{key: w[key] for key in keys} for w in read_wells()]


@api.get('/wells/{well_id}')
def well_detail(well_id: str) -> dict:
    """Return one well and its pooled column if ready."""
    well = next((w for w in read_wells() if w['id'] == well_id), None)
    if well is None:
        raise HTTPException(404, 'Unknown well')
    return {**{k: v for k, v in well.items() if k != 'error'},
            'scores': well['scores'] if well['status'] == 'ready' else None,
            'column': build_column([read_csv(well_id)]) if well['status'] == 'ready' else []}


@api.post('/upload')
def upload(file: UploadFile, background_tasks: BackgroundTasks) -> dict:
    """Save an upload and schedule processing after returning its identifier."""
    well_id = uuid4().hex
    folder = ROOT / 'uploads'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{well_id}{Path(file.filename or "upload").suffix}'
    with path.open('wb') as target:
        shutil.copyfileobj(file.file, target)
    # Provisional map position until the pipeline returns the actual location.
    well = dict(id=well_id, name=Path(file.filename or 'Upload').stem, lat=26.5, lon=-90.,
                block='Unknown', status='processing', total_depth_ft=0, scores=None, error=None)
    save_well(well)
    background_tasks.add_task(process_upload, well, path)
    return {'id': well_id, 'status': 'processing'}


@api.post('/report')
def report(area: Area) -> dict:
    """Return cached area scores and a pooled column for contained ready wells."""
    selected = [w for w in read_wells() if w['status'] == 'ready' and inside(w['lon'], w['lat'], area.polygon)]
    return {'wells': [{k: w[k] for k in ('id', 'name', 'lat', 'lon', 'scores')} for w in selected],
            'scores': aggregate_scores(selected), 'column': build_column([read_csv(w['id']) for w in selected])}


app.include_router(api, prefix='/api')
app.include_router(api, include_in_schema=False)
app.mount('/', StaticFiles(directory=Path(__file__).resolve().parents[1] / 'frontend' / 'dist',
                          html=True, check_dir=False), name='frontend')
