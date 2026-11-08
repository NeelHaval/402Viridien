import numpy as np
from pipeline import run_pipeline
from scoring import score_well
from storage import ROOT, write_csv, write_wells


def seed_wells() -> None:
    """Create ten reproducible, geographically varied wells on first startup."""
    if (ROOT / 'wells.json').exists():
        return
    locations = [(28.1, -92.8), (27.5, -91.6), (26.4, -90.9), (28.0, -89.6), (27.0, -88.7),
                 (25.7, -87.4), (26.0, -92.5), (28.4, -87.0), (25.4, -89.9), (27.1, -93.0)]
    wells, frames = [], []
    for i, (lat, lon) in enumerate(locations):
        np.random.seed(100 + i)
        df = run_pipeline(f'well-{i + 1}')['df']
        if i == 0:
            df.loc[df.index % 4 != 0, ['lithology', 'cluster_id']] = ['Halite', 1]
        elif i == 1:
            df['lithology'] = ['Shale grey' if j % 3 == 0 else 'Sandstone' for j in df.index]
            df['cluster_id'] = [0 if j % 3 == 0 else 2 for j in df.index]
        elif i == 2:
            df[['lithology', 'cluster_id']] = ['Shale grey', 0]
        elif i == 3:
            edges = np.linspace(10000, 14500, 91)
            df = df.reindex(range(90)).copy()
            df['top_depth'], df['base_depth'] = edges[:-1], edges[1:]
            df['top_y'], df['bottom_y'] = (edges[:-1] - 10000) * 4, (edges[1:] - 10000) * 4
            df['lithology'] = ['Shale grey' if j % 2 == 0 else 'Siltstone' for j in df.index]
            df['cluster_id'] = [0 if j % 2 == 0 else 3 for j in df.index]
        well = dict(id=f'well-{i + 1:02}', name=f'Well {i + 1:02}', lat=lat, lon=lon,
                    block=f'GOM {100 + i}', status='ready', total_depth_ft=float(df.base_depth.max()), scores=None, error=None)
        wells.append(well)
        frames.append(df)
        write_csv(well['id'], df)
    for well, df in zip(wells, frames):
        well['scores'] = score_well(df, well['lat'], well['lon'], [w for w in wells if w['id'] != well['id']])
    write_wells(wells)
    np.random.seed(None)
