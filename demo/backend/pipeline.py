# Replace with real pipeline. Keep the return shape.
from pathlib import Path
import time
import numpy as np
import pandas as pd


def run_pipeline(file_path: str) -> dict:
    """Return dummy beds and well metadata in the production pipeline shape."""
    time.sleep(4)
    names = ['Shale grey', 'Halite', 'Sandstone', 'Siltstone', 'Marl', 'Limestone', 'Anhydrite', 'Coal']
    count = np.random.randint(40, 91)
    start, end = np.random.uniform(8000, 12000), np.random.uniform(14000, 20000)
    edges = np.r_[start, start + np.cumsum(np.random.dirichlet(np.ones(count))) * (end - start)]
    ids = np.random.choice(len(names), count, p=np.random.dirichlet(np.ones(len(names))))
    df = pd.DataFrame({'top_y': (edges[:-1] - start) * 4, 'bottom_y': (edges[1:] - start) * 4,
                       'cluster_id': ids, 'lithology': [names[i] for i in ids],
                       'top_depth': edges[:-1], 'base_depth': edges[1:]})
    return {'df': df, 'lat': float(np.random.uniform(25, 28.8)),
            'lon': float(np.random.uniform(-93.5, -86.5)),
            'name': f'Uploaded {Path(file_path).stem}', 'block': 'Unknown'}
