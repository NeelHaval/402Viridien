from collections import defaultdict
import math
import pandas as pd

COLORS = {'Shale grey': '#99a5bb', 'Halite': '#d2b9db', 'Sandstone': '#e1c58d',
          'Siltstone': '#b9cb9b', 'Marl': '#a3c9c4', 'Limestone': '#c9d9e5',
          'Anhydrite': '#c2b3a0', 'Coal': '#737887'}


def build_column(dfs: list[pd.DataFrame]) -> list[dict]:
    """Pool overlapping feet in absolute 250-foot bins and merge dominant neighbours."""
    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        return []
    start = min(float(df.top_depth.min()) for df in dfs)
    end = max(float(df.base_depth.max()) for df in dfs)
    edges = [start] + list(range((math.floor(start / 250) + 1) * 250, math.ceil(end / 250) * 250, 250)) + [end]
    merged = []
    for top, base in zip(edges, edges[1:]):
        feet, contributors = defaultdict(float), set()
        for i, df in enumerate(dfs):
            for bed in df.itertuples():
                overlap = max(0, min(base, bed.base_depth) - max(top, bed.top_depth))
                if overlap:
                    feet[bed.lithology] += overlap
                    contributors.add(i)
        if not feet:
            continue
        dominant = max(feet, key=feet.get)
        if merged and merged[-1]['lithology'] == dominant and merged[-1]['base_ft'] == top:
            segment = merged[-1]
            segment['base_ft'] = base
            segment['contributors'].update(contributors)
            for name, count in feet.items():
                segment['feet'][name] += count
        else:
            merged.append(dict(top_ft=top, base_ft=base, lithology=dominant, feet=feet, contributors=contributors))
    result = []
    for segment in merged:
        feet = segment.pop('feet')
        ordered = sorted(feet, key=feet.get, reverse=True)
        exact = [100 * feet[name] / sum(feet.values()) for name in ordered]
        shares = [math.floor(value) for value in exact]
        for i in sorted(range(len(shares)), key=lambda i: exact[i] - shares[i], reverse=True)[:100 - sum(shares)]:
            shares[i] += 1
        result.append({**segment, 'n_wells': len(segment['contributors']), 'color': COLORS[segment['lithology']],
                       'pct': shares[0], 'breakdown': [{'lithology': name, 'pct': pct} for name, pct in zip(ordered, shares)]})
        result[-1].pop('contributors')
    return result
