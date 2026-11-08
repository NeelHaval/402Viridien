"""Descriptions for the current polygon. All claims come from saved well results."""
import asyncio
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import httpx

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.descriptions import FactError, PROMPT_VERSION, WARNING, build_facts, saved_csv_path, cached, validate_selection, request_selection, complete_note
from app.services.geo import point_in_polygon
from app.store import wells as repo

MAX_AREA_WELLS = 32
MAX_AREA_BYTES = 8 * 1024 * 1024


class AreaRequest(BaseModel):
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=256)

    @field_validator('polygon')
    @classmethod
    def valid(cls, points):
        if any(not (math.isfinite(x) and math.isfinite(y) and -180 <= x <= 180 and -90 <= y <= 90) for x, y in points):
            raise ValueError('Polygon coordinates must be finite longitude and latitude values')
        if len(set(points)) < 3:
            raise ValueError('Polygon needs three distinct points')
        return points


def canonical_polygon(points):
    points = [(0.0 if x == 0 else x, 0.0 if y == 0 else y) for x, y in points]
    if points[0] == points[-1]:
        points.pop()
    variants = []
    for order in (points, list(reversed(points))):
        first = min(range(len(order)), key=lambda i: order[i:]+order[:i])
        variants.append(order[first:] + order[:first])
    canonical = min(variants)
    return hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest()


def area_facts(db, csv_dir, polygon):
    with db.connect() as conn:
        candidates = conn.execute("SELECT id,lat,lon FROM wells WHERE archived_at IS NULL AND status IN ('ready','partial') AND lat IS NOT NULL AND lon IS NOT NULL ORDER BY id LIMIT 1001").fetchall()
    if len(candidates) > 1000:
        raise FactError('area_too_large', 'The saved-well count exceeds the area note limit.')
    ids = [row['id'] for row in candidates if point_in_polygon(row['lon'], row['lat'], polygon)]
    if len(ids) > MAX_AREA_WELLS:
        raise FactError('area_too_large', 'Select an area with at most 32 completed wells.')
    members, sizes = [], 0
    thickness = defaultdict(float)
    pixel_thickness = defaultdict(float)
    total, gaps, unknown_depth, warned = 0.0, 0.0, 0, 0
    ranges = []
    for well_id in ids:
        row = repo.get(db, well_id)
        # build_facts checks the path before opening it. The cumulative bound prevents large area packets.
        facts, result_hash = build_facts(row, csv_dir)
        sizes += saved_csv_path(row, csv_dir).stat().st_size
        if sizes > MAX_AREA_BYTES:
            raise FactError('area_too_large', 'Saved results exceed the area note size limit.')
        members.append({'result_id': well_id, 'result_hash': result_hash, 'csv_sha256': facts['csv_sha256'],
                        'source_sha256': facts['source_sha256'], 'location_provenance': facts['location_provenance'], 'interval_count': facts['interval_count'],
                        'depth_range': facts['depth_range'], 'lithology_shares': facts['lithology_shares'],
                        'warnings': facts['warnings'], 'validation_flags': facts['validation_flags']})
        if facts['depth_range'] is None:
            unknown_depth += 1
        else:
            ranges.append(facts['depth_range'])
        for label,length in facts.get('pixel_thickness', {}).items():
            pixel_thickness[label] += length
        warned += bool(facts['warnings'] or facts['validation_flags'])
        gaps += facts['missing_coverage_ft'] or 0
        total += facts.get('summed_interval_thickness_ft', 0)
        for label, length in facts.get('lithology_thickness_ft', {}).items():
            thickness[label] += length
        thickness['Other saved labels'] += facts.get('unlisted_lithology_thickness_ft', 0)
    n = len(members)
    highlights = {'count': f'The selected area contains {n} completed wells with saved results.'}
    reviews = {'scan_check': 'Check the saved labels and interval boundaries against the source scans.'}
    facts = {'polygon_hash': canonical_polygon(polygon), 'members': members, 'well_count': n,
             'missing_coverage_ft': round(gaps, 3), 'wells_with_depth_unavailable': unknown_depth,
             'wells_with_warnings': warned, 'depth_range_ft': None, 'lithology_shares': []}
    if ranges:
        start, end = min(r[0] for r in ranges), max(r[1] for r in ranges)
        facts['depth_range_ft'] = [start, end]
        highlights['range'] = f'Saved intervals in this area span {start:g} to {end:g} ft across {len(ranges)} wells.'
    if total:
        ordered = sorted(thickness.items(), key=lambda pair: (-pair[1], pair[0]))
        facts['lithology_shares'] = [{'lithology': label, 'percent': round(100 * length / total, 1)}
                                    for label, length in ordered[:6] if length]
        label, length = ordered[0]
        if label.lower() not in ('unknown', 'blank', 'other saved labels'):
            highlights['dominant'] = f'{label} accounts for {100 * length / total:.1f}% of summed saved interval thickness in the selected area.'
    if pixel_thickness:
        pixel_total = sum(pixel_thickness.values())
        facts['pooled_pixel_shares'] = [{'lithology': label, 'percent': round(100*v/pixel_total,1)} for label,v in pixel_thickness.items()]
        label,v = max(pixel_thickness.items(), key=lambda item:item[1])
        highlights['pixels'] = f'{label} accounts for {100*v/pixel_total:.1f}% of pooled saved pixel thickness in wells without depths; this is not a depth share.'
    if unknown_depth:
        reviews['depth_check'] = f'Check depth calibration for the {unknown_depth} wells with unavailable saved depths.'
    if warned or gaps:
        reviews['coverage'] = f'Check {gaps:g} ft of saved coverage gaps and extraction warnings on {warned} wells.'
    if any(m['location_provenance'] == 'unavailable' for m in members):
        reviews['location'] = 'Check saved well locations because at least one location source is unavailable.'
    facts['choices'] = {'highlights': highlights, 'reviews': reviews}
    return facts


def register_area(router, config, generate, current_user, problem, reserve):
    slots = asyncio.Semaphore(1)

    @router.post('/api/report/description')
    async def description(area: AreaRequest, request: Request):
        user = current_user(request.session)
        if config.required and not user:
            return problem(401, 'login_required', 'Sign in to generate an area note.')
        if not config.gemini_key:
            return problem(503, 'gemini_not_configured', 'Gemini notes are disabled until setup is complete.')
        try:
            facts = area_facts(request.app.state.db, request.app.state.settings.csv_dir, area.polygon)
            if not facts['members']:
                return JSONResponse({'note': 'No completed wells with saved results are inside this area. Select another area to generate a note.',
                                     'warning': WARNING, 'empty': True, 'cached': False, 'model': None,
                                     'result_hash': facts['polygon_hash']}, headers={'Cache-Control': 'no-store'})
            key = hashlib.sha256(json.dumps({'polygon': facts['polygon_hash'],
                'results': sorted(m['result_hash'] for m in facts['members']),
                'model': config.model, 'prompt_version': PROMPT_VERSION}, sort_keys=True).encode()).hexdigest()
            result = cached(request.app.state.db, key)
            if result:
                return JSONResponse({**result, 'cached': True}, headers={'Cache-Control': 'no-store'})
            if not reserve(user['sub'] if user else 'public-summary'):
                return problem(429, 'rate_limit', 'Wait one minute before the next Gemini request.')
            prompt = ('Select one highlight_id and one review_id from choices in this saved area JSON. '
                      'Return only those two JSON keys and allowed IDs; never write prose.\n' + json.dumps(facts))
            if len(prompt.encode()) > 24000:
                raise FactError('area_too_large', 'The area facts exceed the note size limit.')
            async def bounded_generate(conf, question, *, system=None):
                async with slots:
                    return await generate(conf, question, system=system)
            note, fallback, used_model = await complete_note(bounded_generate, config, prompt, facts)
            result = {'note': note, 'fallback': fallback, 'error': 'Gemini is unavailable. This is a saved-data summary. Try again.' if fallback else None, 'warning': WARNING, 'model': used_model, 'prompt_version': PROMPT_VERSION,
                      'result_hash': key, 'polygon_hash': facts['polygon_hash'], 'well_count': facts['well_count'],
                      'result_hashes': sorted(m['result_hash'] for m in facts['members'])}
            if not fallback and used_model == config.model:
                cached(request.app.state.db, key, result)
            return JSONResponse({**result, 'cached': False}, headers={'Cache-Control': 'no-store'})
        except FactError as error:
            return problem(error.status, error.code, error.message)
        except (asyncio.TimeoutError, OSError, ValueError, httpx.HTTPError, sqlite3.Error):
            return problem(502, 'description_unavailable', 'The saved area note is unavailable. Try again.')
