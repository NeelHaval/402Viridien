"""Audit regenerated reports, queue locations, native donor tiles and old seams."""
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image

from genlab import candidates, learn, paths
from genlab.swatch import edge_continuity, remove_frame
from generator._texture import estimate_period, trim_frame


def main():
    reports = list(candidates.reports())
    by_id = {r['doc_id']: r for r in reports}
    expected_queue = []
    counts, borrowed = Counter(), Counter()
    old_swatch_count = defects = fixed = fixed_both = frame_fixed = 0
    frame_removed = frame_removed_safe = 0
    doubts = []
    for report in reports:
        assert report['version'] == candidates.VERSION, report['doc_id']
        for e in report['entries']:
            counts[e['tier']] += 1
            assert (paths.CANDIDATES / e['sheet_png']).exists()
            with Image.open(paths.CANDIDATES / e['sheet_png']) as sheet:
                assert e['row_y'] + e['row_height'] <= sheet.height
            if e['tier'] in ('verified', 'borrowed', 'swatch'):
                expected_queue.append(dict(doc_id=report['doc_id'], legend_id=e['legend_id'],
                                           label=e['label'], tier=e['tier'],
                                           sheet_png=e['sheet_png'], row_index=e['row_index']))
            crop_path = paths.CROPS / report['doc_id'] / ('_'.join(map(str, e['swatch_box'])) + '.png')
            with Image.open(crop_path) as im:
                raw = np.asarray(im.convert('RGB'))
            cleaned, texture, repeat = candidates.swatch(raw)
            assert repeat['seam_risk'] == e['seam_risk']
            if e['tier'] == 'swatch':
                assert not e['seam_risk'] and e['seam_x']['ok'] and e['seam_y']['ok']
            if e['tier'] == 'ambiguous':
                assert e['twins']
            if e['tier'] == 'borrowed':
                donor = e['donor']
                dr = by_id[donor['doc_id']]
                de = next(x for x in dr['entries'] if x['legend_id'] == donor['legend_id'])
                assert dr['training_allowed'] and de['tier'] in ('human', 'verified')
                assert dr['doc_id'] != report['doc_id'] and e['matches'] == 0 and not e['twins']
                assert candidates.name_key(e['label']) == candidates.name_key(de['label'])
                source = candidates.load_data(paths.CANDIDATES / donor['data_file'])
                target = candidates.load_data(paths.CANDIDATES / e['data_file'])
                assert np.array_equal(source['tile'], target['tile'])
                match = candidates.pixel_match(cleaned, source, minimum=.92, distance=20)
                assert match >= .92 and abs(match - donor['match']) <= .00001
                borrowed[(e['label'], donor['doc_id'], donor['label'])] += 1
            if e.get('previous_tier') != 'swatch-only':
                continue
            old_swatch_count += 1
            legacy, _ = trim_frame(raw)
            legacy, _ = learn.clean(legacy)
            legacy, _ = learn.clean(legacy.transpose(1, 0, 2))
            legacy = legacy.transpose(1, 0, 2)
            xp = estimate_period(legacy, 1)
            old_px = xp['pixels'] if xp['pixels'] and xp['observed_cycles'] >= 2 else legacy.shape[1]
            _, residual = remove_frame(legacy)
            # Even a measured x period can retain a frame: voting a frame
            # against fill makes a grey vertical rail rather than removing it.
            frame_defect = bool(residual[0] or residual[2])
            edge_defect = old_px == legacy.shape[1] and not edge_continuity(legacy, 1)['ok']
            defect = frame_defect or edge_defect
            removed = frame_defect and bool(repeat['frame_trim'][0] or repeat['frame_trim'][2])
            repaired = defect and repeat['seam_x']['ok'] and (
                removed or texture['tile'].shape[1] < old_px or not edge_defect)
            frame_removed += removed
            frame_removed_safe += removed and not repeat['seam_risk']
            defects += defect
            fixed += repaired
            fixed_both += repaired and not repeat['seam_risk']
            frame_fixed += frame_defect and repaired
            if defect and not repaired:
                doubts.append(dict(doc_id=report['doc_id'], legend_id=e['legend_id'],
                                   label=e['label'], tier=e['tier'], seam_x=e['seam_x'], seam_y=e['seam_y']))
    queue = json.loads((paths.CANDIDATES / 'review-queue.json').read_text(encoding='utf-8'))
    assert queue == expected_queue
    summary = json.loads((paths.CANDIDATES / 'summary.json').read_text(encoding='utf-8'))
    assert dict(counts) == {k: v for k, v in summary['overall'].items() if v}
    result = dict(documents=len(reports), entries=sum(counts.values()), tiers=dict(counts),
                  review_queue=len(queue), baseline_swatch_only=old_swatch_count,
                  old_x_seam_defects=defects, old_x_seam_fixed=fixed,
                  old_x_seam_fixed_both_axes_safe=fixed_both,
                  old_frame_x_seam_fixed=frame_fixed,
                  frame_artifacts_removed=frame_removed,
                  frame_artifacts_removed_both_axes_safe=frame_removed_safe,
                  top_borrowed=[dict(label=label, donor=did, donor_label=dl, count=n)
                                for (label, did, dl), n in borrowed.most_common(10)],
                  unresolved_old_seams=doubts)
    out = paths.OUT / 'scratch' / 'candidate-audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print(json.dumps(result, indent=1))
    print(f'PASS candidate audit: documents={len(reports)} entries={sum(counts.values())} queue={len(queue)}')


if __name__ == '__main__':
    main()
