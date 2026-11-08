"""Check saved state records and make a small image sample. Use Python -I."""
import csv
import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'external_untrusted' / 'alabama'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):
            h.update(block)
    return h.hexdigest()


def read_json(name):
    return json.loads((BASE / 'source' / name).read_text(encoding='utf-8-sig'))


def main():
    audit = [json.loads(line) for line in (BASE / 'access.jsonl').read_text().splitlines()]
    sources = {Path(row['path']).name: row for row in audit if 'sha256' in row}
    for row in sources.values():
        assert digest(ROOT / row['path']) == row['sha256'], row['path']

    records = read_json('county_logs_mobile.json')
    mud = [row for row in records if row['LogType'] == 'MUD']
    wells = read_json('wells_mobile.json')
    by_permit = {}
    for row in wells:
        by_permit.setdefault(row['Permit'], []).append(row)
    catalog = []
    for row in mud:
        matches = by_permit.get(row['Permit'], [])
        out = dict(row)
        for key in ['WellName', 'API', 'FieldName', 'PoolName', 'WellKey']:
            out[key] = ' | '.join(sorted({str(m[key]) for m in matches if m.get(key) is not None}))
        out.update(source_url=sources['county_logs_mobile.json']['url'],
                   source_method='POST', scan_access='Order and payment required. No public scan link in the log table.',
                   downloaded_scan=False, truth_available=False)
        catalog.append(out)
    with (BASE / 'mud_catalog_mobile.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(catalog[0]))
        writer.writeheader()
        writer.writerows(catalog)

    pdf_checks = []
    text_dir = BASE / 'source' / 'pdf_text'
    text_dir.mkdir(exist_ok=True)
    for path in sorted((BASE / 'documents').glob('*.pdf')):
        assert path.read_bytes()[:5] == b'%PDF-'
        with pymupdf.open(path) as doc:
            assert not doc.needs_pass
            pages = []
            all_text = []
            for index, page in enumerate(doc):
                text = page.get_text()
                all_text.append(text)
                if any(word in text.lower() for word in ['mud log', 'mudlog', 'litholog']):
                    pages.append(index + 1)
            text = '\n\f\n'.join(all_text)
            (text_dir / (path.stem + '.txt')).write_text(text, encoding='utf-8')
            pdf_checks.append(dict(file=path.relative_to(ROOT).as_posix(),
                                   bytes=path.stat().st_size, sha256=digest(path),
                                   page_count=len(doc), encrypted=bool(doc.is_encrypted),
                                   keyword_pages_1_based=pages,
                                   text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                   source_url=sources[path.name]['url']))

    samples = []
    images = BASE / 'images'
    images.mkdir(exist_ok=True)
    # These page numbers come from the public exhibit captions and visual checks.
    # All six occurrences show two sections from the same well.
    for doc_no, pages in [(3, [46, 47]), (4, [46, 47]), (7, [13, 14])]:
        pdf = BASE / 'documents' / f'exhibit_{doc_no:02d}.pdf'
        with pymupdf.open(pdf) as doc:
            for page_no, section in zip(pages, ['upper', 'lower']):
                page = doc[page_no - 1]
                assert 'Mud Log' in page.get_text(), (pdf, page_no)
                preview = BASE / 'previews' / f'exhibit_{doc_no:02d}_page_{page_no:03d}.png'
                page.get_pixmap(dpi=100).save(preview)
                pix = page.get_pixmap(dpi=200)
                pixels_sha256 = hashlib.sha256(pix.samples).hexdigest()
                image_path = None
                # Keep one primary image per section plus one later scan per section.
                if doc_no != 4:
                    image_path = images / f'al_sogb_10127_{section}_{2015 if doc_no == 3 else 2023}.png'
                    pix.save(image_path)
                samples.append(dict(
                    well='GW-Edge Gilmore 29-9 #1', permit='10127', well_key='7993',
                    county='Conecuh', field='Southwest Range',
                    pool='Upper Smackover' if section == 'upper' else 'Lower Smackover',
                    section=section, section_group=f'al_sogb_10127_{section}',
                    well_group='al_sogb_10127',
                    status='Primary sample' if doc_no == 3 else 'Repeated source section',
                    source_pdf=pdf.relative_to(ROOT).as_posix(),
                    source_pdf_sha256=digest(pdf), source_url=sources[pdf.name]['url'],
                    source_page_1_based=page_no, source_page_index=page_no - 1,
                    source_page_width_pt=page.rect.width, source_page_height_pt=page.rect.height,
                    image=image_path.relative_to(ROOT).as_posix() if image_path else None,
                    image_sha256=digest(image_path) if image_path else None,
                    rendered_pixels_sha256=pixels_sha256,
                    width=pix.width, height=pix.height, dpi=200,
                    preview=preview.relative_to(ROOT).as_posix(),
                    visible_depth_marks_ft=[13400, 13500] if section == 'upper' else [13600, 13700],
                    depth_status='Visible marks only. Page limits have not been calibrated.',
                    legend_visible=False, exact_truth=False, bbox_available=False,
                    use='Untrusted source review only. No training or evaluation was run.',
                    access='Public hearing exhibit. No sign-in or payment was required.',
                    reuse_license='No open reuse license was found on the checked pages.',
                    gulf_family_fit='Moderate: Gulf Coastal Plain Smackover carbonate mud log. '
                                    'Pattern and gas tracks are present. The legend and full log are absent.'))
    (BASE / 'sample_manifest.json').write_text(json.dumps(samples, indent=2) + '\n')
    (BASE / 'pdf_checks.json').write_text(json.dumps(pdf_checks, indent=2) + '\n')
    summary = dict(
        county_log_records=len(records), mud_records=len(mud),
        mud_permits=len({row['Permit'] for row in mud}),
        saved_pdf_count=len(pdf_checks), saved_pdf_bytes=sum(row['bytes'] for row in pdf_checks),
        sample_wells=1, sample_sections=2, saved_sample_images=4,
        source_page_occurrences=len(samples), sample_dpi=200,
        download_failures=[row for row in audit if 'error' in row],
        audit_hashes_checked=len(sources), pdf_parsing='Passed',
        model_used=False, inherited_weights_used=False, benchmark_run=False,
        exact_truth_created=False)
    (BASE / 'checks.json').write_text(json.dumps(summary, indent=2) + '\n')
    hashes = []
    for path in sorted(BASE.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS':
            hashes.append(f'{digest(path)}  {path.relative_to(BASE).as_posix()}')
    (BASE / 'SHA256SUMS').write_text('\n'.join(hashes) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
