"""Build the metadata-only headless variant of the pinned upstream OCR wheel."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile
import sys

URL = 'https://files.pythonhosted.org/packages/ba/12/1e5497183bdbe782dbb91bad1d0d2297dba4d2831b2652657f7517bfc6df/rapidocr_onnxruntime-1.4.4-py3-none-any.whl'
SHA256 = '971d7d5f223a7a808662229df1ef69893809d8457d834e6373d3854bc1782cbf'
OLD = 'rapidocr_onnxruntime-1.4.4.dist-info/'
NEW = 'rapidocr_onnxruntime-1.4.4+headless.dist-info/'


def main():
    if len(sys.argv) > 1:
        upstream = Path(sys.argv[1]).read_bytes()
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            upstream = b''.join(iter(lambda: response.read(1024*1024), b''))
    assert hashlib.sha256(upstream).hexdigest() == SHA256, 'Upstream wheel hash mismatch'
    target = Path(__file__).resolve().parents[1] / 'vendor/wheels/rapidocr_onnxruntime-1.4.4+headless-py3-none-any.whl'
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(io.BytesIO(upstream)) as source, zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        for entry in source.infolist():
            if entry.filename == OLD + 'RECORD':
                continue
            data = source.read(entry.filename)
            name = entry.filename.replace(OLD, NEW)
            if name == NEW + 'METADATA':
                data = data.replace(b'Version: 1.4.4\n', b'Version: 1.4.4+headless\n')
                assert b'Requires-Dist: opencv-python>=4.5.1.48' in data
                data = data.replace(b'Requires-Dist: opencv-python>=4.5.1.48', b'Requires-Dist: opencv-python-headless==4.11.0.86')
            info = zipfile.ZipInfo(name, entry.date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = entry.external_attr
            output.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')
            records.append([name, 'sha256=' + digest, str(len(data))])
        stream = io.StringIO(newline='')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerows(records + [[NEW+'RECORD', '', '']])
        info = zipfile.ZipInfo(NEW+'RECORD', (2024, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        output.writestr(info, stream.getvalue().encode('utf-8'))
    print(target.name, hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
