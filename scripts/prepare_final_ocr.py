"""Download and check the English recognition weights before an offline run."""
import argparse
import hashlib
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Check existing weights without a download')
    args = parser.parse_args()
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, detector=False, verbose=False, download_enabled=not args.offline)
    path = Path(reader.model_storage_directory) / 'english_g2.pth'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    print('OCR weights checked: ' + str(path))
    print('SHA256 ' + digest.hexdigest())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
