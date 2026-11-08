"""Local paths and side-effect-free shared utilities."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
import os
import re
from pathlib import Path

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
ANALYSIS = ROOT.parent / 'analysis'
DATA = HOME / 'data'
CACHE = HOME / 'cache'
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('USE_TF', '0')


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def write_json(path, value):
    path = Path(path).resolve()
    path.relative_to(HOME)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def norm(label):
    # Exact make_mix_page.legend_types normalization; intentionally no OCR fixes.
    return re.sub(r'[^a-z]', '', label.lower()).replace('sandandsandstone', 'sandstone')


def slug(value):
    return re.sub(r'[^a-z0-9]+', '-', value.lower()).strip('-')


def pid_slug(pid):
    return pid.replace('/', '__')


def family(name):
    if name.startswith('0000000009'): return 'D-582 1-bit grid'
    if name.startswith('MC765'): return 'MC765 colour grid'
    if 'mc761' in name.lower(): return 'MC761 mineral/lithology'
    if any(k in name for k in ('Rio_', 'Critter', 'Callaway')): return "Texas 'Rock Types' box"
    if any(k in name for k in ('bp_gc743', 'gb506', 'hadrian', 'gc743_dc141')): return 'Geoservices grid'
    if any(k in name for k in ('Allegheny', 'Trail Mtn', 'FREEMAN', 'Shell_')): return 'One-off layout'
    return 'Gulf 3-panel'


def short(name):
    match = re.search(r'log\d\d', name)
    if name.startswith('0000000009') and match: return 'D582 ' + match[0]
    text = re.sub(r'^\d{10,}[ _-]*', '', name).replace('_', ' ')
    return text[:22] + ' ' + hashlib.sha1(name.encode()).hexdigest()[:4]


def local_file(value):
    path = (HOME / value).resolve()
    path.relative_to(HOME)
    return path


def relative(path):
    return Path(path).resolve().relative_to(HOME).as_posix()


def tile(image, width, height):
    from PIL import Image
    out = Image.new('RGB', (width, height), 'white')
    for y in range(0, height, image.height):
        for x in range(0, width, image.width): out.paste(image, (x, y))
    return out


def font(size):
    from PIL import ImageFont
    for name in ('C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try: return ImageFont.truetype(name, size)
        except OSError: pass
    return ImageFont.load_default(size=size)
