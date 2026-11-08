"""Output locations for the generator lab. Everything lives under data/genlab."""
from pathlib import Path

HOME = Path(__file__).resolve().parent
LAB = HOME.parent
DATA = LAB / 'data'
OUT = DATA / 'genlab'
CROPS = OUT / 'crops'
GEN = OUT / 'generated'
DATASET = OUT / 'dataset'
RATINGS = OUT / 'ratings.jsonl'
STATE = OUT / 'state.json'
CANDIDATES = OUT / 'candidates'
VERDICTS = OUT / 'verdicts'
PROPOSALS = OUT / 'proposals'
FLEET = OUT / 'fleet'
PREFILL = LAB.parent / 'bakeoff' / 'labeltool' / 'data' / 'prefill.jsonl'
QUEUE = DATA / 'pattern-queue.json'
SEEDS = DATA / 'seeds'
