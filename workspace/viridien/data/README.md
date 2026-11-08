# Data: source, label quality, and allowed use

Use the manifest that belongs to each dataset. A legend name does not prove that
the pattern occurs in the column. AI agreement is not human accuracy.

| Data | Location | Use | Main limit |
| --- | --- | --- | --- |
| Original scans and region boxes | Organiser archive; set `LITHOLOGY_DATASET` | Input to original-log tools | No complete answer set |
| Organiser answer | `../bakeoff/gt/gold/` | Score the example window | One black-and-white window |
| Accepted real crops | `real/manifest.json` and real image bundle | Follow split and quality flags | Some boxes are examples, not complete beds |
| Training crops | `real/train.jsonl` | Train a matcher | Entire well groups stay together |
| Test crops | `real/test.jsonl` | Evaluate after model selection | See per-record `evaluation_status` |
| Quality holds | `real/needs-review.jsonl` | Inspect and correct | Excluded from usable train/test data |
| Generator sources | `real/generator-sources.json` | Supply eligible training pixels | No test pixels or quality holds |
| Human interval labels | `../human_labels/` | Evaluate complete intervals | Partial logs and some mixed-width beds |
| AI labels | `../bakeoff/gt/silver/`, labeller consensus | Development checks | Model errors and agreement bias |
| Machine suggestions | Lab seed and review queue | Prepare human review | Suggestions are not accepted labels |
| Generated patterns | Generated-pattern bundle | Review and regression work | Experimental; later critiques found defects |
| Historical synthetic v5 | `viridien/archives/synthetic-v5/` after download | Reproduce historical work | Not approved for headline accuracy |
| Outside shared development | `../external_shared/` | Development and smoke tests | Different layouts from most original logs |
| Published outside check | `../external_shared_verify/` | Separate reported check | Previously used in our development |
| Sealed outside verification | Private; not published | Controlled final verification | Never recover it from other folders |
| New outside sources | `../reports/external/` | Inspect and curate | Source leads are not validated test data |

See [four native examples](real/examples/README.md) before loading the full bundle.

## Read a real crop record

[catalog.json](catalog.json) fixes the snapshot ID, counts, bundle sizes, and
checksums. The manifest keeps the source document and well group, original box,
label IDs and printed names, annotation version, pixel hash, split, quality flags,
and provenance. `needs-review.jsonl` retains held rows from either split.

Boxes use original image pixels in `[x0,y0,x1,y1]` order. Right and bottom edges
are excluded. A multi-label crop does not define the position of each component.
Do not use a short pattern-example box as a bed-thickness measurement.

Native PNGs are authoritative. The 224 × 224 RGB copies keep their aspect ratio
with padding. Masks are 255 for content and 0 for padding. Divide RGB by 255 for
floating point inputs. Ignore mask 0 pixels. Do not stretch crops to a square.

## Fixed split and duplicate rules

`real/split-policy.json` fixes the seed and document-to-well mapping. Known pages,
scales, and sidetracks from one well stay in one split. A group goes to test when
the first eight hex digits of `SHA256(seed + ':' + source_group)`, modulo five,
equal zero. Keep this rule when labels arrive.

An exact training crop that matches a test crop is held as `duplicate_of_test`.
Other holds include a legend-fill mismatch and source flags. Accepted work stays
saved when it is withheld. The workspace checker rejects held rows marked usable
for generation. These checks do not establish label correctness.

Earlier tools can have used a group before this split existed. Such test records
are `retrospective`. Only reserved, unseen groups can support a prospective claim.
The crop split is separate from outside-data well splits. Do not shuffle crops
across these divisions.

## Load one training crop

From the repository root, after `download real`:

```python
import json
from pathlib import Path
import numpy as np
from PIL import Image

root = Path('viridien/data/real')
rows = [json.loads(s) for s in (root / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
row = rows[0]
rgb = np.asarray(Image.open(root / row['normalized_png']).convert('RGB')) / 255.0
valid = np.asarray(Image.open(root / row['valid_mask_png'])) > 0
print(row['sample_id'], rgb.shape, valid.sum(), row['targets'])
```

## Source credit

Source URLs and credits remain in each outside subset's manifest. Challenge
images and derived crops remain subject to the organiser's terms. This snapshot
does not grant a new licence for source images. Check rights before public reuse.
Private reviewer sessions, raw agent logs, and sealed data are excluded.

## Outside pool records

`external/splits.json` and `external/manifest.csv` record the private outside
pool split and source URLs. Only `external_shared/` and
`external_shared_verify/` include published images and answer files. The private
`ext_dev`, `ext_test`, and sealed verification pools are not release bundles.
Their suite names remain available for the original controlled environment.
