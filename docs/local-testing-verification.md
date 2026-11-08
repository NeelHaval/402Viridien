# Local workflow verification

Verified on Windows with Python 3.12.7. The isolated environment installed the
core package with `requirements.lock`: NumPy 2.2.6, Pillow 11.3.0 and psutil 7.0.0.
The software check installed pytest 8.4.2. No real-data model inference, model
API calls, training or label promotion ran during this verification.

Commands and results:

```text
python scripts/local_test.py --check
LOCAL_INPUT_CHECK_OK suite=demo cases=1

python scripts/local_test.py --software-tests
356 passed, 10 skipped in 65.65s (0:01:05)

python scripts/local_test.py --suite quick-dev --check --data-dir <verification directory>
DOWNLOAD_OK profile=dev files=165 auto_label=false independent_accuracy=false
LOCAL_INPUT_CHECK_OK suite=quick-dev cases=2

python scripts/local_test.py --suite dev --check --data-dir <verification directory>
DOWNLOAD_OK profile=dev files=165 auto_label=false independent_accuracy=false
LOCAL_INPUT_CHECK_OK suite=dev cases=51
```

The quick and full DEV input checks used the isolated constrained environment.
Private assets were downloaded through GitHub CLI by immutable asset ID. Their
sizes and hashes were checked before extraction. Cached reuse and extracted
source hashes were checked too. A drive with free space held the verification
data; `--data-dir` avoids forcing large downloads onto the system drive.

The 10 skips were explicit optional science-dependency and Windows capability
skips. Hosted CI also has a science-dependency job. CI results remain separate
from these local results. The stress notebook check was updated for current
main's prediction, calibration and submission entry points. It parses the AST
and executes no notebook cells. The former `train_cnn` assertion described a
removed notebook implementation.

Full synthetic scoring, held-out evaluation, DJS v2 inference and independent
accuracy measurement: not run. Optional large profiles use the same checked
download/extraction path; every large archive was not downloaded again for this
verification. Original frozen release assets were not changed.
