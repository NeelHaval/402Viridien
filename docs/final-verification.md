# Final command validation

The implementation was tested on the team VM from a fresh public clone and a
fresh Python 3.12 environment. See `final-validation-receipt.json` for the tested
commit, file comparisons, input hashes and measured page times.

```sh
git clone --depth 1 https://github.com/NeelHaval/402Viridien clean-repo
python3 -m venv clean-venv
clean-venv/bin/pip install -r clean-repo/requirements-final.txt pytest
cd clean-repo
../clean-venv/bin/python -m pytest tests/test_final.py -q

PACKET=/srv/viridien/submission/20261108/incoming/scoreboard/depth_completion_v1/packet
../clean-venv/bin/python -m pipeline.final \
  --images "$PACKET/inputs" --boxes "$PACKET/inputs" \
  --out /srv/viridien/verify/clean-released-final --workers 2 --allow-assumed-units
../clean-venv/bin/python scripts/compare_final.py \
  --actual /srv/viridien/verify/clean-released-final \
  --official outputs/official --report /srv/viridien/verify/clean-diff-final.json

../clean-venv/bin/python -m pipeline.final \
  --images /srv/viridien/verify/challenge-inputs/images \
  --boxes /srv/viridien/verify/challenge-inputs/boxes \
  --out /srv/viridien/verify/clean-challenge --workers 2
```

The three released pages contain 72, 133 and 358 intervals. Both CSV layouts,
at both official file locations, match in all fields. This is 563 intervals and
12 CSV files. The comparison checks headers, row counts, pixel coordinates,
labels, cluster IDs and depth values.

The first run with `requirements-s19.txt` gave 156 Woody depth anchors instead
of the saved fit's 157. This changed 15 top depths and 16 bottom depths by 0.1.
The final dependency file pins the OpenCV, NumPy, SciPy, scikit-learn and ONNX
Runtime versions used for the official depth fit. These pins restored all 157
anchors and removed all depth differences. An older compatible tifffile version
is also pinned because the current version requires NumPy 2.

The official comparison uses `--allow-assumed-units` because the released Nichols
depths have assumed feet units. The default command was tested separately on
that page: all 72 intervals remain, both depth fields are blank, and the sidecar
states `Printed depth unit is not verified`. New verification data uses this
default unit check.

The three additional pages are the GB506 MD log, Rio San Gabriel log and
GC743/DC802 log. The last image has 187,000,000 pixels, above Pillow's normal
error limit. All three use automatic OCR and S19 inference. No saved predictions
or saved calibration fits are used.

The generated-page test runs the actual command. It checks PNG input, output
headers, full column image height, blank unavailable depths, root CSV copies,
manifest hashes, a missing annotation and forced page timeouts. A failed page
must not stop the other page.

The deployed wrapper is `/srv/viridien/verify/run_final.sh`. Its source is
`scripts/run_final.sh`. The installed working copy is `/srv/viridien/verify/repo`
and its environment is `/srv/viridien/verify/venv`. Validation logs and output
bundles remain under `/srv/viridien/verify/`.
