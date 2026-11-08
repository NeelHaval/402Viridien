The first baseline checkpoint has three QA/gold cases from the frozen 68-case list. It uses saved S19 and DINO predictions on the same cases. The remaining 65 baseline cases are not run in this checkpoint.

S24 uses the inference functions from Benedek commit d39919f1bbcf254b357a99eba1cf0845c391b135. The adapter uses CPU EasyOCR and adds the shared harness depth calibration. The adapter smoke check uses small artificial inputs. It does not establish parity with a full original notebook run.

The shared score.py SHA-256 is 50ee28c914bc0dc9f9f8a255d5d200e5d9e61ff9ef1f9c7187e2bdc4e5c59de6. Both methods use this scorer. The strict policy and the approved Clay, Shale, Clay/Shale and Clay-Shale equivalence are separate views. Raw CSV labels and contacts remain unchanged. The availability-v2 policy blocks empty truth and invalid legend lithology. Unreviewed silver is descriptive evidence. The paired bootstrap seed is 1108.

The new Drive packet has three images and box coordinates. It has no interval truth. It uses the full interpreted column for S19 and S24. It does not resize or shorten a column. Accuracy is unavailable. Runtime and prediction differences can be reported.

Both remote jobs use the real v4 packet directory, the same worker lock, two CPU cores and the existing read-only lith Python environment. The queue wait limit is 60 seconds. The compute limit is 300 seconds. The outer limit is 380 seconds. The new test has a 10 GiB address-space limit. Each completed model case is saved before the next case starts. The refresh command records unprocessed cases and a compute timeout.
