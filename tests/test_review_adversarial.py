"""Independent reviewer regressions: split identity, integrity, and metric oracles."""
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from viridian402 import crops
from viridian402.data import read_rgb
from viridian402.groups import normalize_group
from viridian402.metrics import score


def _json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _crop_fixture(root, aliases=False):
    root.mkdir()
    (root / "objects").mkdir()
    pair = ("api:6081140580", "challenge:60811405800300") if aliases else ("well:alpha", "well:bravo")
    for number in range(1000):
        seed = f"independent-review-{number}"
        splits = ["test" if int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()[:8], 16) % 5 == 0 else "train" for g in pair]
        if set(splits) == {"train", "test"}:
            break
    policy = {"schema": 1, "seed": seed, "document_groups": {}, "previously_seen_groups": []}
    records = []
    for index, (group, split) in enumerate(zip(pair, splits)):
        ident = f"document-{index}"
        policy["document_groups"][ident] = group
        image = Image.new("RGB", (8, 8), (220, 40, 20) if index else (30, 80, 210))
        pixel = hashlib.sha256(b"8x8:" + image.tobytes()).hexdigest()
        native, normalized, mask = (f"objects/{pixel}{suffix}.png" for suffix in ("", "-224", "-mask"))
        image.save(root / native)
        image.resize((224, 224)).save(root / normalized)
        Image.new("L", (224, 224), 255).save(root / mask)
        records.append({"sample_id": ident, "document_id": ident, "source_group": group,
                        "split": split, "evaluation_status": "prospective", "accepted": True,
                        "kind": "real", "quality": "accepted", "quality_issues": [],
                        "source_size": [8, 8], "box_xyxy": [0, 0, 8, 8],
                        "targets": [{"id": "rock-a", "key": "a"}], "native_png": native,
                        "normalized_png": normalized, "valid_mask_png": mask, "pixel_sha256": pixel,
                        "normalization": {"source_size": [8, 8], "target_size": [224, 224], "content_xyxy": [0, 0, 224, 224]},
                        "usable_for_generation": split == "train", "usable_for_test": split == "test"})
    manifest = {"schema": "strata.real.v1", "snapshot": "independent-review", "split_policy": policy,
                "records": records, "labels": [{"id": "rock-a", "key": "a"}],
                "stats": {"accepted": 2, "train": 1, "test": 1, "needs_review": 0, "test_groups": 1}}
    _json(root / "manifest.json", manifest)
    _json(root / "split-policy.json", policy)
    for name, selected in (("train", [r for r in records if r["split"] == "train"]),
                           ("test", [r for r in records if r["split"] == "test"]), ("needs-review", [])):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
    _json(root / "generator-sources.json", [r for r in records if r["split"] == "train"])
    return root


def test_crop_fixed_split_rejects_same_physical_well_aliases(tmp_path):
    root = _crop_fixture(tmp_path / "real", aliases=True)
    assert normalize_group("api:6081140580") == normalize_group("challenge:60811405800300")
    with pytest.raises(ValueError, match="(?i)(group|leak|alias|canonical)"):
        crops.audit_crops(root)


def test_crop_training_groups_string_is_rejected(tmp_path, monkeypatch):
    root = _crop_fixture(tmp_path / "real")
    group = crops.audit_crops(root)["test"][0]["source_group"]
    monkeypatch.setattr(crops, "_predict", lambda *args: (["a"], None, .1, 1))
    with pytest.raises(ValueError, match="(?i)(training_groups|list|group)"):
        crops.classify(root, config={"training_groups": group}, output=tmp_path / "runs")


def test_crop_previous_use_alias_is_not_claimed_prospective(tmp_path):
    root = _crop_fixture(tmp_path / "real")
    manifest = json.loads((root / "manifest.json").read_text())
    test_group = next(r["source_group"] for r in manifest["records"] if r["split"] == "test")
    manifest["split_policy"]["previously_seen_groups"] = [test_group.upper()]
    assert normalize_group(test_group.upper()) == normalize_group(test_group)
    _json(root / "manifest.json", manifest)
    _json(root / "split-policy.json", manifest["split_policy"])
    with pytest.raises(ValueError, match="(?i)(status|previous|prospective|retrospective)"):
        crops.audit_crops(root)


def test_crop_saved_gate_cannot_contradict_metrics(tmp_path, monkeypatch):
    root = _crop_fixture(tmp_path / "real")
    monkeypatch.setattr(crops, "_predict", lambda *args: (["wrong"], None, .1, 1))
    run, summary = crops.classify(root, output=tmp_path / "runs", min_score=1)
    assert not summary["quality_gates"]["passed"]
    summary["quality_gates"]["passed"] = True
    _json(run / "summary.json", summary)
    with pytest.raises(ValueError, match="(?i)(gate|check|summary|integrity)"):
        crops.load_crop_run(run)


@pytest.mark.parametrize("field", ["training_groups", "training_labels", "test_only_labels", "counts"])
def test_crop_saved_training_claims_match_hashed_artifacts(tmp_path, monkeypatch, field):
    root = _crop_fixture(tmp_path / "real")
    monkeypatch.setattr(crops, "_predict", lambda *args: (["a"], None, .1, 1))
    run, summary = crops.classify(root, output=tmp_path / "runs")
    if field == "counts":
        summary[field]["test"] = 0
    else:
        summary[field] = ["not-in-the-audited-artifact"]
    _json(run / "summary.json", summary)
    with pytest.raises(ValueError):
        crops.load_crop_run(run)


def test_bundle_rejects_platform_specific_relative_paths(tmp_path):
    module_path = Path(__file__).parents[1] / "scripts" / "datasets.py"
    spec = importlib.util.spec_from_file_location("review_dataset_bundle", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "image.png").write_bytes(b"fixture")
    manifest = tmp_path / "manifest.json"
    _json(manifest, {"records": [{"image": "assets\\image.png"}]})
    with pytest.raises(ValueError, match="(?i)(portable|relative|path)"):
        module.pack([manifest], tmp_path / "bundle.zip")


def test_bundle_output_cannot_overwrite_source_manifest(tmp_path):
    module_path = Path(__file__).parents[1] / "scripts" / "datasets.py"
    spec = importlib.util.spec_from_file_location("review_dataset_bundle_collision", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "manifest.json"
    _json(manifest, {"records": []})
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="(?i)(output|overwrite|input|source)"):
        module.pack([manifest], manifest)
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("unsafe", ["NUL", "a./file.txt", "a /file.txt", "COM1.txt"])
def test_archive_rejects_windows_aliases_before_writing(tmp_path, unsafe):
    module_path = Path(__file__).parents[1] / "scripts" / "datasets.py"
    spec = importlib.util.spec_from_file_location("review_dataset_archive_alias", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / "alias.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("first.txt", "must not be written before all paths are validated")
        bundle.writestr(unsafe, "fixture")
    with pytest.raises(ValueError, match="(?i)(unsafe|portable|reserved|path|alias)"):
        module.safe_extract(archive, tmp_path / "result")
    assert not (tmp_path / "result" / "first.txt").exists()


def test_randomized_intervals_match_independent_raster_and_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(8128)
    for _ in range(300):
        intervals, rasters, clusters = [], [], []
        for side in range(2):
            rows, pixels, ids = [], np.full(83, "", dtype=object), np.full(83, "__missing__", dtype=object)
            y = 0
            while y < 83:
                y += int(rng.integers(0, 4))
                if y >= 83:
                    break
                end = min(83, y + int(rng.integers(1, 12)))
                label, cluster = str(rng.choice(["a", "b", "c"])), int(rng.integers(0, 4))
                rows.append({"top_y": y, "bottom_y": end, "lithology": label, "cluster_id": cluster})
                pixels[y:end], ids[y:end] = label, str(cluster)
                y = end
            intervals.append(rows)
            rasters.append(pixels)
            clusters.append(ids)
        result = score(intervals[0], intervals[1], (7, 77))
        predicted, truth = rasters[0][7:77], rasters[1][7:77]
        filled = truth != ""
        predicted, truth = predicted[filled], truth[filled]
        labels = sorted(set(truth) | (set(predicted) - {""}))
        assert result["row_acc"] == pytest.approx(np.mean(predicted == truth))
        assert result["macro_f1"] == pytest.approx(sklearn_metrics.f1_score(truth, predicted, labels=labels, average="macro", zero_division=0))
        assert result["cluster_ari"] == pytest.approx(sklearn_metrics.adjusted_rand_score(truth, clusters[0][7:77][filled]))


def test_randomized_multilabel_scores_match_sklearn_with_errors():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(314159)
    vocabulary = ["a", "b", "c", "untrained", "spurious"]
    for _ in range(150):
        cases = []
        for case_index in range(13):
            truth = sorted(rng.choice(vocabulary[:4], size=int(rng.integers(1, 4)), replace=False).tolist())
            predicted = sorted(rng.choice(vocabulary, size=int(rng.integers(0, 5)), replace=False).tolist())
            error = "simulated worker failure" if rng.random() < .2 else None
            cases.append({"targets": truth, "predicted": predicted, "error": error, "candidate_labels": vocabulary[:3]})
        observed = crops._metrics(cases)
        actual_predictions = [set(case["predicted"]) if not case["error"] else set() for case in cases]
        classes = sorted(set().union(*(set(case["targets"]) | predicted for case, predicted in zip(cases, actual_predictions))))
        truth_matrix = np.asarray([[label in case["targets"] for label in classes] for case in cases], dtype=int)
        prediction_matrix = np.asarray([[label in predicted for label in classes] for predicted in actual_predictions], dtype=int)
        assert observed["micro_f1"] == pytest.approx(sklearn_metrics.f1_score(truth_matrix, prediction_matrix, average="micro", zero_division=0))
        assert observed["macro_f1"] == pytest.approx(sklearn_metrics.f1_score(truth_matrix, prediction_matrix, average="macro", zero_division=0))
        assert observed["exact_set_accuracy"] == pytest.approx(np.mean(np.all(truth_matrix == prediction_matrix, axis=1)))


def test_rgb16_tiff_strip_reader_matches_independent_pillow_decode(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.asarray([[[0, 65535, 32768], [40000, 20000, 10000]],
                       [[1000, 2000, 3000], [12345, 23456, 34567]]], dtype=np.uint16)
    path = tmp_path / "rgb16.tif"
    tifffile.imwrite(path, data, photometric="rgb", rowsperstrip=1)
    with Image.open(path) as image:
        expected = np.asarray(image.convert("RGB"))
    assert np.array_equal(read_rgb(path, (0, 0, 2, 2)), expected)
