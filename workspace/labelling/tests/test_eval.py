"""Tests for eval.py on small synthetic GT / prediction fixtures."""
import csv
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import eval as ev  # noqa: E402

HDR = ["top_y", "bottom_y", "cluster_id", "lithology", "top_depth", "base_depth"]
# 10 px/ft, y 0..1000 -> depth 1000..1100 ft
TICKS = [{"y": 0, "depth": 1000}, {"y": 500, "depth": 1050}, {"y": 1000, "depth": 1100}]


def d(y):
    return 1000 + y / 10


def write_csv(path, rows, depths=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        for r in rows:
            top, bot, cid, lith = r[:4]
            w.writerow([top, bot, cid, lith, f"{d(top):.1f}" if depths else "", f"{d(bot):.1f}" if depths else ""])


def write_gt(root, stem, rows, split="test"):
    """rows: (top, bot, cluster, lith, flags)"""
    write_csv(root / "labels_csv" / f"{stem}.csv", rows)
    label = {"image_name": f"{stem}.tif", "split": split, "legend": [],
             "depth_calibration": {"anchors": [{"y": t['y'], "depth_ft": t['depth']} for t in TICKS]},
             "layers": [{"top_px": t, "bottom_px": b, "flags": list(f),
                         "composition": [{"legend_id": c, "type": l, "proportion": 1., "x_range": [0,100]}]}
                        for t,b,c,l,f in rows]}
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "labels" / f"{stem}.json").write_text(json.dumps(label))


@pytest.fixture
def env(tmp_path):
    def run(test_stems, seen=(), model="m", extra=()):
        (tmp_path / "splits.json").write_text(json.dumps({"test": list(test_stems), "test_bw": list(seen)}))
        ev.main([str(tmp_path / "preds" / model), "--gt-dir", str(tmp_path / "labels_csv"),
                 "--labels-dir", str(tmp_path / "labels"), "--splits", str(tmp_path / "splits.json"),
                 "--out", str(tmp_path / "results"), *extra])
        return json.loads((tmp_path / "results" / model / "summary.json").read_text())
    run.root = tmp_path
    return run


GT3 = [(0, 300, 0, "sandstone", ()), (300, 600, 1, "shale", ()), (600, 1000, -1, "blank", ())]


def block(s, b="test", agg="mean_over_pages"):
    return s["blocks"][b][agg]


def test_depth_model_extrapolates():
    dm = ev.DepthModel([{"y": 0, "depth": 0}, {"y": 10, "depth": 10}, {"y": 20, "depth": 30}])
    assert dm.depth(-5) == pytest.approx(-5)
    assert dm.depth(15) == pytest.approx(20)
    assert dm.depth(30) == pytest.approx(50)
    assert dm.y_at(dm.depth(27.3)) == pytest.approx(27.3)


def test_perfect(env):
    write_gt(env.root, "p1", GT3)
    write_csv(env.root / "preds/m/p1.csv", [r[:4] for r in GT3])
    s = env(["p1"])
    for agg in ("mean_over_pages", "pooled"):
        b = block(s, agg=agg)
        assert b["acc"] == 1 and b["macro_f1"] == 1
        assert b["bf1@2"] == 1 and b["bf1@5"] == 1
        assert b["ari"] == 1 and b["purity"] == 1
        assert b["depth_mae"] == pytest.approx(0, abs=0.06)
    assert s["blocks"]["test"]["pooled"]["n_bins"] == 100


def test_shift_3ft(env):
    gt = [(0, 500, 0, "sandstone", ()), (500, 1000, 1, "shale", ())]
    write_gt(env.root, "p1", gt)
    write_csv(env.root / "preds/m/p1.csv", [(0, 530, 0, "sandstone"), (530, 1000, 1, "shale")])
    b = block(env(["p1"]))
    assert b["bf1@2"] == pytest.approx(2/3) and b["bf1@5"] == 1
    assert b["mean_err"] == pytest.approx(1)
    assert b["acc"] == pytest.approx(0.97)


def test_pipe_set_matching(env):
    gt = [(0, 500, 0, "clay|shale", ()), (500, 1000, 1, "sandstone", ())]
    write_gt(env.root, "p1", gt)
    write_csv(env.root / "preds/m/p1.csv", [(0, 250, 0, "shale"), (250, 500, 0, "limestone|clay"),
                                            (500, 1000, 1, "sand|siltstone")])
    s = env(["p1"])
    assert block(s)["acc"] == pytest.approx(0.5)
    conf = list(csv.reader(open(env.root / "results/m/confusion.csv")))
    hdr = conf[0]
    rows = {r[1]: dict(zip(hdr, r)) for r in conf[1:]}
    assert rows["shale"]["shale"] == "25" and rows["clay"]["clay"] == "25"
    assert rows["sandstone"]["sand"] == "50"


def test_uncertain_and_unknown_excluded_thin_bed_kept(env):
    gt = [(0, 300, 0, "sandstone", ()), (300, 500, 1, "shale", ("uncertain",)),
          (500, 600, 2, "limestone", ("uncertain",)), (600, 800, -1, "unknown", ("uncertain",)),
          (800, 1000, 3, "coal", ("thin_bed",))]
    write_gt(env.root, "p1", gt)
    write_csv(env.root / "preds/m/p1.csv", [(0, 300, 0, "sandstone"), (300, 800, 9, "dolomite"),
                                            (800, 1000, 3, "coal")])
    s = env(["p1"])
    assert s["blocks"]["test"]["pooled"]["n_bins"] == 50
    assert block(s)["acc"] == 1


def test_thin_pred_bed_merged(env):
    gt = [(0, 500, 0, "sandstone", ()), (500, 1000, 1, "shale", ())]
    write_gt(env.root, "p1", gt)
    write_csv(env.root / "preds/m/p1.csv", [(0, 200, 0, "sandstone"), (200, 210, 5, "limestone"),
                                            (210, 500, 0, "sandstone"), (500, 1000, 1, "shale")])
    b = block(env(["p1"], extra=["--thin-ft", "2"]))
    assert b["bf1@2"] == 1 and b["acc"] == 1
    pp = list(csv.DictReader(open(env.root / "results/m/per_page.csv")))
    assert pp[0]["n_pred_rows"] == "4" and pp[0]["n_pred_rows_merged"] == "2"


def test_merge_thin_into_thicker_neighbour():
    rows = [{"top": 0, "bot": 100}, {"top": 100, "bot": 105}, {"top": 105, "bot": 400}]
    out = ev.merge_thin(rows, lambda r: r["bot"] - r["top"], 10)
    assert [(r["top"], r["bot"]) for r in out] == [(0, 100), (100, 400)]


def test_missing_file(env, capsys):
    write_gt(env.root, "p1", GT3)
    write_gt(env.root, "p2", GT3)
    write_csv(env.root / "preds/m/p1.csv", [r[:4] for r in GT3])
    s = env(["p1", "p2"])
    out = capsys.readouterr().out
    assert "MISSING PREDICTION" in out and "p2" in out
    assert block(s)["acc"] == pytest.approx(0.5)
    assert s["blocks"]["test"]["pooled"]["acc"] == pytest.approx(0.5)
    assert s["blocks"]["test"]["n_missing_pred"] == 1
    pp = {r["stem"]: r for r in csv.DictReader(open(env.root / "results/m/per_page.csv"))}
    assert pp["p2"]["status"] == "missing_pred" and float(pp["p2"]["acc"]) == 0


def test_synthetic_variant_routing(env, capsys):
    write_gt(env.root, "p1", GT3)
    write_gt(env.root, "s1", GT3, split="test_bw")
    write_csv(env.root / "preds/m/p1.csv", [r[:4] for r in GT3])
    write_csv(env.root / "preds/m/s1.csv", [r[:4] for r in GT3])
    # gray variant: everything predicted as shale
    write_csv(env.root / "preds/m/p1__gray.csv", [(0, 1000, 0, "shale")])
    write_csv(env.root / "preds/m/p1__bogus.csv", [(0, 1000, 0, "shale")])
    write_csv(env.root / "preds/m/s1__otsu.csv", [(0, 1000, 0, "shale")])
    s = env(["p1"], seen=["s1"])
    out = capsys.readouterr().out
    assert set(s["blocks"]) == {"test", "test_bw", "test_bw_synthetic/gray"}
    assert block(s)["acc"] == 1 and block(s, "test_bw")["acc"] == 1
    assert block(s, "test_bw_synthetic/gray")["acc"] == pytest.approx(0.3)  # only the shale 30 ft
    assert "bogus" in out and "s1__otsu" in out


def test_compare_sorted(env, capsys):
    write_gt(env.root, "p1", GT3)
    write_csv(env.root / "preds/good/p1.csv", [r[:4] for r in GT3])
    write_csv(env.root / "preds/bad/p1.csv", [(0, 1000, 0, "shale")])
    (env.root / "splits.json").write_text(json.dumps({"test": ["p1"], "test_bw": []}))
    rows = ev.main(["--compare", str(env.root / "preds/bad"), str(env.root / "preds/good"),
                    "--gt-dir", str(env.root / "labels_csv"), "--labels-dir", str(env.root / "labels"),
                    "--splits", str(env.root / "splits.json"), "--out", str(env.root / "results")])
    cmp_rows = list(csv.DictReader(open(env.root / "results/compare.csv")))
    assert [r["model"] for r in cmp_rows] == ["good", "bad"]


def test_thin_ft_warning(env, capsys):
    write_gt(env.root, "p1", GT3)
    write_csv(env.root / "preds/m/p1.csv", [r[:4] for r in GT3])
    env(["p1"], extra=["--thin-ft", "3"])
    assert "WARNING: --thin-ft" in capsys.readouterr().out


def test_pseudo_mode(tmp_path, capsys):
    pdir = tmp_path / "pseudo"
    # pseudo GT: cluster only; page px_per_ft = 10 from page_features
    with open(pdir.mkdir() or pdir / "t1.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(HDR)
        w.writerows([[0, 500, 0, "", "", ""], [500, 800, 1, "", "", ""], [800, 1000, 0, "", "", ""]])
    pf = tmp_path / "pf.csv"
    pf.write_text("stem,px_per_ft\nt1,10\n")
    preds = tmp_path / "preds" / "m"
    preds.mkdir(parents=True)
    with open(preds / "t1__gray.csv", "w", newline="") as fh:  # boundaries shifted by 30 px = 3 ft
        w = csv.writer(fh); w.writerow(HDR)
        w.writerows([[0, 530, 7, "sandstone", "", ""], [530, 830, 8, "shale", "", ""],
                     [830, 1000, 7, "sandstone", "", ""]])
    with open(preds / "t1.csv", "w", newline="") as fh:  # perfect
        w = csv.writer(fh); w.writerow(HDR)
        w.writerows([[0, 500, 3, "", "", ""], [500, 800, 4, "", "", ""], [800, 1000, 3, "", "", ""]])
    ev.main([str(preds), "--gt", f"pseudo:{pdir}", "--page-features", str(pf), "--out", str(tmp_path / "res")])
    out = capsys.readouterr().out
    assert all(l.startswith("[PSEUDO]") for l in out.strip().splitlines())
    s = json.loads((tmp_path / "res" / "m__pseudo" / "summary.json").read_text())
    assert s["mode"] == "PSEUDO" and s["units"] == "px"
    p = s["blocks"]["PSEUDO"]["mean_over_pages"]
    assert p["bf1@2"] == 1 and p["ari"] == 1 and p["purity"] == 1
    g = s["blocks"]["PSEUDO/gray"]["mean_over_pages"]
    assert g["bf1@2"] == 0 and g["bf1@5"] == 1 and g["mean_err"] == pytest.approx(30)
    assert "acc" not in g
    assert not (tmp_path / "res" / "m").exists()


def test_gaps_overlaps_and_boundary_filtering(env):
    gt=[(0,300,0,'sandstone',()),(200,400,1,'shale',()),(700,1000,2,'coal',())]
    write_gt(env.root,'p1',gt)
    # shale is valid in the overlap; predictions inside the unlabelled gap are ignored.
    write_csv(env.root/'preds/m/p1.csv',[(0,200,0,'sandstone'),(200,400,1,'shale'),(500,600,9,'bad'),(700,1000,2,'coal')])
    s=env(['p1']); b=block(s,agg='pooled')
    assert b['n_bins']==70 and b['acc']==1
    assert b['bf1@2']==pytest.approx(10/11) # GT top/bottom at 300 has no prediction


def test_no_thin_preprocessing_by_default(env):
    gt=[(0,1000,0,'sandstone',())];write_gt(env.root,'p1',gt)
    write_csv(env.root/'preds/m/p1.csv',[(0,100,0,'sandstone'),(100,110,1,'shale'),(110,1000,0,'sandstone')])
    assert block(env(['p1']))['acc']==pytest.approx(.99)


def test_composition_optional_l1(env):
    gt=[(0,1000,0,'sandstone',())];write_gt(env.root,'p1',gt)
    write_csv(env.root/'preds/m/p1.csv',[r[:4] for r in gt])
    assert 'composition_l1' not in block(env(['p1']))
    (env.root/'preds/m/p1.composition.json').write_text(json.dumps([{'top_px':0,'bottom_px':1000,'composition':[
        {'type':'sandstone','proportion':.8},{'type':'shale','proportion':.2}]}]))
    s=env(['p1']);assert block(s)['composition_l1']==pytest.approx(.4)
    assert block(s,agg='pooled')['composition_l1']==pytest.approx(.4)


def test_uncertain_clustering_excluded(env):
    gt=[(0,500,0,'sandstone',()),(500,1000,1,'shale',('uncertain',))];write_gt(env.root,'p1',gt)
    write_csv(env.root/'preds/m/p1.csv',[(0,1000,3,'sandstone')])
    b=block(env(['p1']),agg='pooled');assert b['n_clu_bins']==50 and b['n_bins']==50


def test_shale_colour_variants_are_distinct_classes(env):
    gt=[(0,500,0,'shale gray',()),(500,1000,1,'shale brown',())]
    write_gt(env.root,'p1',gt)
    write_csv(env.root/'preds/m/p1.csv',[(0,1000,9,'shale')])
    assert block(env(['p1']))['acc']==0
    write_csv(env.root/'preds/m/p1.csv',[(0,1000,9,'shale gray')])
    assert block(env(['p1']))['acc']==pytest.approx(.5)
