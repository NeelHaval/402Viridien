"""Seeded local contract attacks; no external data or model subprocesses."""
from copy import deepcopy
import json
import random

import pytest
from PIL import Image

from viridian402.data import load_manifest, save_intervals, validate_intervals
from viridian402.groups import normalize_group


def _fixture(root):
    Image.new("RGB",(16,96),(31,81,121)).save(root/"one.png")
    Image.new("RGB",(16,96),(32,82,122)).save(root/"two.png")
    (root/"boxes.json").write_text(json.dumps({"width":16,"height":96,"bounding_boxes":{"interpreted_lithology_column":[0,0,16,96]}}),encoding="utf-8")
    save_intervals(root/"truth.csv",[{"top_y":12,"bottom_y":84,"lithology":"Shale","cluster_id":0}])
    record={"id":"one","image":"one.png","annotation":"boxes.json","truth":"truth.csv","truth_window":[12,84],"window":[12,84],"source":"human","group":"api:6081140512","split":"dev","tags":["local"],"label_provenance":{"kind":"stress-fixture"}}
    manifest={"schema_version":1,"name":"isolated-stress","records":[record]}
    return root/"manifest.json",manifest


def _write(path,value,bom=False):
    path.write_text(json.dumps(value),encoding="utf-8-sig" if bom else "utf-8")


def test_seeded_partial_truth_window_contract(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xC001)
    path,manifest=_fixture(tmp_path)
    for iteration in range(stress_iterations):
        top=rng.randint(12,82);bottom=rng.randint(top+1,84)
        current=deepcopy(manifest);current["records"][0]["window"]=[top,bottom]
        _write(path,current,bom=iteration%2==0)
        normalized=load_manifest(path)["records"][0]
        assert normalized["window"]==[top,bottom],(stress_seed,iteration)
        assert normalized["truth_window"]==[12,84]
        assert normalized["truth_sha256"] and normalized["image_sha256"]
        bad=deepcopy(current);bad["records"][0]["window"]=rng.choice(([11,bottom],[top,85],[top,top]))
        _write(path,bad)
        with pytest.raises(ValueError):load_manifest(path)


def test_seeded_group_alias_and_duplicate_image_leakage(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xC002)
    path,manifest=_fixture(tmp_path)
    aliases=[("api:60811405120000","api:6081140512"),("challenge:0000000009_d-582_log02","well:d582"),("challenge:MC765_P011_ST00BP00_1inchMD","well:mc765_p011"),("challenge:Shell_08852_KE001_ST01BP00_1inTVD_MudLog","document:shell08852ke001")]
    for iteration in range(stress_iterations):
        left,right=rng.choice(aliases)
        assert normalize_group(left)==normalize_group(right),(stress_seed,iteration,left,right)
        assert normalize_group(normalize_group(left))==normalize_group(left)
        current=deepcopy(manifest);a=current["records"][0];a["group"]=left
        b=deepcopy(a);b.update(id="two",image="two.png",group=right,split="test")
        current["records"].append(b);_write(path,current)
        with pytest.raises(ValueError,match="group leakage"):load_manifest(path)
        # Distinct group names cannot hide identical image bytes across splits.
        b.update(image="one.png",group="well:unrelated")
        _write(path,current)
        with pytest.raises(ValueError,match="duplicate image leakage"):load_manifest(path)
        b["split"]="dev";_write(path,current)
        assert len(load_manifest(path)["records"])==2


def test_seeded_interval_roundtrip_and_invalid_geometry(stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xC003)
    for iteration in range(stress_iterations):
        points=sorted(rng.sample(range(1,200),rng.randint(2,12)))
        rows=[{"top_y":str(a),"bottom_y":str(b),"lithology":rng.choice(["Sandstone","Shale","Unknown"]),"cluster_id":str(index),"top_depth":a/2,"base_depth":b/2} for index,(a,b) in enumerate(zip(points[::2],points[1::2]))]
        before=deepcopy(rows);normalized=validate_intervals(rows,window=[0,200])
        assert rows==before,(stress_seed,iteration,"validator mutated input")
        assert validate_intervals(normalized)==normalized
        assert all("base_depth" not in row and "bottom_depth" in row for row in normalized)
        bad=deepcopy(normalized)
        invalid=rng.choice(["zero","overlap","fraction","missing-cluster","nonfinite","half-depth"])
        if invalid=="zero":bad[0]["bottom_y"]=bad[0]["top_y"]
        elif invalid=="overlap":bad.append(deepcopy(bad[-1]))
        elif invalid=="fraction":bad[0]["top_y"]+=.25
        elif invalid=="missing-cluster":bad[0].pop("cluster_id")
        elif invalid=="nonfinite":bad[0]["top_depth"]=rng.choice([float("nan"),float("inf"),float("-inf")])
        else:bad[0].pop("bottom_depth")
        with pytest.raises(ValueError):validate_intervals(bad)


def test_signed_64_bit_contract_and_depth_alias_conflicts(stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xC004)
    for iteration in range(stress_iterations):
        cluster=rng.choice([-(2**63),2**63-1,rng.randint(-(2**62),2**62)])
        row={"top_y":str(2**63-3),"bottom_y":str(2**63-1),"lithology":"Shale","cluster_id":str(cluster)}
        result=validate_intervals([row])[0]
        assert result["cluster_id"]==cluster and result["bottom_y"]==2**63-1,(stress_seed,iteration)
        bad={**row,"cluster_id":rng.choice([2**63,-2**63-1,"1e1000000000"])}
        with pytest.raises(ValueError):validate_intervals([bad])
        with pytest.raises(ValueError,match="conflicting"):validate_intervals([{**row,"top_depth":0,"bottom_depth":1,"base_depth":2}])


@pytest.mark.parametrize("value",[[],None,"manifest",3,True,{"schema_version":1,"records":None}])
def test_wrong_json_top_level_is_controlled_value_error(tmp_path,value):
    path=tmp_path/"manifest.json";_write(path,value)
    with pytest.raises(ValueError):load_manifest(path)


@pytest.mark.parametrize("value",[[],None,"boxes",3,False])
def test_wrong_annotation_top_level_is_controlled_value_error(tmp_path,value):
    path,manifest=_fixture(tmp_path)
    _write(tmp_path/"boxes.json",value);_write(path,manifest)
    with pytest.raises(ValueError):load_manifest(path)


@pytest.mark.parametrize("text",["{",'{"schema_version":',"not json",'{"records": [}'])
def test_malformed_json_is_rejected(tmp_path,text):
    path=tmp_path/"manifest.json";path.write_text(text,encoding="utf-8")
    with pytest.raises(ValueError):load_manifest(path)


def test_seeded_manifest_label_metadata_and_checksums(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xC005)
    path,manifest=_fixture(tmp_path)
    for iteration in range(stress_iterations):
        current=deepcopy(manifest);record=current["records"][0]
        attack=rng.choice(["provenance","truth-window","duplicate-tags","nontext-tag","checksum","unlabelled-truth","duplicate-id","missing-file"])
        if attack=="provenance":record.pop("label_provenance")
        elif attack=="truth-window":record.pop("truth_window")
        elif attack=="duplicate-tags":record["tags"]=["same","same"]
        elif attack=="nontext-tag":record["tags"]=[42]
        elif attack=="checksum":record[rng.choice(["image_sha256","annotation_sha256","truth_sha256"])]= "0"*64
        elif attack=="unlabelled-truth":record["source"]="challenge_unlabelled"
        elif attack=="duplicate-id":current["records"].append(deepcopy(record))
        else:record["image"]="missing.png"
        _write(path,current)
        with pytest.raises(ValueError):load_manifest(path)


def test_seeded_synthetic_lineage_cannot_cross_parent_split_or_group(tmp_path,stress_seed,stress_iterations):
    rng=random.Random(stress_seed ^ 0xC006)
    path,manifest=_fixture(tmp_path)
    for iteration in range(stress_iterations):
        current=deepcopy(manifest)
        child=deepcopy(current["records"][0]);child.update(id="child",image="two.png",source="synthetic")
        child["lineage"]={"parent_id":"one","parent_group":"api:60811405120000","parent_split":"dev"}
        current["records"].append(child);_write(path,current)
        assert len(load_manifest(path)["records"])==2,(stress_seed,iteration)
        if rng.choice([True,False]):child["lineage"]["parent_split"]="train"
        else:child["lineage"]["parent_group"]="well:unrelated"
        _write(path,current)
        with pytest.raises(ValueError,match="synthetic"):load_manifest(path)


def test_declared_truth_coverage_does_not_create_truth_pixels(tmp_path):
    path,manifest=_fixture(tmp_path)
    manifest["records"][0].update(truth_window=[0,96],window=[0,10])
    _write(path,manifest)
    with pytest.raises(ValueError,match="no truth-filled pixels"):load_manifest(path)
