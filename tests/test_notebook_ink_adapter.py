"""Contract and source-parity checks for the notebook inference-only port."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("scipy", reason="Notebook candidate needs the optional SciPy dependency")

from scipy.ndimage import label, center_of_mass
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist

from viridian402 import notebook_ink as model


def fixture_request(tmp_path):
    # Three repeated colour beds, with full-image legend swatches above them.
    rgb = np.full((340, 80, 3), 255, dtype=np.uint8)
    rgb[20:120, 10:50] = (240, 140, 140)
    rgb[120:220, 10:50] = (140, 140, 240)
    rgb[220:320, 10:50] = (240, 140, 140)
    rgb[0:12, 0:12] = (240, 140, 140)
    rgb[0:12, 20:32] = (140, 140, 240)
    image = tmp_path / "page.png"
    Image.fromarray(rgb).save(image)
    return dict(image=str(image), image_size=[80, 340], window=[20,320],
                annotation={"interpreted_lithology_column":[10,20,50,320]})


def test_supplied_swatches_preserve_coordinates_and_repeated_page_ids(tmp_path):
    request = fixture_request(tmp_path)
    request["legend"] = [dict(id=91,label="Red rock",swatch_box=[0,0,12,12]),
                         dict(id=12,label="Blue rock",swatch_box=[20,0,32,12])]
    rows = model.predict(request, dict(stripe_height=2,boundary_window=2,min_gap=2,ink_k=1))
    assert [r["lithology"] for r in rows] == ["Red rock","Blue rock","Red rock"]
    assert [r["cluster_id"] for r in rows] == [0,1,0]
    assert [(r["top_y"],r["bottom_y"]) for r in rows] == [(20,120),(120,220),(220,320)]
    assert all("top_depth" not in r and "bottom_depth" not in r for r in rows)


def test_no_legend_has_unknown_names_and_no_invented_depth(tmp_path):
    request = fixture_request(tmp_path)
    request["depth"] = {"a":2,"b":100,"unit":"m"}
    request["truth"] = "DO_NOT_OPEN"
    request["layers"] = ["DO_NOT_USE"]
    rows = model.predict(request, dict(stripe_height=2,boundary_window=2,min_gap=2,ink_k=1))
    assert len(rows) == 3
    assert [r["cluster_id"] for r in rows] == [0,1,0]
    assert all(r["lithology"] == "Unknown" and "top_depth" not in r for r in rows)


def test_blank_and_window_clipping(tmp_path):
    request = fixture_request(tmp_path)
    request["window"] = [0,340]
    rows = model.predict(request,{})
    assert rows[0]["top_y"] == 20
    assert rows[-1]["bottom_y"] == 320
    Image.new("RGB",(80,340),"white").save(request["image"])
    assert model.predict(request,{}) == []
    request["window"] = [0,10]
    assert model.predict(request,{}) == []


@pytest.mark.parametrize("config", [{"max_feature_stripes":1},{"stripe_height":0},
                                      {"boundary_window":False},{"ink_k":float("nan")},
                                      {"max_component_centers":2001},{"max_window_pixels":20}])
def test_resource_and_parameter_guards(tmp_path,config):
    with pytest.raises(ValueError):
        model.predict(fixture_request(tmp_path),config)


def test_features_and_ink_rule_match_notebook_definitions():
    # Extract definitions only. Never execute notebook training or data cells.
    notebook = json.loads((Path(__file__).parents[1]/"notebooks/main_pipeline.ipynb").read_text(encoding="utf-8"))
    scope = dict(np=np,label=label,center_of_mass=center_of_mass,pdist=pdist,find_peaks=find_peaks,
                 H=8,WINDOW=12,INK_K=6,MIN_GAP=6)
    names = {"trim_borders","stripe_features","column_to_vectors","raw_stripe_features",
             "zscore","column_features","boundary_score","robust_threshold","ink_boundaries"}
    for index in [10,12]:
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        definitions = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),"notebook-functions","exec"),scope)
    rng = np.random.default_rng(42)
    rgb = rng.integers(0,256,(320,32,3),dtype=np.uint8)
    np.testing.assert_allclose(model.column_features(rgb),scope["column_features"](rgb))
    f = model.column_features(rgb)
    np.testing.assert_array_equal(model.ink_boundaries(f,w=3,k=1,gap=2),
                                  scope["ink_boundaries"](f,w=3,k=1,gap=2))
