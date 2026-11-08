import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest

from tools.location import Location
from validation import ValidationResult, apply_corrections, reference, submission_df, validate
from validation.config import COARSE_CLASSES, SUBMISSION_COLUMNS

HEADER_LOC = Location(lat=28.0, lon=-90.0, source="header", confidence=0.9, api_number="608084001400")


@pytest.fixture(autouse=True)
def empty_reference(tmp_path: Path):
    """Run every test against an empty reference folder unless it plants files."""
    reference.set_reference_dir(tmp_path)
    yield tmp_path
    reference.set_reference_dir(None)


def table(names: list[str], thick: float | list[float] = 20.0, gap: float = 0.0) -> pd.DataFrame:
    """Build a contiguous bed table in the six submission columns."""
    t = thick if isinstance(thick, list) else [thick] * len(names)
    tops = np.r_[8000.0, 8000.0 + np.cumsum(np.array(t) + gap)[:-1]]
    return pd.DataFrame({"top_y": (tops - 8000) * 4, "bottom_y": (tops + np.array(t) - 8000) * 4,
                         "cluster_id": range(len(names)), "lithology": names,
                         "top_depth": tops, "base_depth": tops + np.array(t)})


def plant_region(folder: Path) -> None:
    """Write a Macrostrat grid that is 90% mud and 10% sand_silt around (28, -90)."""
    rows = [{"lat": 28 + dy, "lon": -90 + dx, **{c: 0.0 for c in COARSE_CLASSES}, "n_units": 5}
            for dy in (-0.5, 0, 0.5) for dx in (-0.5, 0, 0.5)]
    for r in rows:
        r["mud"], r["sand_silt"] = 0.9, 0.1
    pd.DataFrame(rows).to_csv(folder / "macrostrat_grid.csv", index=False)


def clean_names(n: int = 40) -> list[str]:
    """Mostly shale-like beds with some sand."""
    cycle = ["Shale grey", "Siltstone", "Marl", "Shale grey", "Sandstone"]
    return [cycle[i % len(cycle)] for i in range(n)]


def test_works_without_reference_files() -> None:
    """Missing reference data drops components but confidence still computes."""
    res = validate(table(clean_names()), HEADER_LOC)
    assert 0.0 < res.confidence <= 1.0
    assert "structure" in res.available and "regional" not in res.available and res.components["regional"] is None


def test_gap_scores_lower_on_structure() -> None:
    """A depth gap lowers the structure score."""
    names = clean_names(20)
    assert validate(table(names, gap=15.0)).components["structure"] < validate(table(names)).components["structure"]


def test_inputs_not_mutated_and_submission_columns() -> None:
    """The input is untouched and submission_df leaves exactly six columns."""
    df = table(clean_names())
    before = df.copy(deep=True)
    out = apply_corrections(df, validate(df, HEADER_LOC))
    pd.testing.assert_frame_equal(df, before)
    assert {"lithology_original", "corrected", "validation_confidence"} <= set(out.columns)
    assert list(submission_df(out).columns) == list(SUBMISSION_COLUMNS)


def test_coal_in_shale_is_corrected(empty_reference: Path) -> None:
    """A lone thin coal bed in thick shale is flagged and moved; other beds are untouched."""
    plant_region(empty_reference)
    names = clean_names(40)
    names[20] = "Coal"
    thick = [20.0] * 40
    thick[20] = 1.0
    df = table(names, thick)
    res = validate(df, HEADER_LOC)
    assert "regional" in res.available and res.per_bed.loc[20, "bed_confidence"] < 0.5
    out = apply_corrections(df, res)
    assert out.loc[20, "corrected"] and out.loc[20, "lithology"] in ("Shale grey", "Marl")
    assert out.loc[20, "lithology_original"] == "Coal"
    assert out["corrected"].sum() == 1
    unchanged = out.drop(index=20)
    assert (unchanged["lithology"] == df.drop(index=20)["lithology"]).all()
    for col in ("cluster_id", "top_y", "bottom_y", "top_depth", "base_depth"):
        assert (out[col] == df[col]).all()


def test_change_fraction_capped(empty_reference: Path) -> None:
    """Never more than 5% of beds change, even with many bad beds."""
    plant_region(empty_reference)
    names = clean_names(60)
    thick = [20.0] * 60
    for i in range(5, 60, 6):
        names[i], thick[i] = "Coal", 1.0
    df = table(names, thick)
    out = apply_corrections(df, validate(df, HEADER_LOC))
    assert 0 < out["corrected"].sum() <= int(0.05 * len(df))


def test_low_confidence_changes_nothing(empty_reference: Path) -> None:
    """Below the minimum overall confidence no label changes."""
    plant_region(empty_reference)
    df = table(clean_names(20))
    res = validate(df, HEADER_LOC)
    res.confidence = 0.1
    assert not apply_corrections(df, res)["corrected"].any()


def test_runs_fast_with_cached_reference(empty_reference: Path) -> None:
    """validate + apply_corrections on 100 beds takes under 100 ms once reference data is cached."""
    plant_region(empty_reference)
    rng = np.random.default_rng(0)
    pd.DataFrame({"api": range(5000), "lat": rng.uniform(25, 29, 5000), "lon": rng.uniform(-94, -86, 5000),
                  "water_depth_ft": 3000, "total_depth_ft": rng.uniform(10000, 25000, 5000)}).to_csv(empty_reference / "boem_wells.csv", index=False)
    grid = [(a, b, 1000.0) for a in np.arange(24, 31, 0.1) for b in np.arange(-98, -80, 0.1)]
    pd.DataFrame(grid, columns=["lat", "lon", "depth_m"]).to_csv(empty_reference / "bathymetry_gulf.csv", index=False)
    df = table(clean_names(100))
    apply_corrections(df, validate(df, HEADER_LOC))
    start = time.perf_counter()
    res = validate(df, HEADER_LOC)
    apply_corrections(df, res)
    assert time.perf_counter() - start < 0.1
    assert {"structure", "sequence", "regional", "neighbours", "location"} <= set(res.available)


def test_water_check_flags_land(empty_reference: Path) -> None:
    """An offshore API prefix on a land point scores zero on location."""
    pd.DataFrame([(a, b, -50.0) for a in np.arange(27, 29.1, 0.5) for b in np.arange(-91, -88.9, 0.5)],
                 columns=["lat", "lon", "depth_m"]).to_csv(empty_reference / "bathymetry_gulf.csv", index=False)
    res = validate(table(clean_names(10)), HEADER_LOC)
    assert res.components["location"] == 0.0 and any("offshore_well_on_land" in f for f in res.flags)
