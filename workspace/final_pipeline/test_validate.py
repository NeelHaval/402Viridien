"""Synthetic validation tests; set HTB_RUN_OCR_TESTS=1 to test real dataset headers."""
from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from . import validate as validation
except ImportError:
    import validate as validation

DATASET = Path(__file__).resolve().parents[1] / "viridien" / "dataset"
HADRIAN_STEM = "60808400140000_hadrian 2_masterlog_2in_10000-21962"
CSV_HEADER = "top_y,bottom_y,cluster_id,lithology,top_depth,base_depth\n"


class SyntheticValidationTests(unittest.TestCase):
    """Exercise thickness accounting, API parsing, coordinates, and confidence offline."""

    def test_coarse_classes_and_priority(self) -> None:
        """Specific carbonate keywords win over generic clay keywords."""
        cases = {"Dolomitic clay": "carbonate", "Shale grey": "mud", "Sandstone": "sand_silt",
                 "Halite": "evaporite", "Lignite": "organic", "Granite": "volcanic", "Basalt": "evaporite", "Mystery": "other"}
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(validation.coarse_class(name), expected)

    def test_csv_thickness_shares_entropy_and_bed_rate(self) -> None:
        """Unknown beds and invalid depths do not inflate logged composition."""
        rows = ("0,10,1,Sandstone|Shale grey,1000,1100\n"
                "10,20,2,blank,1100,1200\n20,30,3,Halite,1200,1250\n"
                "30,40,4,Shale,1300,1290\n40,50,5,Clay,nan,1400\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beds.csv"
            path.write_text(CSV_HEADER + rows, encoding="utf-8")
            with self.assertLogs(level="INFO") as messages:
                fingerprint = validation.fingerprint_from_csv(path)
        self.assertIn("Skipped 100.000 ft", " ".join(messages.output))
        self.assertEqual(fingerprint.thickness_ft, 150)
        self.assertEqual(fingerprint.bed_rate, 8)
        self.assertAlmostEqual(fingerprint.proportions["mud"], 1 / 3)
        self.assertAlmostEqual(fingerprint.proportions["evaporite"], 1 / 3)
        self.assertAlmostEqual(fingerprint.entropy, validation.math.log2(3))
        self.assertAlmostEqual(sum(fingerprint.proportions.values()), 1)

    def test_mixed_unknown_name_skips_only_its_share(self) -> None:
        """A known name retains its original equal share when another name is unknown."""
        weights, skipped, _, _ = validation.bed_weights(
            {"top_depth": 0, "base_depth": 100, "lithology": "Sandstone|unknown"})
        self.assertEqual(weights, {"sand_silt": 50})
        self.assertEqual(skipped, 50)

    def test_csv_missing_or_malformed_returns_empty_fingerprint(self) -> None:
        """Bad CSV input yields an explicit zero-thickness result and a warning."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.csv"
            with self.assertLogs(level="WARNING"):
                missing = validation.fingerprint_from_csv(path)
            path.write_text("lithology\nShale\n", encoding="utf-8")
            with self.assertLogs(level="WARNING"):
                malformed = validation.fingerprint_from_csv(path)
        for fingerprint in (missing, malformed):
            self.assertEqual(fingerprint.thickness_ft, 0)
            self.assertIsNone(fingerprint.bed_rate)
            self.assertEqual(sum(fingerprint.proportions.values()), 1)

    def test_macrostrat_observed_schema_and_thickness_fallback(self) -> None:
        """Observed numeric prop and metre thickness determine regional shares."""
        payload = {"success": {"data": [
            {"max_thick": 10, "min_thick": 1, "lith": [
                {"name": "sand", "prop": .75, "type": "siliciclastic", "class": "sedimentary", "atts": []},
                {"name": "shale", "prop": .25, "atts": []}]},
            {"max_thick": 0, "min_thick": 10, "lith": [{"name": "Dolomitic clay", "prop": 1}]}]}}
        fingerprint = validation.fingerprint_from_response(payload)
        self.assertIsNotNone(fingerprint)
        assert fingerprint is not None
        self.assertAlmostEqual(fingerprint.thickness_ft, 20 * validation.FEET_PER_METRE)
        self.assertAlmostEqual(fingerprint.proportions["carbonate"], .5)
        self.assertAlmostEqual(fingerprint.proportions["sand_silt"], .375)
        self.assertIsNone(fingerprint.bed_rate)

    def test_macrostrat_missing_coverage_returns_none(self) -> None:
        """Empty or malformed responses never fabricate regional geology."""
        for payload in (None, {}, {"success": {"data": []}}, {"success": {"data": [{"lith": []}]}}):
            with self.subTest(payload=payload), self.assertLogs(level="WARNING"):
                self.assertIsNone(validation.fingerprint_from_response(payload))

    def test_dominance_fallback(self) -> None:
        """Dominant attributes outweigh subordinate ones when numeric prop is absent."""
        self.assertGreater(validation.lithology_weight({"atts": ["dominant"]}),
                           validation.lithology_weight({"atts": ["subordinate"]}))
        self.assertEqual(validation.lithology_weight({"prop": 0, "atts": ["dominant"]}), 0)

    def test_labelled_dms_and_hemisphere(self) -> None:
        """DMS headers retain signs and report an absent hemisphere."""
        text = 'Latitude: 27 deg 40\' 46.9146" N\nLongitude: 90 deg 17\' 14.3107" W'
        location = validation.location_from_text(text)
        self.assertIsNotNone(location)
        assert location is not None
        self.assertAlmostEqual(location.lat, 27 + 40 / 60 + 46.9146 / 3600)
        self.assertAlmostEqual(location.lon, -(90 + 17 / 60 + 14.3107 / 3600))
        self.assertFalse(location.assumed_hemisphere)
        assumed = validation.location_from_text(text.removesuffix(" W"))
        self.assertTrue(assumed and assumed.assumed_hemisphere)

    def test_coordinates_with_adjacent_header_fields(self) -> None:
        """Other numeric header fields cannot contaminate coordinate values."""
        text = ('Latitude: 27°43\'30.946"N\nX COORD: 2,374,517 ft\n'
                'Longitude: 90°43\'49.427"W\nY COORD: 10,067,765 ft')
        location = validation.location_from_text(text)
        self.assertIsNotNone(location)
        assert location is not None
        self.assertAlmostEqual(location.lat, 27 + 43 / 60 + 30.946 / 3600)
        self.assertAlmostEqual(location.lon, -(90 + 43 / 60 + 49.427 / 3600))

    def test_decimal_and_invalid_coordinates(self) -> None:
        """Reject out-of-range values and conflicting directions."""
        self.assertEqual(validation.parse_coordinate("-90.5", 180), (-90.5, False))
        for text, limit in (("27 60 0 N", 90), ("181 E", 180), ("-27 N", 90), ("27 E", 90), ("nan", 90)):
            with self.subTest(text=text):
                self.assertIsNone(validation.parse_coordinate(text, limit))
        self.assertIsNone(validation.location_from_text("No coordinates here"))

    def test_ocr_v2_and_v3_shapes(self) -> None:
        """Both supported OCR result schemas yield only recognised text."""
        self.assertEqual(validation.ocr_strings([[[[0, 0]], ("Latitude: 27 N", .98)]]), ["Latitude: 27 N"])
        self.assertEqual(validation.ocr_strings([{"rec_texts": ["Longitude: 90 W"]}]), ["Longitude: 90 W"])
        self.assertEqual(validation.ocr_strings(None), [])

    def test_header_annotations_and_missing_data(self) -> None:
        """Header cropping stops before the earliest supplied column or legend."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boxes.json"
            path.write_text(json.dumps({"bounding_boxes": {"depth_column": [0, 100, 10, 200],
                                   "lithology_legend": [0, 80, 100, 90]}}), encoding="utf-8")
            self.assertEqual(validation.header_bottom(path), 80)
            path.write_text("{broken", encoding="utf-8")
            with self.assertLogs(level="WARNING"):
                self.assertIsNone(validation.header_bottom(path))

    def test_confidence_agreement_disagreement_and_assumption(self) -> None:
        """Scores have interpretable endpoints and a documented hemisphere penalty."""
        sand = validation.make_fingerprint({"sand_silt": 100}, "predicted")
        shale = validation.make_fingerprint({"mud": 100}, "macrostrat")
        location = validation.Location(27, -90, False)
        self.assertEqual(validation.compare_fingerprints(sand, sand, location).score, 1)
        self.assertEqual(validation.compare_fingerprints(sand, shale, location).score, 0)
        assumed = validation.Location(27, 90, True)
        self.assertEqual(validation.compare_fingerprints(sand, sand, assumed).score, .8)
        for regional, point in ((None, location), (shale, None)):
            result = validation.compare_fingerprints(sand, regional, point)
            self.assertIsNone(result.score)
            self.assertTrue(result.reasons)

    def test_explicit_coordinates_bypass_ocr(self) -> None:
        """Known coordinates do not require images, OCR models, or a network."""
        args = validation.argparse.Namespace(lat=27.5, lon=-90.5, image=None, annotation=None)
        with patch.object(validation, "ocr_header", side_effect=AssertionError("OCR should not run")):
            self.assertEqual(validation.resolve_location(args), validation.Location(27.5, -90.5, False))


@unittest.skipUnless(os.environ.get("HTB_RUN_OCR_TESTS") == "1", "Set HTB_RUN_OCR_TESTS=1 for real-image PaddleOCR")
class DatasetLocationTests(unittest.TestCase):
    """Run actual header OCR; missing dependencies or failed extraction fail explicitly."""

    def check_dataset_location(self, stem: str, expected_lat: float, expected_lon: float) -> None:
        """Compare actual OCR coordinates against manually read image header values."""
        annotation = DATASET / "bbox_annotation" / f"{stem}.json"
        self.assertTrue(annotation.is_file(), str(annotation))
        payload = json.loads(annotation.read_text(encoding="utf-8"))
        image = DATASET / "images" / payload["image_name"]
        self.assertTrue(image.is_file(), str(image))
        bottom = validation.header_bottom(annotation)
        self.assertIsNotNone(bottom)
        assert bottom is not None
        text = validation.ocr_header(image, bottom)
        location = validation.location_from_text(text or "")
        logging.info("%s: %s", image.name, location)
        self.assertIsNotNone(location, f"Real-image OCR did not produce a parseable location:\n{text}")
        assert location is not None
        self.assertAlmostEqual(location.lat, expected_lat, delta=.001)
        self.assertAlmostEqual(location.lon, expected_lon, delta=.001)
        self.assertFalse(location.assumed_hemisphere)

    def test_hadrian_location_from_real_image(self) -> None:
        """Hadrian prints 26°01'28.8638 N and 92°00'42.2312 W."""
        self.check_dataset_location(HADRIAN_STEM, 26 + 1 / 60 + 28.8638 / 3600, -(92 + 42.2312 / 3600))

    def test_droshky_location_from_real_image(self) -> None:
        """Droshky prints 27°43'30.946 N and 90°43'49.427 W."""
        self.check_dataset_location("60811405120000_1inch tvd mudlog",
                                    27 + 43 / 60 + 30.946 / 3600, -(90 + 43 / 60 + 49.427 / 3600))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
