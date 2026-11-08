"""Offline tests. All generated data stays under external_data/kgs/tests.

Run: python -I -X utf8 external/test_kgs_extract.py
"""
import sys
sys.dont_write_bytecode = True

import csv
import importlib.util
import json
import re
import unittest
import zlib
from unittest.mock import patch
from pathlib import Path

import fitz
import numpy as np
from PIL import Image

spec = importlib.util.spec_from_file_location("kgs_extract", Path(__file__).with_name("kgs_extract.py"))
kgs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kgs)
TEST_OUT = kgs.OUTPUT_ROOT / "tests"


def add_tile(doc, page, rect, colour, xref=0):
    if xref:
        return page.insert_image(rect, xref=xref, keep_proportion=False)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 16, 8), False)
    pix.clear_with(colour)
    return page.insert_image(rect, pixmap=pix, keep_proportion=False)


def miniature(doc, *, legend=True, ref=0):
    """A little vector log, with four rotated depth words and drawn borders."""
    page = doc.new_page(width=300, height=700)
    if legend:
        page.insert_text((110, 30), "ROCK TYPES", fontsize=10)
        ref = add_tile(doc, page, fitz.Rect(20, 40, 63.2, 48), 100, ref)
        page.insert_text((70, 48), "RockA", fontsize=10)
        page.insert_text((110, 80), "ACCESSORIES", fontsize=10)
    for x in (100, 115, 150, 186):
        page.draw_line((x, 150), (x, 650))
    for i in range(5):
        page.insert_text((110, 190+i*100), str(1000+50*i), fontsize=8, rotate=90)
    for y in range(200, 600, 8):
        add_tile(doc, page, fitz.Rect(150, y, 186, y+8), 100, ref)
    return ref


def inline_tiles(doc, page):
    """Replace image Do operators with equivalent BI streams, alternating filters."""
    refs = {im[7].encode(): im[0] for im in page.get_images()}
    for index, ref in enumerate(page.get_contents()):
        def replace(match):
            pix = fitz.Pixmap(doc, refs[match[1]])
            raw = pix.samples
            filtering = b" /F /Fl" if index % 2 else b""
            data = zlib.compress(raw) if filtering else raw
            return (f"BI /W {pix.width} /H {pix.height} /BPC 8 /CS /RGB".encode()
                    + filtering + b" ID\n" + data + b"\nEI")
        doc.update_stream(ref, re.sub(rb"/([A-Za-z0-9]+)\s+Do", replace, doc.xref_stream(ref)))


class SampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not kgs.SAMPLE.is_file():
            raise unittest.SkipTest('Optional source PDF is not installed: ' + str(kgs.SAMPLE))
        TEST_OUT.mkdir(parents=True, exist_ok=True)
        cls.out = TEST_OUT / "sample"
        cls.pages = kgs.extract_pdf(kgs.SAMPLE, cls.out)
        if cls.pages[0]["status"] != "ok":
            raise AssertionError(cls.pages)
        cls.meta = cls.pages[0]["meta"]
        with (cls.out / "truth.csv").open(encoding="utf-8") as f:
            cls.rows = list(csv.DictReader(f))
        cls.doc = fitz.open(kgs.SAMPLE)
        cls.page = cls.doc[0]
        cls.images = kgs.placed_images(cls.page)

    @classmethod
    def tearDownClass(cls):
        cls.doc.close()

    def test_20_rock_types_and_accessory_categories(self):
        entries = self.meta["legend_entries"]
        expected = {"Anhy", "Bent", "Brec", "Cht", "Clyst", "Coal", "Congl", "Dol", "Gyp", "Igne",
                    "Lmst", "Meta", "Mrlst", "Salt", "Shale", "Shcol", "Shgy", "Sltst", "Ss", "Till"}
        self.assertEqual({e["label"] for e in entries if e["kind"] == "rock"}, expected)
        self.assertEqual(sum(e["kind"] == "rock" for e in entries), 20)
        self.assertEqual({e["kind"] for e in entries}, {"rock", "fossil", "mineral", "texture", "stringer"})
        self.assertTrue(all((self.out / e["swatch_png"]).is_file() for e in entries))

    def test_xrefs_and_track_borders(self):
        self.assertEqual(self.meta["rock_xrefs_in_column"],
                         {"39": "Lmst", "43": "Shale", "47": "Ss", "46": "Sltst"})
        np.testing.assert_allclose(self.meta["track_bounds_pt"][:1]+self.meta["track_bounds_pt"][2:3],
                                   [212.4, 248.4], atol=0.001)

    def test_truth_tiles_full_plot_and_logged_interval(self):
        rows = self.rows
        self.assertEqual(len(rows), 124)
        self.assertEqual(rows[0]["lithology"], "(blank)")
        self.assertEqual(rows[-1]["lithology"], "(blank)")
        for a, b in zip(rows, rows[1:]):
            self.assertEqual(int(a["bottom_y"]), int(b["top_y"]))
            self.assertEqual(float(a["bottom_depth"]), float(b["top_depth"]))
        self.assertTrue(all(int(r["bottom_y"]) > int(r["top_y"]) for r in rows))
        box = self.meta["track_bounds_px"]
        self.assertEqual((int(rows[0]["top_y"]), int(rows[-1]["bottom_y"])), (box[1], box[3]))
        np.testing.assert_allclose(self.meta["pattern_depth_range_ft"], [4600, 5150], atol=0.01)
        self.assertEqual(len(rows)-2, 122)

    def test_depth_regression(self):
        fit = self.meta["depth_fit"]
        self.assertLess(fit["rmse_ft"], 2)
        self.assertEqual(len(fit["labels"]), 19)
        self.assertAlmostEqual(fit["slope_ft_per_pt"], 1/3.6, places=6)
        residuals = [l["depth_ft"]-kgs.depth_at(fit, l["y_pt"]) for l in fit["labels"]]
        self.assertAlmostEqual(np.sqrt(np.mean(np.square(residuals))), fit["rmse_ft"], places=9)

    def test_clipping_prevents_wrong_bed_boundary(self):
        # The final shale tile at the first bed contact extends below 4610 ft.
        tiles = [i for i in self.images if i["xref"] == 43 and 2772 < i["bbox"][1] < 2774]
        self.assertEqual(len(tiles), 1)
        tile = tiles[0]
        self.assertGreater(tile["bbox"][3]-tile["visible_bbox"][3], 4)
        shale = self.meta["intervals_pt"][2]
        self.assertAlmostEqual(shale["bottom_pt"], tile["visible_bbox"][3], places=3)
        self.assertAlmostEqual(tile["bbox"][2]-tile["visible_bbox"][2], 7.2, places=3)

    def test_rerendered_column_mean_colours_match_embedded_swatches(self):
        scale = self.meta["render"]["px_per_pt"]
        entries = {e["xref"]: e for e in self.meta["legend_entries"] if e["kind"] == "rock"}
        errors, actual_means, expected_means = {}, {}, {}
        with Image.open(self.out / "image.png") as image:
            for ref, label in self.meta["rock_xrefs_in_column"].items():
                ref = int(ref)
                crops = []
                for tile in self.images:
                    if tile["xref"] != ref or tile["bbox"][1] < self.meta["track_bounds_pt"][1]:
                        continue
                    box, raw = fitz.Rect(tile["visible_bbox"]), fitz.Rect(tile["bbox"])
                    if abs(box.height-raw.height) > 0.01:
                        continue  # Compare complete vertical periods, not cut contacts.
                    a, b, c, d = kgs.pixel_box(box, scale)
                    crops.append(np.asarray(image.crop((a+2, b+1, c-2, d-1))).reshape(-1, 3))
                self.assertGreater(len(crops), 0)
                actual = np.concatenate(crops).mean(axis=0)
                with Image.open(self.out / entries[ref]["swatch_png"]) as swatch:
                    # The right sixth of the embedded tile is outside the track.
                    right = round(swatch.width*36/43.2)
                    expected = np.asarray(swatch)[1:-1, 2:right-2].reshape(-1, 3).mean(axis=0)
                errors[label] = np.abs(actual-expected).tolist()
                actual_means[label], expected_means[label] = actual, expected
                # Anti-alias phase, track strokes and accessory overlays affect
                # these real page crops. 15/255 per channel still distinguishes
                # every sample colour family; also require the nearest swatch.
                np.testing.assert_allclose(actual, expected, atol=15, rtol=0)
        for label, actual in actual_means.items():
            nearest = min(expected_means, key=lambda other: np.linalg.norm(actual-expected_means[other]))
            self.assertEqual(nearest, label)
        kgs.write_json(TEST_OUT / "sample_colour_errors.json", errors)

    def test_overlay_separate_and_alpha_preserved(self):
        accessories = self.meta["accessories"]
        self.assertEqual(len(accessories), 1)
        overlay = accessories[0]
        self.assertEqual(overlay["xref"], 132)
        self.assertEqual(overlay["smask_xref"], 129)
        self.assertTrue(overlay["over_column"])
        self.assertEqual(overlay["kind"], "unclassified_overlay")
        with Image.open(self.out / overlay["image_png"]) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getchannel("A").getextrema(), (0, 255))
        self.assertFalse(any(r["xref"] == 132 for r in self.meta["intervals_pt"]))

    def test_bbox_schema_and_dimensions(self):
        data = json.loads((self.out / "boxes.json").read_text())
        self.assertEqual(set(data), {"image_name", "width", "height", "bounding_boxes"})
        self.assertEqual(set(data["bounding_boxes"]),
                         {"depth_column", "interpreted_lithology_column", "lithology_legend"})
        with Image.open(self.out / "image.png") as image:
            self.assertEqual((data["width"], data["height"]), image.size)
            for x0, y0, x1, y1 in data["bounding_boxes"].values():
                self.assertTrue(0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height)


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        TEST_OUT.mkdir(parents=True, exist_ok=True)

    def test_mixed_widths_paint_order_and_blank_rows(self):
        rocks = {1: {"label": "A"}, 2: {"label": "B"}}
        tiles = [{"xref": 1, "visible_bbox": [0, 0, 6, 10], "order": 0},
                 {"xref": 2, "visible_bbox": [6, 0, 10, 10], "order": 1},
                 {"xref": 2, "visible_bbox": [0, 10, 10, 15], "order": 2},
                 {"xref": 1, "visible_bbox": [0, 10, 3, 15], "order": 3}]
        rows = kgs.truth_intervals([0, 0, 10, 20], tiles, rocks)
        self.assertEqual([r["lithology"] for r in rows], ["A", "B", "(blank)"])
        self.assertEqual([c["width_share"] for c in rows[0]["components"]], [0.6, 0.4])
        self.assertEqual([c["width_share"] for c in rows[1]["components"]], [0.3, 0.7])
        self.assertTrue(all(abs(sum(c["width_share"] for c in r["components"])-1) < 1e-9 for r in rows))

    def test_inline_identity_inheritance_swatches_and_overlay(self):
        path = TEST_OUT / "inline.pdf"
        with fitz.open() as doc:
            ref = miniature(doc)
            add_tile(doc, doc[0], fitz.Rect(160, 300, 166, 306), 101)
            miniature(doc, legend=False, ref=ref)
            for page in doc:
                inline_tiles(doc, page)
            sources = {}
            images = kgs.placed_images(doc[0], sources)
            rock = images[0]["xref"]
            self.assertLess(rock, 0)
            self.assertTrue(all(im["xref"] == rock for im in images[:-1]))
            self.assertNotEqual(rock, images[-1]["xref"])  # One byte/color step is distinct.
            self.assertEqual(kgs.placed_images(doc[0])[0]["xref"], rock)
            doc.save(path)
        pages = kgs.extract_pdf(path, TEST_OUT / "inline")
        self.assertEqual([p["status"] for p in pages], ["ok", "ok"], pages)
        self.assertTrue(pages[1]["meta"]["legend_inherited"])
        meta = pages[0]["meta"]
        self.assertEqual(meta["rock_xrefs_in_column"], {str(rock): "RockA"})
        with Image.open(Path(pages[0]["out"]) / meta["legend_entries"][0]["swatch_png"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (100, 100, 100))
        overlay = meta["accessories"][0]
        self.assertNotEqual(overlay["xref"], rock)
        with Image.open(Path(pages[0]["out"]) / overlay["image_png"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (101, 101, 101))

    def test_forms_preserve_log_truth_and_inline_sources(self):
        path = TEST_OUT / "forms.pdf"
        with fitz.open() as source, fitz.open() as doc:
            miniature(source)
            source[0].insert_text((150, 140), "Lithology", fontsize=8)
            source[0].insert_text((20, 680), "The lithology is uncertain", fontsize=8)
            inline_tiles(source, source[0])
            page = doc.new_page(width=360, height=840)
            page.show_pdf_page(page.rect, source, 0)
            doc.save(path)
        pages = kgs.extract_pdf(path, TEST_OUT / "forms")
        self.assertEqual(pages[0]["status"], "ok", pages)
        meta = pages[0]["meta"]
        np.testing.assert_allclose(meta["track_bounds_pt"], [180, 180, 223.2, 780], atol=0.002)
        self.assertTrue(180 < meta["lithology_header_x_pt"] < 223.2)
        self.assertEqual([r["lithology"] for r in meta["intervals_pt"]], ["(blank)", "RockA", "(blank)"])

    def test_form_matrix_resources_bbox_clip_and_state(self):
        with fitz.open() as source, fitz.open() as doc:
            p = source.new_page(width=100, height=100)
            add_tile(source, p, fitz.Rect(10, 20, 50, 60), 80)
            page = doc.new_page(width=300, height=300)
            page.show_pdf_page(fitz.Rect(50, 60, 250, 260), source, 0)
            # show_pdf_page creates nested forms with separate resources.
            full = next(ref for ref, name, _, _ in page.get_xobjects() if name == "fullpage")
            doc.xref_set_key(full, "BBox", "[20 50 40 70]")
            add_tile(doc, page, fitz.Rect(5, 5, 15, 15), 90)
            ims = kgs.placed_images(page)
            np.testing.assert_allclose(ims[0]["bbox"], [70, 100, 150, 180], atol=0.002)
            np.testing.assert_allclose(ims[0]["visible_bbox"], [90, 120, 130, 160], atol=0.002)
            np.testing.assert_allclose(ims[1]["visible_bbox"], [5, 5, 15, 15], atol=0.002)
            self.assertEqual([im["order"] for im in ims], [0, 1])

    def test_nonrectangular_and_sheared_form_clips_rejected(self):
        for prefix in (b"0 0 m 50 0 l 25 50 l h W n\n", b"1 0.1 0 1 0 0 cm\n"):
            with self.subTest(prefix=prefix), fitz.open() as source, fitz.open() as doc:
                p = source.new_page(width=100, height=100)
                add_tile(source, p, fitz.Rect(10, 20, 50, 60), 80)
                page = doc.new_page(width=100, height=100)
                page.show_pdf_page(page.rect, source, 0)
                full = next(ref for ref, name, _, _ in page.get_xobjects() if name == "fullpage")
                doc.update_stream(full, prefix + doc.xref_stream(full))
                with self.assertRaisesRegex(kgs.Skip, "nonrectangular clip or rotated/sheared"):
                    kgs.placed_images(page)

    def test_polygon_rectangle_clip_is_exact(self):
        with fitz.open() as doc:
            page = doc.new_page(width=100, height=100)
            add_tile(doc, page, fitz.Rect(10, 20, 50, 60), 80)
            ref = page.get_contents()[0]
            doc.update_stream(ref, b"20 50 m 40 50 l 40 70 l 20 70 l h W n\n" + doc.xref_stream(ref))
            np.testing.assert_allclose(kgs.placed_images(page)[0]["visible_bbox"], [20, 30, 40, 50])

    def test_inline_count_ignores_annotation_appearances(self):
        with fitz.open() as doc:
            miniature(doc)
            page = doc[0]
            inline_tiles(doc, page)
            count = len(kgs.placed_images(page))
            page.add_stamp_annot(fitz.Rect(20, 90, 90, 120))
            self.assertEqual(len(kgs.placed_images(page)), count)

    def test_aliases_preserve_full_legend_names(self):
        entries = [{"xref": -1, "kind": "rock", "label": name}
                   for name in ("Ls white", "Ls wh", "Ls white")]
        rock = kgs.legend_map(entries)[-1]
        self.assertEqual(rock["label_aliases"], ["Ls wh", "Ls white"])
        self.assertEqual(rock["label"], "Ls wh / Ls white")

    def test_invisible_ocr_does_not_supply_legend_labels(self):
        with fitz.open() as doc:
            miniature(doc)
            page = doc[0]
            page.insert_text((68, 48), "I Wrong OCR", fontsize=10, render_mode=3)
            legend = kgs.parse_legend(page, kgs.placed_images(page), kgs.border_lines(page))
            self.assertEqual([e["label"] for e in legend["entries"] if e["kind"] == "rock"], ["RockA"])

    def test_lithology_header_selects_interpreted_column(self):
        with fitz.open() as doc:
            miniature(doc)
            page = doc[0]
            ref = page.get_images()[0][0]
            for x in (220, 256):
                page.draw_line((x, 150), (x, 650))
            for y in range(150, 646, 8):
                add_tile(doc, page, fitz.Rect(220, y, 256, y+8), 100, ref)
            images = kgs.placed_images(page)
            borders = kgs.border_lines(page)
            legend = kgs.parse_legend(page, images, borders)
            bounds, _, _ = kgs.find_track(page, images, kgs.legend_map(legend["entries"]),
                                          borders, legend["sections"], track_x_hint=168)
            self.assertEqual(bounds, [150, 150, 186, 650])

    def test_depth_fit_with_three_printed_labels(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=700)
            for i in range(3):
                page.insert_text((110, 200+i*180), str(1000+50*i), fontsize=8, rotate=90)
            fit = kgs.fit_depth(page, [150, 150, 186, 650], [])
            self.assertEqual(len(fit["labels"]), 3)
            self.assertLess(fit["rmse_ft"], 0.001)

    def test_depth_track_calibrates_beyond_short_final_rock_column(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=700)
            for i in range(3):
                page.insert_text((110, 180+i*180), str(1000+50*i), fontsize=8, rotate=90)
            fit = kgs.fit_depth(page, [150, 0, 186, 210],
                                [(100, -1, 260), (100, 0, 612), (115, 0, 612)])
            self.assertEqual(len(fit["labels"]), 3)
            self.assertLess(fit["rmse_ft"], 0.001)

    def test_fractional_boundaries_preserve_subpixel_intervals(self):
        intervals = [{"top_pt": 0, "bottom_pt": 0.01, "lithology": "A"},
                     {"top_pt": 0.01, "bottom_pt": 2, "lithology": "A"}]
        rows = kgs.image_rows(intervals, 1, {"slope_ft_per_pt": 1, "intercept_ft": 0},
                              {1: {"label": "A"}}, subpixel=True)
        self.assertEqual(rows[0]["bottom_y"], rows[1]["top_y"])
        self.assertEqual(rows[0]["bottom_y"], 0.01)

    def test_cached_composite_requires_every_decoded_byte_to_match(self):
        path, bad = TEST_OUT / "cached.pdf", TEST_OUT / "cached_bad.pdf"
        with fitz.open() as doc:
            miniature(doc)
            page = doc[0]
            add_tile(doc, page, fitz.Rect(20, 60, 63.2, 68), 150)
            page.insert_text((70, 68), "RockB", fontsize=10)
            raw = bytes([100])*(16*8*3) + bytes([150])*(16*8*3) + bytes([100])*(16*8*3)
            pix = fitz.Pixmap(fitz.csRGB, 16, 24, raw, False)
            page.insert_image(fitz.Rect(150, 200, 186, 224), pixmap=pix, keep_proportion=False)
            doc.save(path)
            changed = bytes([101]) + raw[1:]
            pix = fitz.Pixmap(fitz.csRGB, 16, 24, changed, False)
            page.insert_image(fitz.Rect(150, 300, 186, 324), pixmap=pix, keep_proportion=False)
            doc.save(bad)
        pages = kgs.extract_pdf(path, TEST_OUT / "cached")
        self.assertEqual(pages[0]["status"], "ok", pages)
        meta = pages[0]["meta"]
        bed = next(r for r in meta["intervals_pt"] if r["lithology"] == "RockB")
        np.testing.assert_allclose([bed["top_pt"], bed["bottom_pt"]], [208, 216], atol=0.002)
        self.assertEqual(len(meta["exact_tile_expansions"][0]["segments"]), 3)
        rejected = kgs.extract_pdf(bad, TEST_OUT / "cached_bad")
        self.assertEqual(rejected[0]["status"], "skip")
        self.assertIn("unmatched cached rock image", rejected[0]["reason"])

    def test_nonlog_pages_skip_before_graphics_and_summary_is_short(self):
        path = TEST_OUT / "nonlogs.pdf"
        with fitz.open() as doc:
            for _ in range(12):
                doc.new_page().insert_text((20, 30), "Ordinary report")
            doc.save(path)
        with patch.object(kgs, "placed_images", side_effect=AssertionError("graphics parsed")):
            pages = kgs.extract_pdf(path, TEST_OUT / "nonlogs")
        self.assertTrue(all("no ROCK TYPES legend" in p["reason"] for p in pages))
        with patch.object(kgs, "extract_pdf", return_value=pages):
            result = kgs.batch(TEST_OUT, TEST_OUT / "short_summary", limit=1)
        self.assertEqual(result[0]["pages_skipped"], 12)
        self.assertEqual(result[0]["reason"], "Skip: no ROCK TYPES legend on this PDF (12)")

    def test_ties_and_merging_adjacent_tiles(self):
        rocks = {1: {"label": "A"}, 2: {"label": "B"}}
        tiles = [{"xref": 1, "visible_bbox": [0, 0, 5, 5], "order": 0},
                 {"xref": 2, "visible_bbox": [5, 0, 10, 5], "order": 1},
                 {"xref": 1, "visible_bbox": [0, 5.0001, 5, 10], "order": 2},
                 {"xref": 2, "visible_bbox": [5, 5, 10, 10], "order": 3}]
        rows = kgs.truth_intervals([0, 0, 10, 10], tiles, rocks)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lithology"], "A")

    def test_raster_only_skip(self):
        path = TEST_OUT / "raster.pdf"
        with fitz.open() as doc:
            page = doc.new_page()
            add_tile(doc, page, page.rect, 128)
            doc.save(path)
        pages = kgs.extract_pdf(path, TEST_OUT / "raster")
        self.assertEqual(pages[0]["status"], "skip")
        self.assertIn("no ROCK TYPES legend", pages[0]["reason"])

    def test_raster_track_with_vector_legend_skip(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=700)
            page.insert_text((110, 30), "ROCK TYPES", fontsize=10)
            add_tile(doc, page, fitz.Rect(20, 40, 63.2, 48), 100)
            page.insert_text((70, 48), "RockA", fontsize=10)
            page.insert_text((110, 80), "ACCESSORIES", fontsize=10)
            add_tile(doc, page, fitz.Rect(150, 150, 186, 650), 200)
            images = kgs.placed_images(page)
            legend = kgs.parse_legend(page, images, [])
            with self.assertRaisesRegex(kgs.Skip, "raster_only_or_no_placed"):
                kgs.find_track(page, images, kgs.legend_map(legend["entries"]), [], legend["sections"])

    def test_multiple_legends_use_geometric_word_locations(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=700)
            # Text insertion order intentionally differs from geometric order.
            page.insert_text((70, 168), "RockB", fontsize=10)
            page.insert_text((110, 150), "ROCK TYPES", fontsize=10)
            page.insert_text((70, 48), "RockA", fontsize=10)
            page.insert_text((110, 30), "ROCK TYPES", fontsize=10)
            add_tile(doc, page, fitz.Rect(20, 40, 63.2, 48), 100)
            add_tile(doc, page, fitz.Rect(20, 160, 63.2, 168), 150)
            page.insert_text((110, 200), "ACCESSORIES", fontsize=10)
            legend = kgs.parse_legend(page, kgs.placed_images(page), [])
            self.assertEqual(len(legend["sections"]), 2)
            self.assertEqual({e["label"] for e in legend["entries"]}, {"RockA", "RockB"})

    def test_multiple_pages_inherit_document_xrefs(self):
        path = TEST_OUT / "multipage.pdf"
        with fitz.open() as doc:
            ref = miniature(doc)
            miniature(doc, legend=False, ref=ref)
            doc.new_page()  # A page with no reusable lithology is skipped.
            doc.save(path)
        pages = kgs.extract_pdf(path, TEST_OUT / "multipage")
        self.assertEqual([p["status"] for p in pages], ["ok", "ok", "skip"], pages)
        self.assertFalse(pages[0]["meta"]["legend_inherited"])
        self.assertTrue(pages[1]["meta"]["legend_inherited"])
        self.assertTrue((TEST_OUT / "multipage/page_002/truth.csv").exists())

    def test_output_escape_rejected(self):
        for path in (kgs.ROOT / "external_untrusted", kgs.ROOT / "bakeoff", kgs.OUTPUT_ROOT / ".."):
            with self.assertRaises(ValueError):
                kgs.output_path(path)

    def test_legend_only_page_inherited_and_metric_depths(self):
        path = TEST_OUT / 'legend_only_metric.pdf'
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=700)
            page.insert_text((20, 20), 'Scale 1:240 Metric', fontsize=10)
            page.insert_text((110, 30), 'ROCK TYPES', fontsize=10)
            ref = add_tile(doc, page, fitz.Rect(20, 40, 63.2, 48), 100)
            page.insert_text((70, 48), 'RockA', fontsize=10)
            page.insert_text((110, 80), 'ACCESSORIES', fontsize=10)
            miniature(doc, legend=False, ref=ref)
            doc.save(path)
        pages = kgs.extract_pdf(path, TEST_OUT / 'legend_only_metric')
        self.assertEqual([p['status'] for p in pages], ['skip', 'ok'], pages)
        meta = pages[1]['meta']
        self.assertTrue(meta['legend_inherited'])
        self.assertEqual(meta['legend_entries'][0]['page'], 1)
        self.assertEqual(meta['depth_fit']['source_unit'], 'm')
        self.assertAlmostEqual(meta['depth_fit']['slope_ft_per_pt'], .5/.3048, places=5)

    def test_small_metric_depth_labels(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=700)
            for i in range(5):
                page.insert_text((110, 190+i*100), str(10+i*5), fontsize=8, rotate=90)
            fit = kgs.fit_depth(page, [150, 150, 186, 650], [], 'm')
            self.assertAlmostEqual(fit['slope_ft_per_pt'], .05/.3048, places=5)

    def test_clipped_numeric_suffix_cannot_break_depth_fit(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=600)
            for i in range(4):
                page.insert_text((110, 200+i*100), str(1000+i*50), fontsize=8, rotate=90)
            page.insert_text((110, 600), '200', fontsize=8, rotate=90)
            fit = kgs.fit_depth(page, [150, 150, 186, 600], [])
            self.assertEqual(len(fit['labels']), 4)
            self.assertAlmostEqual(fit['slope_ft_per_pt'], .5, places=5)

    def test_metric_partial_suffix_near_page_foot_is_ignored(self):
        with fitz.open() as doc:
            page = doc.new_page(width=300, height=600)
            for i in range(4):
                page.insert_text((110, 200+i*100), str(100+i*5), fontsize=8, rotate=90)
            page.insert_text((110, 593), '20', fontsize=8, rotate=90)
            fit = kgs.fit_depth(page, [150, 150, 186, 600], [], 'm')
            self.assertEqual(len(fit['labels']), 4)
            self.assertAlmostEqual(fit['slope_ft_per_pt'], .05/.3048, places=5)

    def test_batch_summary_one_row_and_contact_sheet(self):
        if not kgs.SAMPLE.is_file():
            self.skipTest('Optional KGS source PDF is not installed')
        result = kgs.batch(kgs.SAMPLE.parent, TEST_OUT / "batch", limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["status"], "ok")
        self.assertEqual(result[0]["intervals"], 124)
        self.assertTrue((TEST_OUT / "batch/contact_sheet.png").exists())

    def test_resolution_cannot_erase_an_interval(self):
        with self.assertRaisesRegex(kgs.Skip, "zero image rows"):
            kgs.image_rows([{"top_pt": 0, "bottom_pt": 0.01, "lithology": "A"}], 1,
                           {"slope_ft_per_pt": 1, "intercept_ft": 0}, {1: {"label": "A"}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
