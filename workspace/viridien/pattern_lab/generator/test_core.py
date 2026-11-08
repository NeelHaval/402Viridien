"""Run from pattern_lab: python -X utf8 -m unittest generator.test_core -v."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from generator import create_dataset, generate
from generator._texture import minimum_cut


class PatternGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.tile = np.random.default_rng(42).integers(25, 245, (7, 11, 3), dtype=np.uint8)
        self.image = Image.fromarray(np.tile(self.tile, (7, 6, 1)))
        self.box = [0, 0, self.image.width, self.image.height]
        self.example = dict(id="layer/../../one", box=self.box, legend_ids=[1, 8],
                            document_id="test-document", approved=True,
                            provenance=dict(annotation_file="labels.json", layer_index=17,
                                            composition=[dict(legend_id=1, proportion=.6, x_range=[0, 40]),
                                                         dict(legend_id=8, proportion=.4, x_range=[40, 66])]))

    def test_same_input_and_seed_are_deterministic(self):
        first, first_meta = generate(self.image, self.box, seed=19)
        second, second_meta = generate(self.image, self.box, seed=19)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_meta, second_meta)

    def test_dimensions_and_exclusive_source_coordinates(self):
        result, meta = generate(self.image, [3, 2, 59, 44], width=127, height=259)
        self.assertEqual(result.size, (127, 259))
        self.assertEqual(result.mode, "RGB")
        self.assertEqual(meta["source_box"], [3, 2, 59, 44])
        self.assertEqual(meta["source_crop_size"], [56, 42])
        self.assertEqual(meta["scale"], [1., 1.])
        self.assertTrue(meta["limitations"])

    def test_exact_repeat_keeps_native_symbol_scale(self):
        result, meta = generate(self.image, self.box, width=180, height=480)
        pixels = np.asarray(result)
        self.assertEqual(meta["estimated_period"], dict(x=11, y=7))
        self.assertEqual(meta["method"], "native-repeat")
        np.testing.assert_array_equal(pixels[:-7], pixels[7:])
        np.testing.assert_array_equal(pixels[:, :-11], pixels[:, 11:])
        source_palette = {tuple(value) for value in self.tile.reshape(-1, 3)}
        self.assertTrue(all(tuple(value) in source_palette for value in pixels.reshape(-1, 3)))

    def test_seed_changes_phase_for_nonuniform_source(self):
        first, _ = generate(self.image, self.box, seed=1)
        second, _ = generate(self.image, self.box, seed=2)
        self.assertNotEqual(first.tobytes(), second.tobytes())

    def test_quilting_is_deterministic_and_copies_source_colours(self):
        pixels = np.random.default_rng(10).integers(20, 250, (29, 31, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        first, meta = generate(image, [0, 0, 31, 29], width=85, height=97, seed=4, variation=.6)
        second, other = generate(image, [0, 0, 31, 29], width=85, height=97, seed=4, variation=.6)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(meta, other)
        self.assertEqual(meta["method"], "exemplar-quilting")
        self.assertGreater(meta["passes"]["vertical"]["seams"], 0)
        self.assertGreater(meta["passes"]["horizontal"]["seams"], 0)
        source_palette = {tuple(value) for value in pixels.reshape(-1, 3)}
        self.assertTrue(all(tuple(value) in source_palette for value in np.asarray(first).reshape(-1, 3)))

    def test_constant_and_tiny_sources(self):
        for size in ((1, 1), (2, 3), (18, 20)):
            with self.subTest(size=size):
                image = Image.new("RGB", size, (202, 80, 155))
                result, meta = generate(image, [0, 0, *size], width=19, height=21)
                self.assertEqual(result.size, (19, 21))
                self.assertEqual(result.getpixel((18, 20)), (202, 80, 155))
                self.assertEqual(meta["estimated_period"], dict(x=1, y=1))

    def test_single_pixel_outputs_and_palette_sources(self):
        source = self.image.convert("P")
        output, _ = generate(source, self.box, width=1, height=1)
        self.assertEqual(output.size, (1, 1))
        self.assertEqual(output.mode, "RGB")

    def test_frame_removal_records_exact_interior(self):
        pixels = np.tile(self.tile, (7, 6, 1))
        framed = np.pad(pixels, ((1, 1), (1, 1), (0, 0)), constant_values=0)
        source = Image.fromarray(framed)
        _, meta = generate(source, [0, 0, source.width, source.height])
        self.assertEqual(meta["frame_trim"], [1, 1, 1, 1])
        self.assertEqual(meta["texture_box"], [1, 1, source.width - 1, source.height - 1])

    def test_minimum_cut_uses_connected_low_error_path(self):
        error = np.full((4, 6), 50.)
        desired = np.array([0, 1, 2, 2, 1, 0])
        error[desired, np.arange(6)] = 0
        np.testing.assert_array_equal(minimum_cut(error), desired)

    def test_invalid_generation_inputs_fail(self):
        bad = [dict(box=[-1, 0, 2, 2]), dict(box=[0, 0, 100, 2]),
               dict(box=[1, 1, 1, 2]), dict(box=[0, 0, 2.5, 2]),
               dict(width=0), dict(width=True), dict(height=1.2),
               dict(seed=-1), dict(variation=-.1), dict(variation=1.1),
               dict(variation=float("nan")), dict(variation=True),
               dict(width=16000, height=16000)]
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                options = dict(options)
                box = options.pop("box", self.box)
                generate(self.image, box, **options)

    def test_approved_dataset_has_exact_provenance_and_labels(self):
        original = deepcopy(self.example)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "export"
            manifest = create_dataset(self.image, [self.example], output, count_per_example=2, seed=73)
            saved = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest, saved)
            self.assertEqual(manifest["examples"], [original])
            self.assertEqual(manifest["sample_count"], 2)
            self.assertEqual(len(list(output.glob("*.png"))), 2)
            for index, sample in enumerate(manifest["samples"]):
                self.assertEqual(sample["source_box"], self.box)
                self.assertEqual(sample["legend_ids"], [1, 8])
                self.assertEqual(sample["provenance"], original["provenance"])
                self.assertEqual(sample["output_box"], [0, 0, 180, 480])
                self.assertEqual(sample["annotations"], [dict(box=[0, 0, 180, 480], legend_ids=[1, 8])])
                self.assertEqual(sample["generation"]["seed"], 73 + index)
                self.assertTrue(sample["source_approved"])
                self.assertIsNone(sample["generation_review"])
                self.assertEqual(Path(sample["file"]).name, sample["file"])
                with Image.open(output / sample["file"]) as image:
                    self.assertEqual(image.size, (180, 480))
            self.assertEqual(self.example, original)

    def test_any_unapproved_example_rejects_the_whole_export_without_writes(self):
        for approval in (False, None, "true", 1):
            with self.subTest(approved=approval), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "not-created"
                unapproved = dict(self.example, id="unapproved", approved=approval)
                with self.assertRaisesRegex(ValueError, "not explicitly approved"):
                    create_dataset(self.image, [self.example, unapproved], output)
                self.assertFalse(output.exists())

    def test_missing_approval_or_invalid_later_box_does_not_write(self):
        for bad in ({key: value for key, value in self.example.items() if key != "approved"},
                    dict(self.example, id="other", box=[0, 0, 9999, 2])):
            with tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "not-created"
                with self.assertRaises(ValueError):
                    create_dataset(self.image, [self.example, bad], output)
                self.assertFalse(output.exists())

    def test_dataset_is_deterministic_and_does_not_overwrite_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "a", Path(folder) / "b"
            first = create_dataset(self.image, [self.example], a, count_per_example=1, seed=6)
            second = create_dataset(self.image, [self.example], b, count_per_example=1, seed=6)
            self.assertEqual(first, second)
            self.assertEqual((a / "manifest.json").read_bytes(), (b / "manifest.json").read_bytes())
            self.assertEqual((a / first["samples"][0]["file"]).read_bytes(),
                             (b / second["samples"][0]["file"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "new or empty"):
                create_dataset(self.image, [self.example], a, count_per_example=1)


if __name__ == "__main__":
    unittest.main()
