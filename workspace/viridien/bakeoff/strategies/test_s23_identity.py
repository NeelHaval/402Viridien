"""CPU-only identity, ranking, and run-wiring checks for S23."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from strategies import scoreboard_v4 as base
from strategies import s21_v4dino as dino
from strategies import s23_v4dino_identity as strategy


class FakeClassifier:
    P = {'fill_dark': 140, 'gamma': 1.0, 'unknown': 99.0}
    def __init__(self):
        self.distances = {'Clay': 0.0, 'Shale': 1.0, 'Clay/Shale': 2.0}
    def strip_borders(self, a): return a
    def features(self, a): return {'mask': np.ones((4, 4))}
    def twins(self, log): return {'Clay': 'Clay / Shale', 'Shale': 'Clay / Shale'}
    def log_scale(self, log): return 1.0
    def legend(self, log, scale):
        return [(name, {'d': value, 'mask': np.ones((4, 4))})
                for name, value in self.distances.items()]
    def distance(self, f, s, alpha): return s['d']
    def ncc(self, a, b): return 1.0


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.classifier = strategy.preserve_caption_identity(FakeClassifier())
        self.ink = np.zeros((10, 10, 3), dtype=np.uint8)
        self.names = ['Clay', 'Shale', 'Shale', 'Clay/Shale']
        self.vector = np.array([1., 0.])

    def classify(self, similarities):
        refs = np.array([[value, 0.] for value in similarities])
        return dino._classify(self.classifier, self.ink, 'sample', self.vector,
                              self.names, refs, check=False)

    def test_twin_groups_disabled_and_genuine_slash_caption_preserved(self):
        self.assertEqual(self.classifier.twins('sample'), {})
        self.classifier.distances = dict.fromkeys(self.classifier.distances, 0.0)
        for target, sims in [('Clay', [1, 0, 0, 0]),
                             ('Shale', [0, 0, 1, 0]),
                             ('Clay/Shale', [0, 0, 0, 1])]:
            with self.subTest(target=target):
                self.assertEqual(self.classify(sims)['label'], target)

    def test_hand_and_similarity_remain_separate_inputs(self):
        self.assertEqual(self.classify([0, 0, 0, 0])['label'], 'Clay')
        self.classifier.distances = {'Clay': 2., 'Shale': 1., 'Clay/Shale': 0.}
        self.assertEqual(self.classify([0, 0, 0, 0])['label'], 'Clay/Shale')
        self.classifier.distances = dict.fromkeys(self.classifier.distances, 0.0)
        self.assertEqual(self.classify([0, 0, 1, 0])['label'], 'Shale')
        self.assertEqual(self.classify([1, 0, 0, 0])['label'], 'Clay')

    def test_actual_log06_native_captions_are_separate(self):
        name = next(n for n in base._native() if 'd-582' in n.lower() and 'log06' in n.lower())
        classifier = base.make_classifier(SimpleNamespace(name=name), {})
        self.assertEqual(classifier.twins(name)['Clay'], 'Clay / Shale')
        self.assertEqual(classifier.twins(name)['Shale'], 'Clay / Shale')
        strategy.preserve_caption_identity(classifier)
        self.assertEqual(classifier.twins(name), {})
        labels, images = dino._swatches(classifier, name)
        self.assertIn('Clay', labels)
        self.assertIn('Shale', labels)
        self.assertEqual(labels.count('Clay'), 3)
        self.assertEqual(labels.count('Shale'), 3)
        self.assertEqual(len(labels), len(images))

    def test_run_injects_identity_before_both_passes(self):
        classifier = FakeClassifier()
        result = {'beds': []}
        seen = []
        answers = []
        def segment(log, ctx, classifier, window):
            seen.append(dict(classifier.twins(log.name)))
            answers.append([classifier.classify(self.ink, log.name, check=False),
                            classifier.classify(self.ink, log.name, check=False)])
            return result
        swatch_images = [self.ink] * 4
        vectors = np.array([[1., 0.], [0., 1.], [0., 1.], [-1., 0.],
                            [0., 1.], [1., 0.]])
        with patch.object(base, 'make_classifier', return_value=classifier),              patch.object(base, 'segment', side_effect=segment),              patch.object(base, 'diagnostics', return_value={}),              patch.object(base, 'to_rows', return_value=['row']),              patch.object(dino, '_swatches', return_value=(self.names, swatch_images)),              patch.object(dino, 'embed', return_value=vectors),              patch.object(dino, '_model', return_value=(None, None, None, {'id': 'fake'})):
            classifier.distances = dict.fromkeys(classifier.distances, 0.)
            rows = strategy.run(SimpleNamespace(name='sample'), (0, 10), {'calib': None})
        self.assertEqual(rows, ['row'])
        self.assertEqual(seen, [{}, {}])
        self.assertEqual([answer['label'] for answer in answers[1]], ['Shale', 'Clay'])
        self.assertEqual(strategy.run.last_diagnostics['dino_chunk_vectors'], 2)
        self.assertEqual(strategy.run.last_diagnostics['legend_identity'], 'exact printed caption')

if __name__ == '__main__':
    unittest.main(verbosity=2)
