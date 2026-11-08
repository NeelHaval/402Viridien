"""Small checks for the comparison gate and paired-log confidence intervals."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).with_name('build_comparison.py')
SPEC = importlib.util.spec_from_file_location('jev_comparison_build', PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class ComparisonTests(unittest.TestCase):
    def test_bootstrap_uses_log_means_and_fixed_seed(self):
        groups = {'log_a': [0.0, 1.0], 'log_b': [0.5]}
        result = builder.bootstrap(groups)
        self.assertEqual(result, builder.bootstrap(groups))
        self.assertAlmostEqual(result['delta'], .5)
        self.assertEqual((result['low'], result['high'], result['logs']), (.5, .5, 2))
        self.assertIn('no useful precision', builder.bootstrap({'one': [1.]})['note'])

    def test_rejects_final_while_cases_are_missing(self):
        own = builder.load(builder.JEV / 'results.json')['records']
        if len(own) >= 68:
            self.skipTest('all Jev cases are now complete')
        with self.assertRaisesRegex(ValueError, 'Final gate not met'):
            builder.build(require_complete=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
