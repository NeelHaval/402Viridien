"""Deterministic S21 fusion and native-canvas tests without GPU work."""
import numpy as np
from strategies import s21_v4dino as strategy


class Fake:
    P = {'fill_dark': 140, 'gamma': 1.0, 'unknown': 99.0}
    def strip_borders(self, a): return a
    def features(self, a): return {'mask': np.ones((4, 4))}
    def twins(self, log): return {}
    def log_scale(self, log): return 1.0
    def legend(self, log, scale):
        return [(name, {'d': distance, 'mask': np.ones((4, 4))})
                for name, distance in [('A', 0.0), ('B', 0.5), ('C', 2.0)]]
    def distance(self, f, s, alpha): return s['d']
    def ncc(self, a, b): return 1.0


def main():
    classifier = Fake()
    ink = np.zeros((10, 10, 3), dtype=np.uint8)
    vector = np.array([1., 0.])
    references = np.array([[0., 1.], [1., 0.], [0., 1.]])
    result = strategy._classify(classifier, ink, 'x', vector, ['A', 'B', 'C'], references, check=False)
    assert result['state'] == 'ok' and result['label'] == 'B', result
    assert result['margin'] >= 0
    result = strategy._classify(classifier, ink, 'x', vector, ['A', 'B', 'C'], references, unknown=-1, check=False)
    assert result['state'] == 'unknown'
    blank = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert strategy._classify(classifier, blank, 'x', vector, ['A', 'B', 'C'], references, check=False)['state'] == 'blank'
    tiled = strategy.canvas(np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))
    assert tuple(tiled[1, 2]) == (1, 2, 3) and tuple(tiled[1, 3]) == (4, 5, 6)
    print('PASS: fusion ranking, rejection, blank state, and native tiling')

if __name__ == '__main__':
    main()
