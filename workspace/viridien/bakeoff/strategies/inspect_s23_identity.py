"""Inspect the frozen D-582 log06 candidate labels without reading truth."""
from collections import Counter
from types import SimpleNamespace
from strategies import scoreboard_v4 as adapter


def main():
    cache = adapter._native()
    names = [n for n in cache if 'd-582' in n.lower() and 'log06' in n.lower()]
    assert len(names) == 1, names
    name = names[0]
    classifier = adapter.make_classifier(SimpleNamespace(name=name), {})
    pairs = [(label, a.shape) for label, a in cache[name] if label.lower() in ('clay', 'shale', 'clay/shale', 'clay / shale')]
    twins = classifier.twins(name)
    print('log:', name)
    print('native swatches:', pairs)
    print('twin mapping:', {label: twins.get(label, label) for label, _ in pairs})
    assert Counter(label.lower() for label, _ in pairs)['clay'] >= 1
    assert Counter(label.lower() for label, _ in pairs)['shale'] >= 1
    assert twins.get('Clay') == twins.get('Shale') == 'Clay / Shale'
    print('PASS: separate native Clay and Shale swatches collapse into a twin group')

if __name__ == '__main__':
    main()
