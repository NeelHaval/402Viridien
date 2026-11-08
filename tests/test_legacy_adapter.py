from types import SimpleNamespace

import pytest

from viridian402.legacy import _ocr_engine


@pytest.mark.parametrize('attribute',['text_rec','text_recognizer'])
def test_rapidocr_batch_recognition_interface_compatibility(attribute):
    recognition=lambda crops:([('Shale',.9) for _ in crops],.01)
    def factory(**options):
        assert options['det_model_path'] is None
        return SimpleNamespace(**{attribute:recognition})
    engine=_ocr_engine(factory)
    assert engine.text_rec(['crop'])==([('Shale',.9)],.01)


def test_missing_recognition_interface_explains_failure():
    with pytest.raises(RuntimeError,match='batch recognition'):_ocr_engine(lambda **options:SimpleNamespace())
