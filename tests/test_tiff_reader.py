import numpy as np
from PIL import Image
import pytest

from viridian402.data import read_rgb


@pytest.mark.parametrize('kind',['rgb8','rgb16','white','gray16'])
def test_strip_reader_matches_pillow_color_semantics(tmp_path,kind):
    tifffile=pytest.importorskip('tifffile')
    rng=np.random.default_rng(19)
    rgb=kind.startswith('rgb');sixteen=kind.endswith('16')
    array=rng.integers(0,65536 if sixteen else 256,(15,11,3) if rgb else (15,11),dtype=np.uint16 if sixteen else np.uint8)
    path=tmp_path/'fixture.tif'
    tifffile.imwrite(path,array,photometric='rgb' if rgb else 'miniswhite' if kind=='white' else 'minisblack',rowsperstrip=3)
    expected=np.asarray(Image.open(path).crop((1,2,10,14)).convert('RGB'))
    assert np.array_equal(read_rgb(path,(1,2,10,14)),expected)


def test_strip_budget_applies_even_to_tiny_requested_crop(tmp_path):
    tifffile=pytest.importorskip('tifffile')
    path=tmp_path/'fixture.tif';tifffile.imwrite(path,np.zeros((10,1000,3),np.uint8),rowsperstrip=10)
    with pytest.raises(MemoryError,match='strip'):
        read_rgb(path,(0,0,1,1),max_decode_bytes=100)
