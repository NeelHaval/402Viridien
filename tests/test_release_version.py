import importlib.util
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('release_version',Path(__file__).parents[1]/'scripts/check_version.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_release_tag_must_match_built_package(tmp_path):
    project=tmp_path/'pyproject.toml';project.write_text('[project]\nversion="0.2.0"\n')
    assert module.check_version('v0.2.0',project)=='0.2.0'
    with pytest.raises(ValueError,match='does not match'):module.check_version('v9.0.0',project)
