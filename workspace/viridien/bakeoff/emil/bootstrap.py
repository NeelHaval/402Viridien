"""Load the working installed Arrow wheel without enabling general user-site imports."""
import importlib.util
from pathlib import Path
import site
import sys


def installed_arrow():
    if sys.platform != 'win32':
        return
    # Conda's Arrow DLL is broken on this machine; the already-installed user wheel
    # is the one ordinary Python/pytest uses. Keep -I and load only this package.
    folder = Path(site.getusersitepackages())/'pyarrow'
    if folder.is_dir() and 'pyarrow' not in sys.modules:
        spec = importlib.util.spec_from_file_location('pyarrow',folder/'__init__.py',submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['pyarrow'] = module
        spec.loader.exec_module(module)
