from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup

#TODO: make a proper setup file (compiling Cython source code and
#another in `seedtrick/` making a whole package out of it)

from glob import glob
import os.path

SUBMODULE_DIRS = ['kernels/', 'algo/', 'svm/']

def configuration(parent_package='', top_path=None):
    config = Configuration(None, parent_package, top_path)
    config.set_options(
        ignore_setup_xxx_py=True,
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True
    )
    config.add_subpackage('seedtrick')
    return config

def find_pyx():
    pyx_files = list()
    for dir_path in SUBMODULE_DIRS:
        pyx_files.extend(sorted(glob(os.path.join('seedtrick', dir_path, '*.pyx'))))
    return pyx_files

extensions = list()
for file_path in find_pyx():
    ext_name = os.path.splitext(file_path.replace('/', '.'))[0]
    print('#', ext_name, file_path)
    extensions.append(
        Extension(
            ext_name,
            [file_path],
            language='c',
        )
    )

setup_args = {
    'name' : 'SeedTrick',
    'version' : '0.0.1',
    'description' : 'Implementation of some (multiple) instance kernels',
    'author' : 'Robin Petit',
    'configuration' : configuration
}

extensions = cythonize(
    extensions,
    compiler_directives={'embedsignature': True}
)
np_setup(**setup_args)
