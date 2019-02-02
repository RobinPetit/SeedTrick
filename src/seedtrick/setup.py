import os

from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup as np_setup

SUBPACKAGES = ['kernels', 'algo']

SRC_FILES = [
    # kernels
    (['kernels/base.c'], 'kernels.base'),
    (['kernels/string_kernels.c'], 'kernels.string_kernels'),
    (['kernels/svm_kernels.c'], 'kernels.svm_kernels'),
    (['kernels/basic.c'], 'kernels.basic'),
    (['kernels/mik.c'], 'kernels.mik'),
    # algo
    (['algo/suffixtree.c', 'algo/_suffixtree.c'], 'algo.suffixtree')
]

libraries = ["m"]
include_dirs = []

config = Configuration('seedtrick', '', '')
for subpackage in SUBPACKAGES:
    config.add_subpackage(subpackage)
for paths, ext_name in SRC_FILES:
    sources = list(map(lambda s: os.path.join('seedtrick', s), paths))
    print('@', ext_name, sources)
    config.add_extension(
        ext_name,
        sources=sources,
        include_dirs=include_dirs+[os.curdir],
        libraries=libraries,
    )

np_setup(**config.todict())
