#cython:language_level=3

import numpy as np
cimport numpy as np

cdef class Kernel:
    def __init__(self):
        pass

    def __call__(self, X, Y):
        return np.nan
