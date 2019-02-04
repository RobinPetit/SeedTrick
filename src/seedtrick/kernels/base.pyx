#cython:language_level=3

import numpy as np
cimport numpy as np

cdef class Kernel:
    r'''
    Base class for kernel implementations. Kernel classes must be callable by overloading
    the ``__call__`` method.
    '''
    def __init__(self):
        pass

    def __call__(self, X, Y):
        return np.nan
