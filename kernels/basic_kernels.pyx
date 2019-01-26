#cython:language_level=3

import numpy as np
cimport numpy as np

from ._base cimport Kernel

cdef class RBFKernel(Kernel):
    cdef float gamma
    def __init__(self, float gamma):
        self.gamma = gamma

    def __call__(self, x1, x2):
        return np.exp(-self.gamma*((x1-x2)**2).sum())

cdef class PolyKernel(Kernel):
    cdef float c, d
    def __init__(self, float c, float d):
        self.c = c
        self.d = d

    def __call__(self, x1, x2):
        return (x1.dot(x2) + self.c)**self.d

cdef class CauchyKernel(Kernel):
    cdef float sigma
    def __init__(self, float sigma):
        self.sigma = sigma

    def __call__(self, x1, x2):
        cdef float ret = ((x1-x2)**2).sum() / self.sigma**2
        return 1 / (1 + ret)

cdef class MinKernel(Kernel):
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        return np.minimum(x1, x2).sum()

cdef class MaxKernel(Kernel):
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        return np.maximum(x1, x2).sum()

cdef class SigmoidKernel(Kernel):
    cdef float gamma, r
    def __init__(self, float gamma, float r):
        self.gamma = gamma
        self.r = r

    def __call__(self, x1, x2):
        return np.tanh(self.gamma*np.dot(x1, x2) + self.r)

cdef class NormalizedLinearKernel(Kernel):
    def __init__(self):
        pass

    def __call__(self, x1, x2):
        return np.dot(x1, x2) / np.sqrt(np.linalg.norm(x1)*np.linalg.norm(x2))


