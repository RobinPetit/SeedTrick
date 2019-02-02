#cython:language_level=3

import numpy as np
cimport numpy as np
from libc cimport math

from sklearn.metrics.pairwise import euclidean_distances

from seedtrick.kernels.base cimport Kernel

cdef class MultiInstanceKernel:
    cdef bint normalized
    cdef dict params
    def __init__(self, bint normalized):
        self.normalized = normalized
        self.params = dict()

    def __call__(self, x1, x2):
        cdef float ret = self._compute(x1, x2)
        if self.normalized:
            ret /= math.sqrt(self._compute(x1, x1) * self._compute(x2, x2))
        return ret

    def get_params(self):
        return self.params

    cdef float _compute(self, np.ndarray[np.float_t, ndim=2] x1, np.ndarray[np.float_t, ndim=2] x2):
        return -.11

MIK = MultiInstanceKernel

cdef class MinMaxKernel(MultiInstanceKernel):
    cdef float c, d
    def __init__(self, bint normalized, float c, float d):
        super().__init__(normalized)
        self.c = c
        self.d = d
        self.params['c'] = c
        self.params['d'] = d

    cdef float _compute(self, np.ndarray[np.float_t, ndim=2] x1, np.ndarray[np.float_t, ndim=2] x2):
        cdef np.ndarray[np.float_t, ndim=1] sx1, sx2
        cdef unsigned int N1 = x1.shape[1], N2 = x2.shape[1]
        assert N1 == N2
        cdef unsigned int i
        cdef float ret = np.dot(x1.min(axis=0), x2.min(axis=0))
        ret += np.dot(x1.max(axis=0), x2.max(axis=0))
        return (ret + self.c)**self.d

cdef class SetKernel(MultiInstanceKernel):
    cdef Kernel kernel
    def __init__(self, bint normalized, Kernel kernel):
        super().__init__(normalized)
        self.kernel = kernel

    cdef float _compute(self, np.ndarray X, np.ndarray Y):
        cdef float ret = 0
        cdef unsigned int i, j
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                ret += self.kernel(X[i], Y[j])
        return ret

cdef class RBFSetKernel(MultiInstanceKernel):
    cdef float gamma
    def __init__(self, bint normalized, float gamma):
        super().__init__(normalized)
        self.gamma = gamma
        self.params['gamma'] = gamma

    cdef float _compute(self, np.ndarray[np.float_t, ndim=2] X, np.ndarray[np.float_t, ndim=2] Y):
        return np.exp(-self.gamma*euclidean_distances(X, Y)).sum()

