#cython:language_level=3

import numpy as np
cimport numpy as np

from ._base cimport Kernel

from sklearn.metrics.pairwise import euclidean_distances

cdef class MultiInstanceKernel:
    pass

cdef class MinMaxKernel(MultiInstanceKernel):
    cdef float c, d
    def __init__(self, float c, float d):
        self.c = c
        self.d = d

    def __call__(self, x1, x2):
        return self._compute(x1, x2)

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
    def __init__(self, Kernel kernel):
        self.kernel = kernel

    def __call__(self, X, Y):
        return self._compute(X, Y)

    cdef float _compute(self, X, Y):
        cdef float ret = 0
        cdef unsigned int i, j
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                ret += self.kernel(X[i], Y[j])
        return ret

cdef class RBFSetKernel(MultiInstanceKernel):
    cdef float gamma
    def __init__(self, float gamma):
        self.gamma = gamma

    def __call__(self, X, Y):
        return np.exp(-self.gamma*euclidean_distances(X, Y)).sum()
