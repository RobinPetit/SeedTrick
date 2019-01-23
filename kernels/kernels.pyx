#cython:language_level=3

from abc import abstractmethod
import numpy as np
cimport numpy as np

cdef class SVMKernel:
    cdef object kernel
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, Y):
        cdef unsigned int N1 = X.shape[0]
        cdef unsigned int N2 = Y.shape[0]
        cdef np.ndarray[np.float_t, ndim=2] ret = np.empty((N1, N2))
        cdef unsigned int i, j
        for i in range(N1):
            ret[i,i] = self.kernel(X[i], X[i])
            for j in range(i+1, N2):
                ret[j, i] = ret[i,j] = self.kernel(X[i], Y[j])
        return ret
