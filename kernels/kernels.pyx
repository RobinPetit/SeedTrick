#cython:language_level=3

import numpy as np
cimport numpy as np

cdef class SVMKernel:
    cdef object kernel
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, X, Y):
        cdef unsigned int N1 = len(X)
        cdef unsigned int N2 = len(Y)
        cdef np.ndarray[np.float_t, ndim=2] ret = np.empty((N1, N2))
        cdef unsigned int i, j
        cdef np.float_t x
        for i in range(min(N1, N2)):
            ret[i,i] = self.kernel(X[i], Y[i])
        for i in range(N1):
            for j in range(i+1, N2):
                x = self.kernel(X[i], Y[j])
                ret[i, j] = x
                if j < N1:
                    ret[j, i] = x
        return ret
