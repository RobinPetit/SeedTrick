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
        cdef np.ndarray[np.float_t, ndim=2] ret = np.empty((N1, N2), dtype=np.float)
        cdef unsigned int i, j
        cdef np.float_t x
        if X is Y:
            for i in range(min(N1, N2)):
                a = X[i]
                b = Y[i]
                K = self.kernel(X[i], Y[i])
                ret[i,i] = self.kernel(X[i], Y[i])
            for i in range(N1):
                for j in range(i+1, N2):
                    x = self.kernel(X[i], Y[j])
                    ret[i, j] = x
                    if j < N1 and j < N2:
                        ret[j, i] = x
        else:
            for i in range(N1):
                for j in range(N2):
                    ret[i,j] = self.kernel(X[i], Y[j])
        return ret

    def get_kernel(self):
        return self.kernel
