#cython:language_level=3
from libc cimport math
cimport numpy as np
import numpy as np

cdef class StringSubsequenceKernel:
    r'''
    Implementation of the String Subsequence Kernel (SSK) defined in [1].

    Definition
    ----------

    Let :math:`\Sigma` be an alphabet. For :math:`n \geq 0`, let :math:`\Sigma^n` be the
    set of all finite strings of length :math:`n` on :math:`\Sigma`.

    The SSK is defined as follows:

    .. math::
        \mathsf{SSK}_N : \Sigma^* \times \Sigma^* \to \mathbb R^+ :
        (s, t) \mapsto \sum_{u \in \Sigma^N}\sum_{\mathbf i : u = s[\mathbf i]}\sum_{\mathbf j : u = t[\mathbf j]}\lambda^{\ell(\mathbf i) + \ell(\mathbf j)}

    References
    ----------

    [1] Lodhi, H., Saunders, C., Shawe-Taylor, J., Cristianini, N., & Watkins, C. (2002).
    Text classification using string kernels. Journal of Machine Learning Research, 2(Feb), 419-444.
    https://papers.nips.cc/paper/1869-text-classification-using-string-kernels.pdf
    '''

    cdef unsigned int N
    cdef float lamb
    cdef bint normalized
    cdef np.ndarray K_prime

    def __init__(self, unsigned int N, float lambda_=.9, bint normalized=True):
        self.N = N
        self.lamb = lambda_
        self.normalized = normalized

    def __call__(self, str x1, str x2):
        cdef const unsigned char[:] s = x1.encode(), t = x2.encode()
        cdef double ret = self._kernel(s, t)
        if self.normalized:
            ret /= math.sqrt(self._kernel(s, s) * self._kernel(t, t))
        return ret

    cdef double _kernel(self, const unsigned char [:]s, const unsigned char[:] t):
        buff = self._precompute_arrays(s, t)
        return self._K_n(s, t)

    cdef double _K_n(self, const unsigned char [:]s, const unsigned char[:] t):
        cdef double ret = 0
        if min(len(s), len(t)) < self.N:
            return 0
        for k in range(len(t)):
            if t[k] == s[-1]:
                ret += self.K_prime[self.N-1, len(s)-2, k-1]
        ret *= self.lamb*self.lamb
        x = self._K_n(s[:-1], t)
        ret += x
        return ret

    cdef void _precompute_arrays(self, const unsigned char[:] s, const unsigned char[:] t):
        self.K_prime = np.empty((self.N, len(s), len(t)))
        cdef unsigned int i, j, k
        cdef int m, n
        cdef double K_second, v
        # K_second is a temp var to compute K_i''(s[:j], t[:k])
        for i in range(self.N):
            for len_s in range(len(s)):
                for len_t in range(len(t)):
                    if i == 0:
                        self.K_prime[i, len_s, len_t] = 1
                    elif len_s+1 < i or len_t+1 < i:
                        self.K_prime[i, len_s, len_s] = 0
                    else:
                        K_second = 0
                        for j in range(len_t+1):
                            if s[len_s] == t[j]:
                                m, n = len_s-1, j-1
                                if m < 0 or n < 0:
                                    v = 1
                                else:
                                    v = self.K_prime[i-1, m, n]
                                K_second += v * math.pow(self.lamb, len_t-j+2)
                        self.K_prime[i, len_s, len_t] = K_second
                        if len_s > 0:
                            self.K_prime[i, len_s, len_t] += self.lamb * self.K_prime[i, len_s-1, len_t]

cdef class StringSubsequenceKernelDP(StringSubsequenceKernel):
    def __init__(self, unsigned int N, float lambda_=.9, bint normalized=True):
        StringSubsequenceKernel.__init__(self, N, lambda_, normalized)

    cdef double _k_prime(self, unsigned int i, const unsigned char[:] s, const unsigned char[:] t):
        cdef const unsigned char[:] s_minus1
        cdef char x
        cdef double ret = 0
        cdef int j
        if i == 0:
            return 1
        elif min(len(s), len(t)) < i:
            return 0
        else:
            x = s[-1]
            s_minus1 = s[:-1]
            for j in range(len(t)):
                if t[j] == x:
                    ret += self._k_prime(i-1, s_minus1, t[:j])*math.pow(self.lamb, len(t)-j+1)
            ret += self.lamb * self._k_prime(i, s_minus1, t)
            return ret

    cdef double _kernel(self, const unsigned char[:] s, const unsigned char[:] t):
        cdef unsigned char x
        cdef const unsigned char[:] s_minus1
        cdef double ret = 0
        cdef unsigned int j
        if min(len(s), len(t)) < self.N:
            return 0
        x = s[-1]
        s_minus1 = s[:-1]
        for j in range(len(t)):
            if t[j] == x:
                ret += self._k_prime(self.N-1, s_minus1, t[:j])
        ret *= self.lamb*self.lamb
        ret += self._kernel(s_minus1, t)
        return ret

