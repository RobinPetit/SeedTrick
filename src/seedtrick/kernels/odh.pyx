#cython:language_level=3

import numpy as np
cimport numpy as np

from libc.string cimport strlen, strcpy
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport pow, sqrt

from seedtrick.kernels.base cimport Kernel

# Need a sparse matrix to store the matrix of projections X = [\Phi(S_i)]_i
# that is only 1 in 20^(2k) dense
from scipy.sparse import csr_matrix
SparseMatrix = csr_matrix

cdef object _compute_X(const char **strings, unsigned int nb_strings, unsigned int K):
    cdef unsigned int L_max = 0
    cdef unsigned int i, j, k, d
    cdef unsigned int N
    cdef unsigned int D, M = <unsigned int>pow(20, K)
    cdef unsigned int latent_dimensions
    for i in range(nb_strings):
        N = strlen(strings[i])
        if N > L_max:
            L_max = N
    D = L_max - K + 1
    latent_dimensions = D*M*M
    counts = SparseMatrix((nb_strings, latent_dimensions), dtype=np.float)
    for k in range(nb_strings):
        for i in range(strlen(strings[k])-K+1):
            for j in range(i, strlen(strings[k])-K+1):
                counts[k, _kmer_to_idx(&strings[k][i], K)*M*D + _kmer_to_idx(&strings[k][j], K)*D + i-j] += 1
    return counts

cdef void _normalize_array_XX(np.float_t[:,:] K, unsigned int N):
    cdef unsigned int i, j
    cdef np.float_t *norms = <np.float_t *>malloc(N * sizeof(np.float_t))
    for i in range(N):
        norms[i] = <np.float_t>sqrt(K[i,i])
    for i in range(N):
        for j in range(N):
            K[i,j] / (norms[i]*norms[j])
    free(norms)

cdef object _get_count_matrix(list X, unsigned int K):
    cdef unsigned int N_X = len(X)
    cdef char **strings_X = <char **>malloc(N_X * sizeof(char *))
    for i in range(N_X):
        strings_X[i] = <char *>malloc((len(X[i])+1) * sizeof(char))
        strcpy(strings_X[i], X[i].encode('ASCII'))
    ret = _compute_X(strings_X, N_X, K)
    for i in range(N_X):
        free(strings_X[i])
    free(strings_X)
    return ret

cdef void _normalize_rows(counts):
    #assert isinstance(counts, csr_matrix)
    cdef unsigned int N = counts.shape[0]
    coo = counts.tocoo()
    norms = np.zeros(N, dtype=np.float)
    for (i, j, count) in zip(coo.row, coo.col, coo.data):
        norms[i] += count*count
    for (i, j) in zip(coo.row, coo.col):
        counts[i, j] /= norms[i]

cdef class ODHKernel(Kernel):
    r'''
    Implementation of the Oligomer Distance Histogram (ODH) Kernel from [1].

    **Definition**

    Let :math:`\Sigma` be an alphabet. For :math:`K > 0`, for :math:`\mathcal D \subset \Sigma^*`, let
    :math:`L_{\max} := \max_{u \in \mathcal D}|u|`, :math:`\Xi := \Sigma^K` the set of :math:`K`-mers
    (ordered :math:`m_i, i=1, \ldots, M` for :math:`M := |\Xi| = |\Sigma|^K`) and :math:`D := L_{\max}-K`.

    .. math::
        \Phi : \Sigma^* \to \Sigma^{(D+1)M^2} : s \mapsto [h_{ij}^d(s)]_{(i, j, d)=(1, 1, 0)}^{(M, M, D)}

    where :math:`h_{ij}^d(s)` is the number of couples :math:`(\alpha, \beta), \alpha \leq \beta` such that:

    .. math::
        (i)   \quad &s_{\alpha \ldots \alpha+K} &= m_i, \\
        (ii)  \quad &s_{\beta \ldots \beta+K} &= m_j, \\
        (iii) \quad &\beta-\alpha &= d.

    The kernel is then given by:

    .. math::
        \kappa_K : \Sigma^* \times \Sigma^* \to \mathbb R : (u, v) \mapsto \langle \Phi(u), \Phi(v) \rangle

    References:
        [1] Lingner, T., & Meinicke, P. (2006).
        Remote homology detection based on oligomer distances. Bioinformatics, 22(18), 2224-2231.
    '''
    cdef unsigned int k
    cdef bint normalized
    def __init__(self, unsigned int k, bint normalized):
        self.k = k
        self.normalized = normalized

    def __call__(self, X, Y):
        pass

    def get_K_matrix(self, X, Y):
        r'''
        Get the matrix :math:`\mathbf K` where :math:`K_{ij} = \langle \Phi(X_i), \Phi(Y_j) \rangle`.
        '''
        cdef bint Y_is_None = Y is None or X is Y
        counts_X = _get_count_matrix(X, self.k)
        if Y is None:
            ret = counts_X * counts_X.T
            if self.normalized:
                _normalize_array_XX(ret, len(X))
        else:
            counts_Y = _get_count_matrix(Y, self.k)
            if self.normalized:
                _normalize_rows(counts_X)
                _normalize_rows(counts_Y)
            ret = counts_X * counts_Y.T
        return ret

