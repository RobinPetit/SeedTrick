#cython:language_level=3

import numpy as np
cimport numpy as np

from libc.string cimport strlen, strncpy, strcpy
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libc.math cimport pow, sqrt

from seedtrick.kernels.base cimport Kernel

# Need a sparse matrix to store the matrix of projections X = [\Phi(S_i)]_i
# that is only 1 in 20^(2k) dense
####from scipy.sparse import csr_matrix, lil_matrix
####SparseMatrix = lil_matrix
from seedtrick.algo.sparse import SparseMatrix
from seedtrick.algo.sparse cimport cdot_sparse_matrices

cdef SparseMatrix _compute_X(const char **strings, unsigned int nb_strings, unsigned int K, int max_dist, kmer2idx_t kmer2idx):
    cdef unsigned int L_max = 0
    cdef unsigned int i, j, k, d
    cdef unsigned int N
    cdef unsigned int D
    cdef unsigned int M = <unsigned int>pow(20, K) if kmer2idx == _kmer_to_idx_aa else <unsigned int>pow(4, K)
    cdef unsigned int len_k, j_max

    for i in range(nb_strings):
        N = strlen(strings[i])
        if N > L_max:
            L_max = N
    if max_dist < 0:
        D = L_max - K + 1
    else:
        D = <unsigned int>max_dist
    nb_non_zero_entries = D*(L_max - K + 1) - (D*(D-1)) // 2
    counts = SparseMatrix((nb_strings, nb_non_zero_entries))
    for k in range(nb_strings):
        print(k)
        len_k = strlen(strings[k])
        for i in range(len_k-K+1):
            j_max = i+D if i+D < len_k-K+1 else len_k-K+1
            for j in range(i, j_max):
                if j-i >= D:
                    break
                counts[k, kmer2idx(&strings[k][i], K)*M*D + kmer2idx(&strings[k][j], K)*D + j-i] += 1
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

cdef SparseMatrix _get_count_matrix(list X, unsigned int K, int max_dist, bint aa):
    cdef unsigned int N_X = len([x for x in X if len(x) >= K])
    cdef kmer2idx_t kmer2idx =  &_kmer_to_idx_aa if aa else &_kmer_to_idx_nt
    cdef char **strings_X = <char **>malloc(N_X * sizeof(char *))
    cdef int i, j
    j = 0
    for i in range(len(X)):
        if len(X[i]) < K:
            continue
        strings_X[j] = <char *>malloc((len(X[i])+1) * sizeof(char))
        strcpy(strings_X[j], X[i].encode('ASCII'))
        j += 1
    assert j == N_X
    ret = _compute_X(strings_X, N_X, K, max_dist, kmer2idx)
    for i in range(N_X):
        free(strings_X[i])
    free(strings_X)
    return ret

cdef void _normalize_rows(counts):
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
    def __init__(self, unsigned int k, bint normalized, bint aa, int max_dist=-1):
        self.k = k
        self.normalized = normalized
        self.max_dist = max_dist
        self.aa = aa

    def __call__(self, X, Y):
        '''
        See :meth:`get_K_matrix`.
        '''
        return self.get_K_matrix(X, Y)

    def get_K_matrix(self, X, Y):
        r'''
        Get the matrix :math:`\mathbf K` where :math:`K_{ij} = \langle \Phi(X_i), \Phi(Y_j) \rangle`.

        Args:
            X (list):
                list of strings.
            Y (list):
                list of strings (or ``None`` if ``Y = X``).

        Return:
            np.ndarray:
                Kernel matrix
        '''
        cdef bint Y_is_None = Y is None or X is Y
        counts_X = _get_count_matrix(X, self.k, self.max_dist, self.aa)
        if Y_is_None:
            ret = cdot_sparse_matrices(counts_X, counts_X)
            if self.normalized:
                _normalize_array_XX(ret, len(X))
        else:
            counts_Y = _get_count_matrix(Y, self.k, self.max_dist, self.aa)
            if self.normalized:
                _normalize_rows(counts_X)
                _normalize_rows(counts_Y)
            ret = cdot_sparse_matrices(counts_X, counts_Y)
        assert isinstance(ret, np.ndarray)
        return ret

    cdef double single_instance(self, str x, str x_prime):
        return (_get_count_matrix([x], self.k, self.max_dist, self.aa).dot(_get_count_matrix([x_prime], self.k, self.max_dist, self.aa).T))[0,0]

    cdef SparseMatrix vectorize(self, seqs):
        if not isinstance(seqs, list):
            seqs = list(seqs)
        ret = _get_count_matrix(seqs, self.k, self.max_dist, self.aa)
        return ret
