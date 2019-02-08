cimport numpy as np
import numpy as np

from libc.string cimport strlen

from scipy.sparse import dok_matrix, lil_matrix

# TODO: implement ODH on several different sparse matrix implementations and find the most efficient (np.array, lil_matrix, dok_matrix, csr_matrix)

sparse_matrix_t = lil_matrix

def count_odh_kmers(s, k, D=None):
    if D is None:
        D = len(s)-k+1
    M = NB_AMINO_ACIDS**k
    shape = (M, M, D)
    print('Dimension: {}'.format(shape))
    ret = np.zeros(shape, dtype=np.uint)
    ret = lil_matrix()
    _count_kmers_array(s.encode('ASCII'), k, ret)
    return ret

cdef void _count_kmers_lil(const char *s, unsigned int k, counts):  # TODO: add lil sparse matrix
    cdef unsigned int i, j
    cdef unsigned int N = strlen(s);
    for i in range(N-k+1):
        for j in range(i, N-k+1):
            counts[_kmer_to_idx(&s[i], k), _kmer_to_idx(&s[j], k), j-i] += 1

cdef void _count_kmers_array(const char *s, unsigned int k, np.uint_t[:,:,:] counts):
    #counts[i,j,d] == h_{ij}^d(s)
    cdef unsigned int i, j
    cdef unsigned int N = strlen(s);
    for i in range(N-k+1):
        for j in range(i, N-k+1):
            counts[_kmer_to_idx(&s[i], k), _kmer_to_idx(&s[j], k), j-i] += 1
