#cython:language_level=3

from seedtrick.kernels.base cimport Kernel
from seedtrick.algo.sparse cimport SparseMatrix

cdef extern from "_odh.h":
	unsigned int _kmer_to_idx_aa(const char *, unsigned int)
	unsigned int _kmer_to_idx_nt(const char *, unsigned int)
	ctypedef unsigned int (*kmer2idx_t)(const char *, unsigned int)

cdef class ODHKernel(Kernel):
	cdef unsigned int k
	cdef bint normalized
	cdef bint aa  # True if amino acid sequence, False if nucleotide sequence
	cdef int max_dist

	cdef double single_instance(self, str x, str x_prime)

	cdef SparseMatrix vectorize(self, seqs, bint verbose)
