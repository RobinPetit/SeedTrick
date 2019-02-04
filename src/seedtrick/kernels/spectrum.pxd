from seedtrick.algo.suffixtree cimport KmerSuffixTree
from seedtrick.kernels.base cimport Kernel

cdef extern from "kmersuffixtree.h":

	ctypedef struct edge_t:
		node_t *child

	ctypedef struct node_t:
		edge_t *edges
		unsigned int nb_children
		int counts[2]

	ctypedef struct kmer_suffix_tree_t:
		node_t *root

cdef class SpectrumKernel(Kernel):
	cdef KmerSuffixTree s_t
	cdef unsigned int k
	cdef bint normalized
	cdef unsigned int kernel_value
	cdef unsigned int kernel_normalization_values[2]

	cdef void _compute_kernel(self, const node_t *node)
