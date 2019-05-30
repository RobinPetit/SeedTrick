#cython:language_level=3

cimport numpy as np

cdef extern from "_sparse.h":
	ctypedef struct node_t

	ctypedef struct node_t:
		float value
		unsigned int idx
		node_t *prev
		node_t *next

	ctypedef struct linked_list_t:
		unsigned int length
		node_t *first
		node_t *last

	cdef linked_list_t make_ll()
	cdef void free_ll(linked_list_t *ll)
	cdef float get_value(linked_list_t *ll, unsigned int idx)
	cdef void append(linked_list_t *ll, unsigned int idx, float value)
	cdef float sum_of_components(const linked_list_t *ll)
	cdef float dot_product(const linked_list_t *x, const linked_list_t *y)
	cdef linked_list_t copy_ll(const linked_list_t *ll)
	cdef linked_list_t _subtract(const linked_list_t *x, const linked_list_t *y)
	cdef linked_list_t _add(const linked_list_t *x, const linked_list_t *y)
	cdef void divide_by_scalar(linked_list_t *ll, float scalar)

	cdef void _add_inplace(linked_list_t *dest, const linked_list_t *src)

cdef float vectors_dist(SparseVector x, SparseVector y)

cdef class SparseVector:
	cdef linked_list_t ll

	cdef inline float get(self, np.uint64_t idx)
	cdef inline void set(self, np.uint64_t idx, float value)
	cdef inline unsigned int length(self)
	cdef void flush(self)
	cdef float dot(self, SparseVector other)
	cdef void plus_equal(self, SparseVector other)
	cdef void divide_by_scalar(self, float scalar)

	cdef float squared_ell2_norm(self)

cdef class SparseMatrix:
	cdef np.ndarray rows
	cdef unsigned int N

	cdef unsigned int length(self)

cdef SparseMatrix clone_matrix(SparseMatrix m)
cdef np.ndarray cdot_sparse_matrices(SparseMatrix x, SparseMatrix y)
cdef np.ndarray pairwise_distances(SparseMatrix x, SparseMatrix y)
