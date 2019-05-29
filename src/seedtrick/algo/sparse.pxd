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
	cdef float dot_product(const linked_list_t *x, const linked_list_t *y)
	cdef linked_list_t copy_ll(const linked_list_t *ll)

cdef class SparseVector:
	cdef linked_list_t ll

	cdef inline float get(self, np.uint64_t idx)
	cdef inline void set(self, np.uint64_t idx, float value)
	cdef inline unsigned int length(self)
	cdef void flush(self)
	cdef float dot(self, SparseVector other)

	cdef float squared_ell2_norm(self)

cdef class SparseMatrix:
	#cdef list rows
	cdef np.ndarray rows
	cdef unsigned int N, M

	cdef void _set_row(self, unsigned int i, linked_list_t ll)

cdef SparseMatrix clone_matrix(SparseMatrix m, int new_M=*)
cdef SparseVector subtraction_sparse_vectors(SparseVector x, SparseVector y)
cdef np.ndarray cdot_sparse_matrices(SparseMatrix x, SparseMatrix y)
