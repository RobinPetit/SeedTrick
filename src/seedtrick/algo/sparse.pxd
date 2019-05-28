#cython:language_level=3

cimport numpy as np

cdef class SparseVector:
	cdef unsigned int nb_non_negative_entries
	cdef unsigned int i
	cdef np.ndarray values
	cdef np.ndarray indices

	cdef inline np.float_t[:] to_memoryview(self)
	cdef inline np.uint64_t[:] get_indices(self)
	cdef inline float get(self, np.uint64_t idx)
	cdef inline void set(self, np.uint64_t idx, float value)
	cdef inline unsigned int length(self)
	cdef void flush(self)

	cdef float squared_ell2_norm(self)

cdef class SparseMatrix:
	cdef list rows
	cdef unsigned int N, M

cdef SparseMatrix clone_matrix(SparseMatrix m, int new_M=*)
cdef SparseVector subtraction_sparse_vectors(SparseVector x, SparseVector y)
cdef float cdot_sparse_vectors(SparseVector x, SparseVector y)
cdef np.ndarray cdot_sparse_matrices(SparseMatrix x, SparseMatrix y)
