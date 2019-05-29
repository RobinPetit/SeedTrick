#cython:language_level=3

cimport numpy as np
import numpy as np

import cython

@cython.final
cdef class SparseVector:
    def __init__(self):
        self.ll = make_ll()

    def __dealloc__(self):
        free_ll(&self.ll)

    cdef float get(self, np.uint64_t idx):
        return get_value(&self.ll, idx)

    cdef void set(self, np.uint64_t idx, float value):
        append(&self.ll, idx, value)

    cdef unsigned int length(self):
        return self.ll.length

    cdef float squared_ell2_norm(self):
        return self.dot(self)

    cdef void flush(self):
        free_ll(&self.ll)

    cdef float dot(self, SparseVector other):
        return dot_product(&self.ll, &other.ll)

    def __sub__(self, other):
        return subtraction_sparse_vectors(self, other)

    def __setitem__(self, key, value):
        self.set(int(key), float(value))

    def __getitem__(self, key):
        return self.get(int(key))

    def __len__(self):
        return self.length()

cdef class SparseMatrix:
    def __init__(self, shape):
        cdef unsigned int i
        self.N = shape[0]
        self.M = shape[1]
        self.rows = np.empty(self.N, dtype=SparseVector)
        for i in range(self.N):
            self.rows[i] = SparseVector()

    def __setitem__(self, k, value):
        assert len(k) == 2
        if k[0] >= self.N:
            raise IndexError()
        self.rows[k[0]][k[1]] = value

    def __getitem__(self, k):
        if isinstance(k, int):
            return self.rows[k]
        if isinstance(k, np.ndarray):
            assert k.ndim == 1
            if not (k < len(self)).all():
                raise IndexError()
            ret = SparseMatrix((0, self.M))
            ret.N = k.shape[0]
            ret.rows = self.rows[k]
            return ret
        assert isinstance(k, tuple)
        if len(k) == 1:
            if k[0] >= self.N:
                raise IndexError()
            return self.rows[k[0]]
        elif len(k) == 2:
            if k[0] >= self.N:
                raise IndexError()
            return self.rows[k[0]][k[1]]
        else:
            raise IndexError('Too many dimensions')

    def __len__(self):
        return self.N

    cdef inline void _set_row(self, unsigned int idx, linked_list_t ll):
        assert idx < self.N
        cdef SparseVector[:] self_rows = self.rows
        self_rows[idx].ll = ll

cdef SparseMatrix clone_matrix(SparseMatrix m, int new_M=-1):
    cdef unsigned int i
    if new_M > 0:
        assert new_M > m.M
    else:
        new_M = m.M
    ret = SparseMatrix((0, new_M))
    ret.N = m.N
    cdef SparseVector[:] m_rows = m.rows
    for i in range(m.N):
        ret._set_row(i, copy_ll(&m_rows[i].ll))
    return ret

cdef SparseVector subtraction_sparse_vectors(SparseVector x, SparseVector y):
    cdef SparseVector ret = SparseVector(x.nb_non_negative_entries + y.nb_non_negative_entries)
    cdef unsigned int i
    for i in range(x.i):
        ret.set(x.indices[i], x.values[i] - y.get(x.indices[i]))
    for i in range(y.i):
        ret.set(y.indices[i], x.get(y.indices[i]) - y.values[i])
    return ret

cdef np.ndarray cdot_sparse_matrices(SparseMatrix x, SparseMatrix y):
    cdef np.ndarray ret = np.empty((x.N, y.N), dtype=np.float)
    cdef unsigned int i, j
    cdef SparseVector[:] rows_x = x.rows
    cdef SparseVector[:] rows_y = y.rows
    for i in range(x.N):
        for j in range(y.N):
            ret[i,j] = rows_x[i].dot(rows_y[j])
    return ret
