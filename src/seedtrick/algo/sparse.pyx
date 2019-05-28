#cython:language_level=3

cimport numpy as np
import numpy as np

import cython

@cython.final
cdef class SparseVector:
    def __init__(self, nb_non_negative_entries):
        self.nb_non_negative_entries = nb_non_negative_entries
        self.i = 0
        self.values = np.empty(self.nb_non_negative_entries, dtype=np.float)
        self.indices = np.empty(self.nb_non_negative_entries, np.uint64)

    cdef np.float_t[:] to_memoryview(self):
        return self.values[:self.i]

    cdef np.uint64_t[:] get_indices(self):
        return self.indices[:self.i:]

    cdef float get(self, np.uint64_t idx):
        cdef unsigned int j = 0
        while j < self.i:
            if self.indices[j] == idx:
                return self.values[j]
            j += 1
        return 0.

    cdef void set(self, np.uint64_t idx, float value):
        cdef unsigned int j = 0
        cdef unsigned int k = 0
        while j < self.i:
            if self.indices[j] == idx:
                self.values[j] = value
                if value == 0:
                    # Don't forget to free the available cell
                    for k in range(j+1, self.i):
                        self.values[k-1] = self.values[k]
                        self.indices[k-1] = self.indices[k]
                    self.i -= 1
                return
            j += 1
        if value != 0 and self.i < self.nb_non_negative_entries:
            self.indices[self.i] = idx
            self.values[self.i] = value
            self.i += 1
        else:
            print('Vector too small')

    cdef unsigned int length(self):
        return self.i

    cdef float squared_ell2_norm(self):
        return np.dot(self.values[:self.i], self.values[:self.i])

    cdef void flush(self):
        self.i = 0

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
        self.N = shape[0]
        self.M = shape[1]
        self.rows = [SparseVector(shape[1]) for _ in range(self.N)]

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
            ret.rows = [self.rows[j] for j in k]
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

cdef SparseMatrix clone_matrix(SparseMatrix m, int new_M=-1):
        cdef unsigned int i
        if new_M > 0:
            assert new_M > m.M
        else:
            new_M = m.M
        ret = SparseMatrix((m.N, new_M))
        for i in range(m.N):
            _clone_vector(m[i], ret[i])
        return ret

cdef void _clone_vector(SparseVector src, SparseVector dest):
    dest.values[src.i:] = src.values[src.i:]
    dest.indices[src.i:] = src.indices[src.i:]
    dest.i = src.i

cdef SparseVector subtraction_sparse_vectors(SparseVector x, SparseVector y):
    cdef SparseVector ret = SparseVector(x.nb_non_negative_entries + y.nb_non_negative_entries)
    cdef unsigned int i
    for i in range(x.i):
        ret.set(x.indices[i], x.values[i] - y.get(x.indices[i]))
    for i in range(y.i):
        ret.set(y.indices[i], x.get(y.indices[i]) - y.values[i])
    return ret

cdef float cdot_sparse_vectors(SparseVector x, SparseVector y):
    cdef float ret = 0
    cdef idx = 0
    cdef np.uint64_t[:] indices_x = x.get_indices()
    cdef np.uint64_t[:] indices_y = y.get_indices()
    for i in range(len(indices_x)):
        ret += x.get(indices_x[i]) * y.get(indices_x[i])
    return ret

cdef np.ndarray cdot_sparse_matrices(SparseMatrix x, SparseMatrix y):
    cdef np.ndarray ret = np.empty((len(x), len(y)), dtype=np.float)
    cdef int i, j
    for i in range(len(x)):
        for j in range(len(y)):
            ret[i,j] = cdot_sparse_vectors(x[i], y[j])
    return ret
