#cython:language_level=3

from libc.math cimport sqrt

cimport numpy as np
import numpy as np

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

    cdef void plus_equal(self, SparseVector other):
        _add_inplace(&self.ll, &other.ll)

    cdef void divide_by_scalar(self, float scalar):
        assert scalar != 0
        divide_by_scalar(&self.ll, scalar)

    def __sub__(self, SparseVector other):
        return subtract(self, other)

    def __add__(self, SparseVector other):
        return add(self, other)

    def __iadd__(self, SparseVector other):
        self.plus_equal(other)
        return self

    def __eq__(self, SparseVector other):
        print('Testing equality')
        return self.dist(other) == 0

    def __setitem__(self, key, value):
        self.set(int(key), float(value))

    def __getitem__(self, key):
        return self.get(int(key))

    def __len__(self):
        return self.length()

    def dist(self, SparseVector other):
        return vectors_dist(self, other)

cdef float vectors_dist(SparseVector x, SparseVector y):
    cdef SparseVector diff = subtract(x, y)
    return sqrt(diff.squared_ell2_norm())

cdef class SparseMatrix:
    def __init__(self, unsigned int N):
        cdef unsigned int i
        self.N = N
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
            ret = SparseMatrix(0)
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

    def normalise_rows(self):
        cdef unsigned int i
        cdef SparseVector row
        for i in range(self.N):
            row  = self.rows[i]
            row.divide_by_scalar(sqrt(row.squared_ell2_norm()))

    cdef unsigned int length(self):
        return self.N

cdef SparseMatrix clone_matrix(SparseMatrix m):
    cdef unsigned int i
    cdef SparseVector tmp
    ret = SparseMatrix(0)
    ret.N = m.N
    ret.rows = np.empty(ret.N, dtype=SparseVector)
    cdef SparseVector[:] m_rows = m.rows
    for i in range(m.N):
        tmp = SparseVector()
        tmp.ll = copy_ll(&m_rows[i].ll)
        ret.rows[i] = tmp
    return ret

cdef np.ndarray cdot_sparse_matrices(SparseMatrix x, SparseMatrix y):
    cdef np.ndarray ret = np.empty((x.N, y.N), dtype=np.float)
    cdef unsigned int i, j
    cdef SparseVector row_x, row_y
    for i in range(x.N):
        row_x = x.rows[i]
        for j in range(y.N):
            row_y = y.rows[j]
            ret[i,j] = row_x.dot(row_y)
    return ret

cdef SparseVector subtract(SparseVector x, SparseVector y):
    cdef SparseVector ret = SparseVector()
    ret.ll = _subtract(&x.ll, &y.ll)
    return ret

cdef SparseVector add(SparseVector x, SparseVector y):
    cdef SparseVector ret = SparseVector()
    ret.ll = _add(&x.ll, &y.ll)
    return ret


cdef np.ndarray pairwise_distances(SparseMatrix x, SparseMatrix y):
    cdef np.ndarray ret = np.empty((x.length(), y.length()), dtype=np.float)
    cdef unsigned int i, j
    for i in range(x.length()):
        for j in range(y.length()):
            ret[i,j] = subtract(x.rows[i], y.rows[j]).squared_ell2_norm()
    return ret
