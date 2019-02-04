#cython:language_level=3

from libc.math cimport sqrt

from seedtrick.kernels.base cimport Kernel
from seedtrick.algo.suffixtree cimport KmerSuffixTree

cdef class SpectrumKernel(Kernel):
    def __init__(self, k, bint normalized):
        self.k = k
        self.normalized = normalized

    def __call__(self, s, t):
        self.s_t = KmerSuffixTree(s, t, self.k)
        self.kernel_value = 0
        self.kernel_normalization_values[0] = 0
        self.kernel_normalization_values[1] = 0
        self._compute_kernel(self.s_t._s_t.root)
        ret = self.kernel_value
        if self.normalized:
            ret /= sqrt(self.kernel_normalization_values[0] * self.kernel_normalization_values[1])
        return ret

    cdef void _compute_kernel(self, const node_t *node):
        if node.nb_children == 0:
            self.kernel_value += node.counts[0]*node.counts[1]
            if self.normalized:
                self.kernel_normalization_values[0] += node.counts[0]*node.counts[0]
                self.kernel_normalization_values[1] += node.counts[1]*node.counts[1]
        cdef unsigned int i
        for i in range(node.nb_children):
            self._compute_kernel(node.edges[i].child)
