#cython:language_level=3

from libc.math cimport sqrt

from seedtrick.kernels.base cimport Kernel
from seedtrick.algo.suffixtree cimport KmerSuffixTree

cdef class SpectrumKernel(Kernel):
    r'''
    Implementation of the :math:`k`-Spectrum Kernel defined in [1]

    **Definition**

    Let :math:`\Sigma` be an alphabet (typically the set of Amino Acids). For :math:`k > 0`,
    let :math:`\Phi` be the mapping:

    .. math::
        \Phi^k : \Sigma^* \to {\mathbb N}^{|\Sigma|^k} : s \mapsto (\phi_a^k(s))_{a \in \Sigma^k},

    where for every :math:`a \in \Sigma^k`, :math:`\phi_a^k` is the mapping:

    .. math::
        \phi_a^k : \Sigma^* \to \mathbb N : s \mapsto \left|\left\{i \in \{1, \ldots, |s|-k+1\} : s_is_{i+1}\ldots s_{i+k-1} = a\right\}\right|.

    The :math:`k`-Spectrum Kernel is then defined by:

    .. math::
        \kappa_k : \Sigma^* \times \Sigma^* \to \mathbb R : (s, t) \mapsto \langle \Phi^k(s), \Phi^k(t) \rangle.

    References
    ----------

    [1] Leslie, C., Eskin, E., & Noble, W. S. (2001).
    The spectrum kernel: A string kernel for SVM protein classification. In Biocomputing 2002 (pp. 564-575).
    '''
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
