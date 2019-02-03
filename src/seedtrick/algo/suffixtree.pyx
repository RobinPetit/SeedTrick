cimport libc.stdlib

cdef class SuffixTree:
    cdef kmer_suffix_tree_t *_s_t

    def __init__(self, str s, unsigned int k):
        self._s_t = create_kmer_suffix_tree(s.encode('utf-8'), k)

    def display(self):
        print_kmer_suffix_tree(self._s_t)

    def __dealloc__(self):
        free_kmer_suffix_tree(&self._s_t)
