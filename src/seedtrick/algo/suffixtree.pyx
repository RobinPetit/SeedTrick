cimport libc.stdlib

cdef class SuffixTree:
    cdef suffix_tree_t *_s_t

    def __init__(self, str s):
        self._s_t = create_suffix_tree(s.encode('utf-8'))

    def display(self):
        print_suffix_tree(self._s_t)

    def __dealloc__(self):
        free_suffix_tree(&self._s_t)
