#cython:language_level=3

cimport libc.stdlib

cdef class KmerSuffixTree:
    def __init__(self, str s, str t, unsigned int k):
        self._s_t = create_kmer_suffix_tree(
            s.encode('ascii'), t.encode('ascii'), k
        )
        #self.display()

    def display(self):
        print_kmer_suffix_tree(self._s_t)

    def count(self, s):
        assert isinstance(s, str)
        assert len(s) == self._s_t.k
        cdef unsigned int *count_arrays = get_counts(self._s_t, s.encode('ascii'))
        if count_arrays == NULL:
            return (0, 0)
        else:
            return (count_arrays[0], count_arrays[1])

    def __dealloc__(self):
        free_kmer_suffix_tree(&self._s_t)
