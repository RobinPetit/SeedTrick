cdef extern from "kmersuffixtree.h":

	ctypedef struct kmer_suffix_tree_t:
		pass

	kmer_suffix_tree_t *create_kmer_suffix_tree(const char *, unsigned int k)
	void free_kmer_suffix_tree(kmer_suffix_tree_t **)
	void print_kmer_suffix_tree(const kmer_suffix_tree_t *)
