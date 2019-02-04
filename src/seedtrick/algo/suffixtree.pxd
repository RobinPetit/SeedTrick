cdef extern from "kmersuffixtree.h":

	ctypedef struct edge_t:
		node_t *child

	ctypedef struct node_t:
		edge_t *edges
		unsigned int nb_children
		int counts[2]

	ctypedef struct kmer_suffix_tree_t:
		unsigned int k
		node_t *root

	kmer_suffix_tree_t *create_kmer_suffix_tree(const char *, const char *, unsigned int k)
	void free_kmer_suffix_tree(kmer_suffix_tree_t **)
	void print_kmer_suffix_tree(const kmer_suffix_tree_t *)
	const unsigned int *get_counts(const kmer_suffix_tree_t *, const char *)

cdef class KmerSuffixTree:
	cdef kmer_suffix_tree_t *_s_t
