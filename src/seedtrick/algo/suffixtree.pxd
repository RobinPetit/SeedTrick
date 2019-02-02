cdef extern from "suffixtree.h":
	ctypedef struct suffix_tree_t:
		pass
	suffix_tree_t *create_suffix_tree(const char *)
	void free_suffix_tree(suffix_tree_t **)
	void print_suffix_tree(suffix_tree_t *)
