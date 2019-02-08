cdef extern from "_odh.h":
	unsigned int _kmer_to_idx(const char *, unsigned int)
