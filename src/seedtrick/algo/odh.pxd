cdef extern from "_odh.h":
	unsigned int NB_AMINO_ACIDS
	unsigned int _kmer_to_idx(const char *, unsigned int)
