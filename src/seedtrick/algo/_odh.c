#include "_odh.h"

#include "stdio.h"

#ifndef DEBUG
# define DEBUG 1
#endif

unsigned int _kmer_to_idx_aa(const char *s, unsigned int k) {
	unsigned int ret = 0;
	unsigned int i;
	for(i = 0; i < k; ++i) {
		ret *= NB_AMINO_ACIDS;
		ret += _aa_to_int[s[i] - 'A']-1;
	}
	return ret;
}

static inline unsigned int nt_idx(char nt) {
	switch(nt) {
	case 'A': return 0;
	case 'T': return 1;
	case 'C': return 2;
	case 'G': return 3;
	default:
		if(DEBUG)
			printf("Watch out: found unknown character '%c' (%d)\n", nt, (int)nt);
		return -1;  // Will make everything crash because (unsigned int)(-1) == 0xFFFFFFFF which is big for an index
	}
}

unsigned int _kmer_to_idx_nt(const char *s, unsigned int k) {
	unsigned int ret = 0;
	unsigned int i;
	for(i = 0; i < k; ++i) {
		ret *= 4;
		ret += nt_idx(s[i]);
	}
	return ret;
}
