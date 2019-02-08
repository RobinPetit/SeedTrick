#include "_odh.h"

unsigned int _kmer_to_idx(const char *s, unsigned int k) {
	printf("Finding idx of %.*s\n", k, s);
	unsigned int ret = 0;
	unsigned int i;
	for(i = 0; i < k; ++i) {
		printf("\tFound %d\n", _aa_to_int[s[i]-'A']);
		ret *= NB_AMINO_ACIDS;
		ret += _aa_to_int[s[i] - 'A'];
	}
	printf("\t\tGot %u\n", ret);
	return ret - 1;
}
