#ifndef ALGO_ODH_H
#define ALGO_ODH_H

static const unsigned int NB_AMINO_ACIDS = 20;

typedef unsigned int (*kmer2idx_t)(const char *, unsigned int);

static const int _aa_to_int[26] = {
	 1,  //'A',
	-1,  //'B',
	 2,  //'C',
	 3,  //'D',
	 4,  //'E',
	 5,  //'F',
	 6,  //'G',
	 7,  //'H',
	 8,  //'I',
	-1,  //'J',
	 9,  //'K',
	 10, //'L',
	 11, //'M',
	 12, //'N',
	-1,  //'O',
	 13, //'P',
	 14, //'Q',
	 15, //'R',
	 16, //'S',
	 17, //'T',
	-1,  //'U',
	 18, //'V',
	 19, //'W',
	-1,  //'X',
	 20, //'Y',
	-1,  //'Z'
};

unsigned int _kmer_to_idx_aa(const char *, unsigned int);
unsigned int _kmer_to_idx_nt(const char *, unsigned int);

#endif
