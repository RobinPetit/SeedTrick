#ifndef KMER_SUFFIX_TREE_H
#define KMER_SUFFIX_TREE_H

struct Node;

struct Edge {
	struct Node *child;
	const char *label;
	unsigned int label_length;
};

struct Node {
	struct Edge *edges;
	unsigned int nb_children;
	unsigned int counts[2];
};

typedef struct Edge edge_t;
typedef struct Node node_t;

struct KmerSuffixTree {
	char *s;
	char *t;
	unsigned int k;
	node_t *root;
};

typedef struct KmerSuffixTree kmer_suffix_tree_t;

kmer_suffix_tree_t *create_kmer_suffix_tree(const char *, const char *, unsigned int k);
void free_kmer_suffix_tree(kmer_suffix_tree_t **);
void print_kmer_suffix_tree(const kmer_suffix_tree_t *s_t);
const unsigned int *get_counts(const kmer_suffix_tree_t *, const char *);

#endif

