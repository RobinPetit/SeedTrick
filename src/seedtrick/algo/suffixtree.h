#ifndef SUFFIX_TREE_H
#define SUFFIX_TREE_H

struct Node;

struct Edge {
	struct Node *child;
	const char *label;
	unsigned int label_length;
	int idx_leaf;
};

struct Node {
	struct Edge *edges;
	unsigned int nb_children;
};

typedef struct Edge edge_t;
typedef struct Node node_t;

struct SuffixTree {
	char *string;
	node_t *root;
};

typedef struct SuffixTree suffix_tree_t;

suffix_tree_t *create_suffix_tree(const char *);
void free_suffix_tree(suffix_tree_t **);
void print_suffix_tree(const suffix_tree_t *s_t);

#endif

