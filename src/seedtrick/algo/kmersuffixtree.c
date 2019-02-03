#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kmersuffixtree.h"

static node_t *_make_empty_node(void) {
	node_t *ret = malloc(sizeof(node_t));
	ret->edges = NULL;
	ret->nb_children = 0;
	ret->count = 0;
	return ret;
}

static inline node_t *_make_root(void) {
	return _make_empty_node();
}

static edge_t _make_edge(node_t *child, const char *label, unsigned int label_length) {
	edge_t ret;
	ret.child = child;
	ret.label = label;
	ret.label_length = label_length;
	return ret;
}

static inline edge_t _make_edge_to_leaf(const char *label, unsigned int label_length) {
	node_t *new_leaf = _make_empty_node();
	new_leaf->count = 1;
	return _make_edge(new_leaf, label, label_length);
}

static unsigned int get_substr_len(const char *s1, unsigned int N1, const char *s2, unsigned int N2) {
	if(s1 == NULL || s2 == NULL)
		return 0;
	unsigned int N = (N1 < N2) ? N1 : N2;
	unsigned int i = 0;
	while(i < N && s1[i] != '\0' && s2[i] != '\0' && s1[i] == s2[i])
		++i;
	return i;
}

static inline bool is_leaf(const node_t *node) {
	return node->nb_children == 0;
}

///// Construction of Suffix Tree

static int _insert_suffix_leaf(node_t *node, const char *suffix, unsigned int length) {
	edge_t *new_edges = realloc(node->edges, (node->nb_children+1)*sizeof(edge_t));
	if(new_edges == NULL)
		return -1;
	node->edges = new_edges;
	node->edges[node->nb_children] = _make_edge_to_leaf(suffix, length);
	node->nb_children++;
	return 0;
}

static void _add_node(edge_t *prev_edge, unsigned int nb_children) {
	assert(prev_edge->child == NULL);
	node_t *new_node = malloc(sizeof(node_t));
	new_node->nb_children = nb_children;
	new_node->edges = malloc(nb_children*sizeof(edge_t));
	prev_edge->child = new_node;
}

static void _insert_suffix_branch(node_t *node, const char *suffix, unsigned int length) {
	edge_t *prev_edge = NULL;
	node_t *current_node = node;
	unsigned substr_len;
	unsigned int i;
	bool inserted = false;
	bool moved_forward;
	bool loop = true;
	do {
		i = 0;
		moved_forward = false;
		while(!moved_forward && i < current_node->nb_children) {
			substr_len = get_substr_len(
				suffix, length,
				current_node->edges[i].label,
				current_node->edges[i].label_length
			);
			if(substr_len > 0) {
				prev_edge = &current_node->edges[i];
				length -= substr_len;
				suffix = &suffix[substr_len];
				moved_forward = true;
				if(substr_len < prev_edge->label_length)
					loop = false;
				current_node = current_node->edges[i].child;
			}
			++i;
		}
		if(!moved_forward) {
			_insert_suffix_leaf(current_node, suffix, length);
			inserted = true;
		}
	} while(length > 0 && !inserted && current_node != NULL && loop);
	if(!loop) {
		assert(substr_len < prev_edge->label_length);
		int leaf_count = current_node->count;
		_add_node(prev_edge, 2);
		prev_edge->child->edges[0] = _make_edge_to_leaf(&prev_edge->label[substr_len], prev_edge->label_length - substr_len);
		prev_edge->child->edges[0].child->count = leaf_count;
		prev_edge->child->edges[1] = _make_edge_to_leaf(suffix, length); //, label_idx);
		prev_edge->label_length = substr_len;
	} else if(current_node == NULL) {
		assert(substr_len == prev_edge->label_length && !inserted);
		int leaf_count = current_node->count;
		_add_node(prev_edge, 2);
		prev_edge->child->edges[0]= _make_edge_to_leaf(suffix, 0); //, idx_leaf);
		prev_edge->child->edges[0].child->count = leaf_count;
		prev_edge->child->edges[1] = _make_edge_to_leaf(suffix, length); //, label_idx);
	} else if(length == 0) {
		current_node->count++;
	}
}

static void _build_kmer_suffix_tree(kmer_suffix_tree_t *s_t, unsigned int k) {
	unsigned int N = strlen(s_t->string);
	int i;
	for(i = N-k; i >= 0; --i)
		_insert_suffix_branch(s_t->root, &s_t->string[i], k);
}

kmer_suffix_tree_t *create_kmer_suffix_tree(const char *s, unsigned int k) {
	unsigned int N = strlen(s);
	kmer_suffix_tree_t *s_t = malloc(sizeof(kmer_suffix_tree_t));
	s_t->root = _make_root();
	s_t->string = malloc((N+1) * sizeof(char));
	strcpy(s_t->string, s);
	puts("Building kmer ST");
	_build_kmer_suffix_tree(s_t, k);
	return s_t;
}

static inline void _free(void **ptr) {
	free(*ptr);
	*ptr = NULL;
}

static void _free_node(node_t **node) {
	unsigned int i;
	for(i = 0; i < (*node)->nb_children; ++i) {
		if((*node)->edges[i].child != NULL)
			_free_node(&((*node)->edges[i].child));
	}
	_free((void **)&((*node)->edges));
	_free((void **)node);
}

void free_kmer_suffix_tree(kmer_suffix_tree_t **s_t) {
	_free((void **)&((*s_t)->string));
	_free_node(&(*s_t)->root);
	_free((void **)s_t);
}

static void _print_kmer_suffix_tree(const node_t *node, int nb_tabs) {
	assert(!is_leaf(node));
	unsigned int i;
	int j;
	for(i = 0; i < node->nb_children; ++i) {
		for(j = 0; j < nb_tabs; ++j)
			putchar('\t');
		if(is_leaf(node->edges[i].child))
			printf("%.*s -> [%d]\n", node->edges[i].label_length, node->edges[i].label, node->edges[i].child->count);
		else {
			printf("- %.*s\n", node->edges[i].label_length, node->edges[i].label);
			_print_kmer_suffix_tree(node->edges[i].child, nb_tabs+1);
		}
	}
}

void print_kmer_suffix_tree(const kmer_suffix_tree_t *s_t) {
	if(s_t == NULL)
		return;
	puts("Printing kmer ST");
	_print_kmer_suffix_tree(s_t->root, 0);
}

