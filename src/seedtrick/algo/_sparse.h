#ifndef _SPARSE_H
#define _SPARSE_H

#include <stdbool.h>

struct node {
	float value;
	unsigned long long int idx;
	struct node *next;
	struct node *prev;
};

typedef struct node node_t;

typedef struct {
	unsigned int length;
	node_t *first;
	node_t *last;
} linked_list_t;

linked_list_t make_ll(void);
void append(linked_list_t *ll, unsigned int idx, float value);
const node_t *get_node(const linked_list_t *ll, unsigned int idx);
float get_value(const linked_list_t *ll, unsigned int idx);
void remove_node(linked_list_t *ll, unsigned int idx);
void free_ll(linked_list_t *ll);

float sum_of_components(const linked_list_t *ll);

float dot_product(const linked_list_t *x, const linked_list_t *y);
linked_list_t copy_ll(const linked_list_t *ll);

linked_list_t _subtract(const linked_list_t *x, const linked_list_t *y);
linked_list_t _add(const linked_list_t *x, const linked_list_t *y);

void _add_inplace(linked_list_t *dest, const linked_list_t *src);
void _subtract_inplace(linked_list_t *dest, const linked_list_t *src);

void divide_by_scalar(linked_list_t *ll, float scalar);

#endif // _SPARSE_H
