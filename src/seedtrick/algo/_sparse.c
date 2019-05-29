#include <stdlib.h>  // NULL
#include "_sparse.h"

linked_list_t make_ll(void) {
	linked_list_t ret;
	ret.length = 0;
	ret.first = ret.last = NULL;
	return ret;
}

static node_t *make_node(unsigned int idx, float value) {
	node_t *ret = malloc(sizeof(node_t));
	ret->idx = idx;
	ret->value = value;
	return ret;
}

void append(linked_list_t *ll, unsigned int idx, float value) {
	node_t *new_node;
	if(value == 0) {
		remove_node(ll, idx);
		return;
	}
	if(ll->length == 0) {
		new_node = make_node(idx, value);
		new_node->prev = new_node->next = NULL;
		ll->first = ll->last = new_node;
		ll->length = 1;
		return;
	}
	if(idx > ll->last->idx) {
		new_node = make_node(idx, value);
		ll->last->next = new_node;
		new_node->prev = ll->last;
		new_node->next = NULL;
		ll->last = new_node;
		ll->length++;
	} else if(idx < ll->first->idx) {
		new_node = make_node(idx, value);
		ll->first->prev = new_node;
		new_node->next = ll->first;
		new_node->prev = NULL;
		ll->first = new_node;
		ll->length++;
	} else {
		// First of all: find if idx is already in a node
		node_t *current = ll->first;
		while(current->idx < idx && current->next != NULL) {
			current = current->next;
		}
		// Here either current->idx if idx is already in the vector
		// or current->idx > idx if idx is not yet in the vector
		if(current->idx == idx) {
			current->value = value;
		} else {
			new_node = make_node(idx, value);
			new_node->next = current;
			new_node->prev = current->prev;
			current->prev = new_node;
			new_node->prev->next = new_node;
			ll->length++;
		}
	}
}

const node_t *get_node(const linked_list_t *ll, unsigned int idx) {
	node_t *current;
	if(ll->length == 0 || ll->last->idx < idx)
		return NULL;
	current = ll->first;
	while(current->idx < idx) {
		current = current->next;
	}
	return (current->idx == idx) ? current : NULL;
}

float get_value(const linked_list_t *ll, unsigned int idx) {
	const node_t *node = get_node(ll, idx);
	return (node == NULL) ? 0 : node->value;
}

void remove_node(linked_list_t *ll, unsigned int idx) {
	if(idx > ll->last->idx) {
		return;
	}
	node_t *current = ll->first;
	while(current->idx < idx && current->next != NULL) {
		current = current->next;
	}
	if(current->idx == idx) {
		current->next->prev = current->prev;
		current->prev->next = current->next;
		free(current);
		ll->length--;
	}
}

void free_ll(linked_list_t *ll) {
	node_t *current = ll->first;
	while(current != ll->last) {
		current = current->next;
		free(current->prev);
	}
	free(current);
	ll->length = 0;
	ll->first = ll->last = NULL;
}

float dot_product(const linked_list_t *x, const linked_list_t *y) {
	float ret = 0;
	node_t *current_x = x->first;
	node_t *current_y = y->first;
	while(current_x != NULL) {
		while(current_y != NULL && current_y->idx < current_x->idx) {
			current_y = current_y->next;
		}
		if(current_y == NULL)
			break;
		if(current_x->idx == current_y->idx) {
			ret += current_x->value * current_y->value;
		}
		current_x = current_x->next;
	}
	return ret;
}

linked_list_t copy_ll(const linked_list_t *ll) {
	linked_list_t ret = make_ll();
	if(ll->length == 0) {
		return ret;
	}
	node_t *current = ll->first;
	while(current->next != NULL) {
		append(&ret, current->idx, current->value);
		current = current->next;
	}
	return ret;
}
