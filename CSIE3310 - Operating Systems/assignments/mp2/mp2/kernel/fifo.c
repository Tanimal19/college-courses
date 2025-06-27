#include "fifo.h"

#include "param.h"
#include "types.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "defs.h"
#include "proc.h"

void q_init(queue_t *q){
	q->size = 0;
}

int q_push(queue_t *q, uint64 e){
	if (q_full(q)) {
		if (q_pop_idx(q, 0) == -1) return -1;
	}

	q->bucket[q->size] = e;

	q->size ++;
	return 0;
}

uint64 q_pop_idx(queue_t *q, int idx){
	if (q_empty(q)) {
		return -1;
	}

	int oidx = idx;
	while (*((pte_t *)q->bucket[idx]) & PTE_P) {
		idx ++;
		if (idx >= PG_BUF_SIZE) idx = 0;
		if (idx == oidx) return -1;
	}

	uint64 ret = q->bucket[idx];

	for (int i=idx ; i<q->size-1; i++) {
		q->bucket[i] = q->bucket[i+1];
	}

	q->size --;
	q->bucket[q->size] = 0;

	return ret;
}

int q_empty(queue_t *q){
	if (q->size == 0) {
		return 1;
	}
	else {
		return 0;
	}
}

int q_full(queue_t *q){
	if (q->size == PG_BUF_SIZE) {
		return 1;
	}
	else {
		return 0;
	}
}

int q_clear(queue_t *q){
	q->size = 0;
	for (int i=0; i<PG_BUF_SIZE; i++) {
		q->bucket[i] = 0;
	}
	return 0;
}

int q_find(queue_t *q, uint64 e){
	if (q_empty(q)) {
		return -1;
	}

	for (int i=0; i<q->size; i++) {
		if (q->bucket[i] == e) {
			return i;
		}
	}

	return -1;
}
