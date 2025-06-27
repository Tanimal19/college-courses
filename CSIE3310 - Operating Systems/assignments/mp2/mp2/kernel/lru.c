#include "lru.h"

#include "param.h"
#include "types.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "defs.h"
#include "proc.h"

void lru_init(lru_t *lru){
	lru->size = 0;
}

int lru_push(lru_t *lru, uint64 e){
	if (lru_full(lru)) {
		if (lru_pop(lru, 0) == -1) return -1;
	}

	lru->bucket[lru->size] = e;

	lru->size ++;
	return 0;
}

uint64 lru_pop(lru_t *lru, int idx){
	if (lru_empty(lru)) {
		return -1;
	}

	int oidx = idx;
	while (*((pte_t *)lru->bucket[idx]) & PTE_P) {
		idx ++;
		if (idx >= PG_BUF_SIZE) idx = 0;
		if (idx == oidx) return -1;
	}

	uint64 ret = lru->bucket[idx];

	for (int i=idx ; i<lru->size-1; i++) {
		lru->bucket[i] = lru->bucket[i+1];
	}

	lru->size --;
	lru->bucket[lru->size] = 0;

	return ret;
}

int lru_empty(lru_t *lru){
	if (lru->size == 0) {
		return 1;
	}
	else {
		return 0;
	}
}

int lru_full(lru_t *lru){
	if (lru->size == PG_BUF_SIZE) {
		return 1;
	}
	else {
		return 0;
	}
}

int lru_clear(lru_t *lru){
	lru->size = 0;
	for (int i=0; i<PG_BUF_SIZE; i++) {
		lru->bucket[i] = 0;
	}
	return 0;
}

int lru_find(lru_t *lru, uint64 e){
	if (lru_empty(lru)) {
		return -1;
	}

	for (int i=0; i<lru->size; i++) {
		if (lru->bucket[i] == e) {
			return i;
		}
	}

	return -1;
}
