#include <pthread.h>
#ifndef __MY_THREAD_POOL_H
#define __MY_THREAD_POOL_H

typedef struct job {
  void *(*func)(void *);
  void *arg;
  struct job *next;
} job;

typedef struct tpool {
  int n_threads;
  pthread_t *THREADS;
  pthread_attr_t *attr;

  int wait_threads;
  pthread_mutex_t *wait_lock;
  pthread_cond_t *wait_cond;

  int qlen;
  job *qhead;
  job *qtail;
  pthread_mutex_t *queue_lock;

  int end;
} tpool;


tpool *tpool_init(int n_threads);
void tpool_add(tpool *, void *(*func)(void *), void *);
void tpool_wait(tpool *);
void tpool_destroy(tpool *);
void *start_fn(void *);

#endif