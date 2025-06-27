#include "my_pool.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void tpool_add(tpool *pool, void *(*func)(void *), void *arg) {
  /* add new job into queue tail */
  pthread_mutex_lock(pool->queue_lock);
  job *new = malloc(sizeof(job));
  new->func = func;
  new->arg = arg;
  new->next = NULL;

  if(pool->qhead == NULL){
    pool->qhead = new;
  }

  if(pool->qtail != NULL){
    pool->qtail->next = new;
  }
  pool->qtail = new;

  pool->qlen++;
  pthread_mutex_unlock(pool->queue_lock);

  /* if there's waiting threads, wake up one thread */
  pthread_mutex_lock(pool->wait_lock);

  if(pool->wait_threads > 0){
    pthread_cond_signal(pool->wait_cond);
    pool->wait_threads--;
  }
  pthread_mutex_unlock(pool->wait_lock);

  return;
}

void tpool_wait(tpool *pool) {
  pool->end = 1;

  /* wake up all waiting threads */
  pthread_mutex_lock(pool->wait_lock);
  pthread_cond_broadcast(pool->wait_cond);
  pthread_mutex_unlock(pool->wait_lock);

  /* block wait all threads finish and join */
  for(int i=0 ; i<pool->n_threads ; i++){
    pthread_join(pool->THREADS[i], NULL);
  }

  return;
}

void tpool_destroy(tpool *pool) {
  pthread_attr_destroy(pool->attr);
  pthread_mutex_destroy(pool->wait_lock);
  pthread_cond_destroy(pool->wait_cond);
  pthread_mutex_destroy(pool->queue_lock);
  free(pool);
  return;
}

tpool *tpool_init(int n_threads) {
  
  /* init pool */
  tpool *pool = malloc(sizeof(tpool));
  pool->n_threads = n_threads;
  pool->THREADS = malloc(sizeof(pthread_t) * n_threads);

  pool->attr = malloc(sizeof(pthread_attr_t));
  pthread_attr_init(pool->attr);
  pthread_attr_setdetachstate(pool->attr, PTHREAD_CREATE_JOINABLE);

  pool->wait_threads = 0;
  pool->wait_lock = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(pool->wait_lock, NULL);
  pool->wait_cond = malloc(sizeof(pthread_cond_t));
  pthread_cond_init(pool->wait_cond, NULL);

  pool->qlen = 0;
  pool->qhead = NULL;
  pool->qtail = NULL;
  pool->queue_lock = malloc(sizeof(pthread_mutex_t));
  pthread_mutex_init(pool->queue_lock, NULL);

  pool->end = 0;
  
  /* create threads */
  for(int i=0 ; i<n_threads ; i++){
    if(pthread_create(&pool->THREADS[i], pool->attr, start_fn, pool) != 0)
      printf("pthread_create[%d] error\n", i);
  }

  return pool;
}

void *start_fn(void *arg){
  tpool *pool = (tpool *)arg;
  pthread_t tid = pthread_self();

  while(1){
    /* check queue empty */
    pthread_mutex_lock(pool->queue_lock);
    
    /* queue not empty */
    if(pool->qlen > 0){
      /* get job */
      job *cur = pool->qhead;
      if(pool->qlen == 1){
        pool->qhead = NULL;
        pool->qtail = NULL;
      }
      else{
        pool->qhead = pool->qhead->next;
      }
      pool->qlen--;
      pthread_mutex_unlock(pool->queue_lock);

      /*  
       *  sleep some time, make sure thread will not finish job too fast 
       *  which make the output be 0->2->1 or 1->2->0
       */
      sleep(0.00001);

      /* do job */
      cur->func(cur->arg);
    }
    /* queue is empty */
    else{
      pthread_mutex_unlock(pool->queue_lock);

      /* if tpool_wait() */
      if(pool->end == 1){
        pthread_exit(NULL);
      }

      /* waiting */
      pthread_mutex_lock(pool->wait_lock);
      pool->wait_threads++;
      pthread_cond_wait(pool->wait_cond, pool->wait_lock);
      pthread_mutex_unlock(pool->wait_lock);
    }
  }
}