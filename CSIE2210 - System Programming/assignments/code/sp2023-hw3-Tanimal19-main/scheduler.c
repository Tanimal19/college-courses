#include "threadtools.h"
#include <sys/signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/*
 * Print out the signal you received.
 * If SIGALRM is received, reset the alarm here.
 * This function should not return. Instead, call siglongjmp(sched_buf, 1).
 */
void sighandler(int signo) {
    if(signo == SIGTSTP){
        printf("caught SIGTSTP\n");
    }
    else if(signo == SIGALRM){
        printf("caught SIGALRM\n");
        alarm(timeslice);
    }
    longjmp(sched_buf, 1);
}



/*
 * Prior to calling this function, both SIGTSTP and SIGALRM should be blocked.
 */
void scheduler() {
    int from = setjmp(sched_buf);
    
    #ifdef DEBUG
    printf("[SCHD] from:%d\n", from);
    for(int j=0 ; j<rq_size ; j++){
        printf("rq[%d].id: %d\n", j, ready_queue[j]->id);
    }
    printf("-----\n");
    for(int j=0 ; j<wq_size ; j++){
        printf("wq[%d].id: %d\n", j, waiting_queue[j]->id);
    }
    #endif

    /* called by main.c */
    if(from == 0){
        rq_current = 0;
        longjmp(RUNNING->environment, 1);
    }

    if(bank.lock_owner == -1 && wq_size != 0){
        /* bring first thread in the waiting queue to the ready queue */
        ready_queue[rq_size] = waiting_queue[0];
        bank.lock_owner = ready_queue[rq_size]->id;
        rq_size++;
        
        /* filled the hole of waiting queue */
        for(int j=1 ; j<wq_size ; j++){
            waiting_queue[j-1] = waiting_queue[j];
        }
        wq_size--;
    }
    

    /* Remove the current thread from the ready queue if needed */
    if(from == 2){
        /* move to the end of the waiting queue */
        waiting_queue[wq_size] = RUNNING;
        wq_size++;
        
        /* filled the hole of ready queue */
        if(rq_size > 1){
            if(rq_current == rq_size-1){
                rq_current = 0;
            }
            else{
                RUNNING = ready_queue[rq_size-1];
            }
        }
        rq_size--;
    }
    else if(from == 3){
        /* clean up */
        free(RUNNING);

        /* filled the hole of ready queue */
        if(rq_size > 1){
            if(rq_current == rq_size-1){
                rq_current = 0;
            }
            else{
                RUNNING = ready_queue[rq_size-1];
            }
        }
        rq_size--;
    }
    

    /* for thread_yield */
    if(from == 1 && rq_size != 0){
        if(rq_current == rq_size-1){
            rq_current = 0;
        }
        else{
            rq_current++;
        }
        longjmp(RUNNING->environment, 1);
    }
    /* for lock and thread_exit */
    else if((from == 2 || from == 3) && rq_size != 0){
        longjmp(RUNNING->environment, 1);
    }

    if(rq_size == 0 && wq_size == 0)
        return;
}

    