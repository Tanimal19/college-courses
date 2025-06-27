#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "threadtools.h"

#define MAXARGS 5

int timeslice;
sigset_t base_mask, tstp_mask, alrm_mask;
struct tcb *ready_queue[], *waiting_queue[];
int rq_size, rq_current, wq_size;
jmp_buf sched_buf;
struct Bank bank;


/* prototype of the thread functions */
void fibonacci(int, int);
void factorial(int, int);
void bank_operation(int, int);

/*
 * This function turns stdin, stdout, and stderr into unbuffered I/O, so:
 *   - you see everything your program prints in case it crashes
 *   - the program behaves the same if its stdout doesn't connect to a terminal
 */
void unbuffered_io() {
    setbuf(stdin, NULL);
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
}

/*
 * Initializes the signal masks and the signal handler.
 */
void init_signal() {
    /* initialize the signal masks */
    sigemptyset(&base_mask);
    sigaddset(&base_mask, SIGTSTP);
    sigaddset(&base_mask, SIGALRM);
    sigemptyset(&tstp_mask);
    sigaddset(&tstp_mask, SIGTSTP);
    sigemptyset(&alrm_mask);
    sigaddset(&alrm_mask, SIGALRM);

    /* initialize the signal handlers */
    signal(SIGTSTP, sighandler);
    signal(SIGALRM, sighandler);

    /* block both SIGTSTP and SIGALRM */
    sigprocmask(SIG_SETMASK, &base_mask, NULL);
}

/*
 * Threads are created nowhere else but here.
 */
void init_threads(int fib_num, int *fib_args, int fact_num, int *fact_args, int bank_num, int *bank_args) {
    char padding[64];

    if (fib_num > 0){
        for(int i=0 ; i<fib_num ; i++)
            thread_create(fibonacci, i+10, fib_args[i]);
    }
    
    if (fact_num > 0){
        for(int i=0 ; i<fact_num ; i++)
            thread_create(factorial, i+20, fact_args[i]);
    }

    if (bank_num > 0){
        for(int i=0 ; i<bank_num ; i++)
            thread_create(bank_operation, i+30, bank_args[i]);
    }
}

/* 
 * Calls the scheduler to begin threading.
 */
void start_threading() {
    alarm(timeslice);
    scheduler();
}

void init_bank() {
    bank.balance = 500;
    bank.lock_owner = -1;
}

int main(int argc, char *argv[]) {
    if (argc <= 4) {
        printf("usage: %s [timeslice] [fib_num] [fib_args]... [fact_num] [fact_args]... [bank_num] [bank_args]...\n", argv[0]);
        exit(1);
    }
    timeslice = atoi(argv[1]);

    int fib_num, fact_num, bank_num;
    int fib_args[MAXARGS], fact_args[MAXARGS], bank_args[MAXARGS];

    int cur = 2;
    fib_num = atoi(argv[cur]);
    for(int i=0 ; i<fib_num ; i++)
        fib_args[i] = atoi(argv[i+cur+1]);
    
    cur += fib_num + 1;
    fact_num = atoi(argv[cur]);
    for(int i=0 ; i<fact_num ; i++)
        fact_args[i] = atoi(argv[i+cur+1]);
    
    cur += fact_num + 1;
    bank_num = atoi(argv[cur]);
    for(int i=0 ; i<bank_num ; i++)
        bank_args[i] = atoi(argv[i+cur+1]);

    unbuffered_io();
    init_signal();
    init_bank();
    init_threads(fib_num, fib_args, fact_num, fact_args, bank_num, bank_args);
    start_threading();
}
