#include "kernel/types.h"
#include "user/setjmp.h"
#include "user/threads.h"
#include "user/user.h"
#define NULL 0

static struct thread* current_thread = NULL;
static int id = 1;
static jmp_buf env_st;

struct thread *thread_create(void (*f)(void *), void *arg){
    struct thread *t = (struct thread*) malloc(sizeof(struct thread));
    unsigned long new_stack_p;
    unsigned long new_stack;
    new_stack = (unsigned long) malloc(sizeof(unsigned long)*0x100);
    new_stack_p = new_stack +0x100*8-0x2*8;
    t->fp = f;
    t->arg = arg;
    t->ID  = id;
    t->buf_set = 0;
    t->stack = (void*) new_stack;
    t->stack_p = (void*) new_stack_p;

    t->hastask = 0;
    t->cur_task = NULL;
    t->prior_task = NULL;

    id++;
    return t;
}
void thread_add_runqueue(struct thread *t){
    if (current_thread == NULL) {
        // TODO
        current_thread = t;
        current_thread->previous = t;
        current_thread->next = t;
    }
    else {
        // TODO
        current_thread->previous->next = t;
        t->previous = current_thread->previous;

        current_thread->previous = t;
        t->next = current_thread;
    }
}
void thread_yield(void){
    // TODO
    int val;
    if (current_thread->cur_task == NULL) {
        val = setjmp(current_thread->env);
    }
    else {
        val = setjmp(current_thread->cur_task->env);
    }

    if (val == 0) {
        schedule();
        dispatch();
    }
    else {
        return;
    }
}
void dispatch(void){
    // TODO
    while (current_thread->hastask > 0) {
        current_thread->cur_task = current_thread->prior_task;

        if (current_thread->cur_task->buf_set == 0) {
            current_thread->cur_task->buf_set = 1;

            if (setjmp(current_thread->cur_task->env) == 0) {
                current_thread->cur_task->env->sp = (unsigned long)current_thread->cur_task->stack_p;
                longjmp(current_thread->cur_task->env, 1);
            }

            current_thread->cur_task->fp(current_thread->cur_task->arg);
            task_exit();

            if(current_thread->prior_task != NULL){
                current_thread->prior_task = current_thread->prior_task->next;
            }
        }
        else {
            longjmp(current_thread->cur_task->env, 1);
        }
    }
    current_thread->cur_task = NULL;
    current_thread->prior_task = NULL;

    if (current_thread->buf_set == 0) {
        current_thread->buf_set = 1;

        if (setjmp(current_thread->env) == 0) {
            current_thread->env->sp = (unsigned long)current_thread->stack_p;
            longjmp(current_thread->env, 1);
        }

        current_thread->fp(current_thread->arg);
        thread_exit();
    }
    else {
        longjmp(current_thread->env, 1);
    }
}
void schedule(void){
    // TODO
    current_thread = current_thread->next;
}
void thread_exit(void){
    if (current_thread->next != current_thread) {
        // TODO

        current_thread->previous->next = current_thread->next;
        current_thread->next->previous = current_thread->previous;

        struct thread* over = current_thread;
        schedule();

        free(over->stack);
        free(over);

        dispatch();
    }
    else {
        // TODO
        // Hint: No more thread to execute
        free(current_thread->stack);
        free(current_thread);
        current_thread = NULL;
        longjmp(env_st, 1);
    }
}
void thread_start_threading(void){
    // TODO
    int val = setjmp(env_st);
    if (val == 0) {
        dispatch();
    }
    else {
        return;
    }
}
// part 2
void thread_assign_task(struct thread *t, void (*f)(void *), void *arg){
    // TODO
    struct task *a = (struct task*) malloc(sizeof(struct task));
    unsigned long new_stack = (unsigned long) malloc(sizeof(unsigned long)*0x100);
    unsigned long new_stack_p = new_stack +0x100*8-0x2*8;

    a->fp = f;
    a->arg = arg;
    a->buf_set = 0;
    a->stack = (void*) new_stack;
    a->stack_p = (void*) new_stack_p;
    
    if (t->hastask == 0) {
        a->previous = NULL;
        a->next = NULL;
    }
    else {
        a->previous = NULL;
        t->prior_task->previous = a;
        a->next = t->prior_task;
    }

    t->prior_task = a;
    t->hastask ++;
}
void task_exit(void){
    struct task* prev = current_thread->cur_task->previous;
    struct task* nex = current_thread->cur_task->next;

    if (prev != NULL) prev->next = nex;
    if (nex != NULL) nex->previous = prev;

    free(current_thread->cur_task->stack);
    free(current_thread->cur_task);

    current_thread->hastask--;
}
