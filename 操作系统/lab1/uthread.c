#include "uthread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_THREADS 1024  // 定义最大线程数量

struct thread_queue {
    struct uthread* threads[MAX_THREADS]; // 线程控制块数组
    int front; // 队列前端
    int rear;  // 队列后端
    int count; // 当前线程数量
};

// 初始化全局队列
struct thread_queue active_queue = { .front = 0, .rear = 0, .count = 0 };
static struct uthread* current_thread = NULL;
static struct uthread* main_thread = NULL;

/// @brief 切换上下文
/// @param from 当前上下文
/// @param to 要切换到的上下文
extern void thread_switch(struct context* from, struct context* to);

/// @brief 线程的入口函数
/// @param tcb 线程的控制块
/// @param thread_func 线程的执行函数
/// @param arg 线程的参数
void _uthread_entry(struct uthread* tcb, void (*thread_func)(void*),
    void* arg);

/// @brief 清空上下文结构体
/// @param context 上下文结构体指针
static inline void make_dummpy_context(struct context* context) {
    memset((struct context*)context, 0, sizeof(struct context));
}
int find_tcb_position(struct uthread* tcb) {
    // 遍历整个 active_queue 查找目标 tcb
    for (int i = 0; i != active_queue.rear; i = (i + 1) % MAX_THREADS) {
        if (active_queue.threads[i] == tcb) {
            return i; // 找到 tcb，返回其位置
        }
    }
    printf("errornotfound\n");
    return -1; // 没找到
}
struct uthread* uthread_create(void (*func)(void*), void* arg, const char* thread_name) {

    //         +------------------------+
    // low     |                        |
    //         |                        |
    //         |                        |
    //         |         stack          |
    //         |                        |
    //         |                        |
    //         |                        |
    //         +------------------------+
    //  high   |    fake return addr    |
    //         +------------------------+

    /*
    TODO: 在这里初始化uthread结构体。可能包括设置rip,rsp等寄存器。入口地址需要是函数_uthread_entry.
          除此以外，还需要设置uthread上的一些状态，保存参数等等。

          你需要注意rsp寄存器在这里要8字节对齐，否则后面从context switch进入其他函数的时候会有rsp寄存器
          不对齐的情况（表现为在printf里面Segment Fault）
    */
    // 初始化uthread结构体

    
    // 设置线程名字
    
    // 确保 rsp 地址 16 字节对齐
    // 计算栈的起始地址和rsp寄存器
    struct uthread* uthread = NULL;
    int ret;
    //需要修改
    //申请一块16字节对齐的内存
    ret = posix_memalign((void**)&uthread, 16, sizeof(struct uthread));
    if (0 != ret) {
        printf("error");
        exit(-1);
    }
    //以上为默认代码
    uthread->name = thread_name;
    make_dummpy_context(&uthread->context);
    long long sp = ((long long)&uthread->stack + STACK_SIZE) & (~(long long)15);
    sp -= 8; 
    uthread->context.rsp = sp; // 设置rsp为栈顶地址

    // 设置rip
    uthread->context.rip = (long long)_uthread_entry; 
    // 将参数存储到寄存器
    uthread->context.rdi = (long long)uthread; // tcb
    uthread->context.rsi = (long long)func; // thread function
    uthread->context.rdx = (long long)arg; // argument for thread function
    // 设置状态为初始化
    uthread->state = THREAD_INIT;
    // 将新创建的线程加入到活跃队列 (active_queue)
    active_queue.threads[active_queue.rear] = uthread;
    active_queue.rear = (active_queue.rear + 1) % MAX_THREADS;
    active_queue.count++;
    return uthread;
}

void schedule() {//需要修改
    /*
    TODO: 在这里写调度子线程的机制。这里需要实现一个FIFO队列。这意味着你需要一个额外的队列来保存目前活跃
          的线程。一个基本的思路是，从队列中取出线程，然后使用resume恢复函数上下文。重复这一过程。
    */
    while (1) {
        // 检查队列是否为空
        if (active_queue.count == 0) {
            return; // 返回，结束调度
        }
        // 从队列中取出一个线程控制块
        struct uthread* next_thread = active_queue.threads[active_queue.front];

        // 更新队列前端索引
        active_queue.front = (active_queue.front + 1) % active_queue.rear;
        active_queue.count--;


        if (next_thread->state == THREAD_INIT) {
            // 将状态设置为 THREAD_RUNNING 并开始执行
            next_thread->state = THREAD_STOP;
        }
        if (next_thread->state == THREAD_STOP) {
            // 将状态设置为 THREAD_RUNNING 并开始执行
            // 恢复下一个线程的上下文
            uthread_resume(next_thread);
        }
        if (next_thread->state == THREAD_SUSPENDED) {
            next_thread->state = THREAD_STOP;
        }
    }
}

long long uthread_yield() {//需要修改

    /*
    TODO: 用户态线程让出控制权到调度器。由正在执行的用户态函数来调用。记得调整tcb状态。
    */
    if (current_thread == NULL) {
        printf("错误：当前没有正在运行的线程\n");
        return -1;
    }

    int nextcb = (find_tcb_position(current_thread) + 1) % active_queue.rear;

    // 将当前线程状态设置为挂起
    current_thread->state = THREAD_SUSPENDED;

    // 从队列中取出一个线程控制块
    struct uthread* next_thread = active_queue.threads[nextcb];
    next_thread->state = THREAD_STOP;
    uthread_resume(next_thread);

    // 上下文切换回来后，恢复之前线程的状态
    next_thread->state = THREAD_RUNNING;

    // 返回0表示成功
    return 0;
}

// 打印线程信息的函数
void print_thread_info(struct uthread* thread) {
    if (thread == NULL) {
        printf("Thread is NULL.\n");
        return;
    }

    printf("Thread Name: %s\n", thread->name);
    printf("Thread State: %d\n", thread->state);
    printf("Context:\n");
    printf("  RIP: 0x%llx\n", thread->context.rip);
    printf("  RSP: 0x%llx\n", thread->context.rsp);
    printf("  RBP: 0x%llx\n", thread->context.rbp);
    printf("  RBX: 0x%llx\n", thread->context.rbx);
    printf("  R12: 0x%llx\n", thread->context.r12);
    printf("  R13: 0x%llx\n", thread->context.r13);
    printf("  R14: 0x%llx\n", thread->context.r14);
    printf("  R15: 0x%llx\n", thread->context.r15);
    printf("  RDI: 0x%llx\n", thread->context.rdi);
    printf("  RSI: 0x%llx\n", thread->context.rsi);
    printf("  RDX: 0x%llx\n", thread->context.rdx);
    printf("Stack Address: %p\n", (void*)thread->stack);
}


void uthread_resume(struct uthread* tcb) {//需要修改
    
    
    /*这里的函数，是指恢复到current_uthread
    TODO：调度器恢复到一个函数的上下文。
    */
    // 确保传入的 tcb 不是 NULL
    if (tcb == NULL) {
        printf("错误：传入的线程控制块为空\n");
        return;
    }

    // 检查线程状态是否可以恢复
    if (tcb->state != THREAD_STOP) {
        printf("错误：只能恢复停止状态的线程\n");
        return;
    }
    tcb->state = THREAD_RUNNING;
    // 更新当前线程为要恢复的线程
    struct uthread* previous_thread = current_thread;
    current_thread = tcb;

    // 切换到目标线程的上下文
    thread_switch(&previous_thread->context, &current_thread->context);
}


void thread_destroy(struct uthread* tcb) {
    free(tcb);
}

void _uthread_entry(struct uthread* tcb, void (*thread_func)(void*),
    void* arg) {

    //需要修改
    /*
    TODO: 这是所有用户态线程函数开始执行的入口。在这个函数中，你需要传参数给真正调用的函数，然后设置tcb的状态。
    */
    // 设置状态为运行中
    tcb->state = THREAD_RUNNING;

    // 调用实际的线程函数，并传入参数
    thread_func(arg);
    thread_switch(&current_thread->context, &main_thread->context);
    thread_destroy(tcb);
}

void init_uthreads() {//需要修改
    main_thread = malloc(sizeof(struct uthread));
    make_dummpy_context(&main_thread->context);
    current_thread = main_thread;
}