B11902038 鄭博允

---

# 1.

有可能在呼叫 `setjmp(env_alrm)` 之前，`alarm(60)` 就已經到期並觸發 *SIGALRM*，接著 sig_alrm 便會呼叫 `longjmp(env_alrm)`，但 env_alrm 根本還沒有被設定。

# 2.

```c
jmp_buf     env;
static int  count;

void sig_alrm(int signo){
	sigset_t newmask, oldmask;
	sigemptyset(&newmask);
	sigaddset(&newmask, SIGALRM);
	/* block SIGALRM */
	sigprocmask(SIG_BLOCK, &newmask, &oldmask);

	count++;
	printf("count: %d\n", count);

	longjmp(env, 1);
}

void count_to_num(int num){
	signal(SIGALRM, sig_alrm);
	count = 0;
	setjmp(env, 1);
	
	if(count < num){
		alarm(1);
		pause();
	}
	exit(0);
}
```

上面是一個計數函數，<mark style="background: #FFF3A3A6;">假設這裡不考慮 race condition 的發生 (*SIGALRM* 在 `pause()` 之前就被 delivered)</mark>，則預期輸出如下 :
```
> count_to_num(5)
count: 1
count: 2
count: 3
count: 4
count: 5
```

然而，由於我們使用的是 `setjmp()` 和 `longjmp()`，在 sig_alrm 執行 `longjmp()` 跳回來時，signal mask 沒有被復原，而依舊把 *SIGALRM* block 住，因此 sig_alrm 實際上只會執行一次，接著程式就會因為 `pause()` 收不到 *SIGALRM* 而卡死。

因此在這裡我們必須使用 `sigsetjmp()` 和 `siglongjmp()`，從 sig_alrm 跳回 count_to_num 時 signal mask 才會被復原，不會 block 住 *SIGALRM*。

<br>
<div style="page-break-after: always;"></div>

# 3.

原先在 slide 44 裡面，之所以會發生問題是因為 *SIGALRM* 有可能在 `pause()` 之前就被 delivered，而導致 `pause()` 收不到 *SIGALRM* 而卡死。

在 slide 45 裡面，呼叫 `alarm(seconds)` 之前，我們先將 signal mask 調整成將 *SIGALRM* block 住，因此這之後直到呼叫 `sigsuspend(&suspmask)` 時，*SIGALRM* 都不會被 delivered。而 `sigsuspend(&suspmask)` 則會將 signal mask 復原(不會 block *SIGALRM*)，因此 *SIGALRM* 會等到這時候才被 deliverd，並使 `sigsuspend(&suspmask)` 返回。

<br>

# 4.

## (1)

1. 建立兩個 pipe : 
	fd1 : parent 寫 child 讀
	fd2 : parent 讀 child 寫

2. 接著將 `sig_pipe()` 註冊成 *SIGPIPE* 的 handler

3. `fork()` 出一個 child process
	
4. parent 和 child 執行不同工作
	<mark style="background: #ADCCFFA6;">parent</mark> : 
	一旦從 stdin 讀取到內容，就寫入 fd1[1] 給 child。
	然後再讀取 fd2[0] 來獲取 child 寫入的訊息，並輸出到 stdout。
	並重複上述動作直到 child 把 fd2 關掉。
	<mark style="background: #FFB8EBA6;">child</mark> : 
	先將 fd1[0] `dup2` 到 stdin, 以及 fd2[1] `dup2` 到 stdout。
	接著 `exec()` 一個 program "test"，會從 stdin(fd1[0]) 讀入兩個數字，並將和輸出到 stdout(fd2[1])。

整體而言，這段程式的作用就是調用 "test" 讓使用者可以透過輸入兩個數字，取得它們
的和。

## (2)

manual 的定義：當一個 pipe 的所有讀取端都被關掉後，如果對該 pipe 寫入則會引起 *SIGPIPE*。 

正常情況下，由於 parent 的執行過程中並不會關掉 fd2 的讀取端，因此我們可以認為 *SIG_PIPE* 只會發生在 fd1 上面。
由於我們無法得知 child 所調用的 program "test" 是否會重複執行，我們先假設 "test" 只會執行一次(讀取兩個數字並輸出和)接著便 terminate。

當 child 所調用的 "test" 執行完第一次後便會 terminate，此時 "test" 的 stdin(也就是 fd1[0]) 會被關閉，代表 fd1 這個 pipe 的所有讀寫端都被關閉了，因此當 parent 第二次想要對 fd1[1] 做寫入時，*SIGPIPE* 便會產生，並觸發 `sig_pipe()`。
