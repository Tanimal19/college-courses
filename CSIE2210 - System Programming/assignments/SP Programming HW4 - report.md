B11902038 鄭博允

---

# 1. Reference
(順序不代表任何意義)
**Programming part**
[1] https://blog.csdn.net/computerme/article/details/52421928
[2] https://hpc-tutorials.llnl.gov/posix/example_using_cond_vars/
**Report part**
[1] https://stackoverflow.com/questions/556405/what-do-real-user-and-sys-mean-in-the-output-of-time1
[2] https://en.wikipedia.org/wiki/ABA_problem
[3] https://blog.csdn.net/nirendao/article/details/114682631

<br>

# 2.

我認為這實作方式有關，因此我想先提一下我的實作方式 :
(以下將非 main thread 的 thread 稱為 **worker**)

在此實作方式下，每個 worker 每次從 queue 獲取 job 確實都是依照 FIFO 的順序，但由於使用 condition variable 無法決定喚醒的 worker，因此拿到 job 的情況可能有以下兩種 :
-  A get job 0 -> B get job 1 -> A get job 2
此時輸出為 : 0 -> 1 -> 2
- B get job 0 -> A get job 1 -> A get job 2
此時輸出為 : 1 -> 0 -> 2

歸根究柢，如果要控制獲取 job 的 thread，我們就要想辦法不讓 condition variable 有決定的空間，也就是說，每次 `signal(cond)` 的時候只有一個 worker 在 waiting。
我們可以透過在 `tpool_add()` 之後加上 `sleep()` 來等待一小段時間，確保 thread 可以交替獲取 job。

<br>

# 3.

以下是我的程式使用到的 mutex 以及 cond :
```
pthread_mutex_t *wait_lock;
pthread_cond_t  *wait_cond;
pthread_mutex_t *queue_lock;
```

`wait_lock` 和 `wait_cond` 用來喚醒 waiting 的 worker
`queue_lock` 用來確保 job queue 不會同時被多個 threads 修改

以下為使用到的函數 :
- `pthread_mutex_lock()` : 對 mutex 上鎖，block 直到成功為止
- `pthread_mutex_unlock()` : 對 mutex 解鎖
- `pthread_cond_wait()` : block wait cond
- `pthread_cond_signal()` : unblock 至少一個被 cond block 住的 threads
- `pthread_cond_broadcast()` : unblock 所有一個被 cond block 住的 threads

<br>

# 4.

在 main thread 裡面，accept submission 的部分
使用 `pthread_mutex_lock(queue_lock)` 來避免 busy-waiting 的情況，
因為它會 block 直到能夠存取 queue 為止，而不是持續確認。
而 wait for worker terminate 的部分，
則是使用 `pthread_join()` 來 block wait 直到 worker terminate 為止。

在 worker threads 裡面，獲取 jobs 一樣也是使用 `pthread_mutex_lock(queue_lock)` 來避免 busy-waiting 的情況。

<br>

# 5.

M: 500
![[Pasted image 20231210184321.png | 600]]
單位 : second (s)

|           | 1     | 2     | 5     | 10    | 20    | 25    | 50    | 75    | 100   | 200   |
| --------- | ----- |:----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| real time | 0.051 | 0.033 | 0.022 | 0.023 | 0.024 | 0.028 | 0.022 | 0.037 | 0.050 | 0.061 |
| user time | 0.010 | 0.014 | 0.013 | 0.008 | 0.015 | 0.006 | 0.016 | 0.010 | 0.012 | 0.021 |
| sys time  | 0.013 | 0.007 | 0.007 | 0.015 | 0.014 | 0.027 | 0.017 | 0.043 | 0.057 | 0.069 |

首先我們要了解 real time, user time, sys time 是什麼
- **real time**: 代表整個 program 從開始到結束的時間 (包含 blocking 的時間)
- **user time**: 代表 process 在 user mode 下使用 CPU 的時間 (不包含其他 process 和 blocking 的時間)
- **sys time**: 代表 process 在 kernel 內使用 CPU 的時間

因此整個 program 實際上使用 CPU 的時間應該是 user time + sys time，
而 real time - user time 就大約是該 program 被 blocking 的時間。

我們從圖表可以發現，當 n 很小或很大的時候，real time 和 user time 差距較大，代表整個 process blocking 的時間較長。這應該是因為當 worker threads 太少的時候，`tpool_wait()` 會需要 block 較長時間來等待所有 jobs 全部完成；而 worker threads 太多的時候，則會在處理 mutex 的時候 block 較久 (因為要等很久才輪到自己獲得鎖)。

另外，我們可以發現 sys time 幾乎是隨著 n 增加也在增加，這可能是因為當 worker threads 數量越多時，我們會更頻繁的做 context switch 來在不同 threads 間切換，導致 sys time 越大。

<br>

# 6.

ABA problem 指的情況如下
1. thread A 從 shared memory 讀到 1
2. thread A 因為某些原因被暫停，context switch 到 thread B
3. thread B 對 shared memory 寫入 2
4. thread B 再次對 shared memory 寫入 1
5. thread B 因為某些原因被暫停，context switch 到 thread A
6. thread A 再次從 shared memory 讀到 1

因此，這樣會造成 thread A 忽略 shared memory 被更改的事實，導致後續的程式可能出現問題。

我的程式應該是可以避免 ABA problem，因為任何 thread 要對 pool (shared memory) 裡面的值做更改時，都需要獲得對應的 mutex。並且我對 jobs 使用的是 queue 資料結構，即便有 thread A 正要拿取 job，卻被 thread B 先拿走，接著 `tpool_add()` 又將新的 job 加入，由於新的 job 是加在尾端，thread A 還是可以正確的知道拿到 job 和之前不同。