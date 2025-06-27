B11902038 鄭博允

---

> Reference :
> [1] https://stackoverflow.com/questions/36588387/what-happens-to-threads-when-exec-is-called
> [2] https://stackoverflow.com/questions/1401359/understanding-linux-proc-pid-maps-or-proc-self-maps
> [3] https://man7.org/linux/man-pages/man2/mmap.2.html


# 1.

**reason 1 :**
parent 和 child process 的記憶體空間**幾乎**是獨立的，因此 child process 在運行上有較多的獨立性，安全性較高，出錯時也較不會直接影響其他 process。
然而如果使用 threads，同個 process 底下的 threads 共享許多記憶體空間，這使得 threads 在運行時較容易互相影響，存取共享記憶體的情況會更加頻繁。

**reason 2 :**
在平行處理任務時，使用 `fork()` 可能會較理想，因為不同 process 可以執行完全不同的程式，使得實作較為容易，也能處理更複雜的工作。
相較之下，如果使用 threads，因為它們共享記憶體位址以及大部分的資源，我們可能需要額外的同步機制，來避免 race condition 等問題，這會大大的增加實作難度。

# 2.

Ans : **(a)** 

- **(b)** The signal dispositions are <span style="color:red">per-process</span>.
- **(c)** A call to any exec function from a process with more than one thread shall result in <span style="color:red">all threads being terminated</span> and the new executable image being loaded and executed.
- **(d)** Not all thread-safe functions are reentrant.
- **(e)** Mutex with attribute `PTHREAD_PROCESS_SHARED` can be access by multiple process.


# 3.

Ans : **(a), (d), (e)**

- **(b)** For copy-on-write, only the modified region will be copied, and processes can still shared unchanged region.
- **(c)** one should specify <span style="color:red">MAP_SHARED</span> in the flags passed to `mmap()`.

# 4.

\*以下為在工作站(ws1)上運行的結果

`mmap()` 之後的 `/proc/<PID>/maps`:

![[School/Course Homeworks/System Programming/png/Pasted image 20231214121937.png]]

我們建出來的 memory region 應該位於第 14 行，
可以發現 address 為 `7f4abe1dd000-7f4abe1de000`，大小是 4096 bytes，
和我們當初想要的 1024 bytes 有落差。
這是因為 `mmap()` 會將該 region 對齊 page size，也就是我們申請的 region  size 會被調整為 page size 的整數倍。
而在 Linux 裡面 page size 為 4096 bytes，因此我們申請的 region size 就變成 4096 bytes 了。 

<div style="page-break-after: always;"></div>

program source code :
```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>

int main() {
    printf("pid: %d\n", getpid());

    void *mem_ptr = mmap(NULL, 1024, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    while(1){
        // just to hold the process
    }

    return 0;
}
```