B11902038 鄭博允

---
以下用 fd 來代表 file descriptor


# 1.
> Reference :
> https://zh.wikipedia.org/zh-tw/%E5%BF%99%E7%A2%8C%E7%AD%89%E5%BE%85
> https://zhuanlan.zhihu.com/p/367591714

  <br>

busy waiting 指的是一個 process 在反覆檢查特定條件是否為 `TRUE` 的狀態，而導致該 process 持續占用系統資源，使其他 process 無法運行。

在此 assignment 中，因為 server 要同時處裡多台 client 的請求，如果單純用 `read()` 會導致 busy waiting，因此我在 server 獲取不同 client 請求的部分利用了 `select()` 來避免 busy waiting (包括 client 獲取連線的請求)。

儘管使用 select() 或 poll()，還是有可能發生 busy waiting 的狀況。這兩個函數雖然可以同時監控多個 fds，但在 fd 就緒後，實際上對 fd 的讀寫還是 blocking 的，有可能就在這裡發生 busy waiting 的情況。

<br>

# 2.

> Reference :
> https://en.wikipedia.org/wiki/Starvation_(computer_science)

<br>

starvation 指的是在多個 process 並行的環境下，process 一直無法獲得其所需資源，而無法運行的情況。

在此 assignment 中，我認為 client 的 request 不太會發生 starvation 的情況。假使現在整個 Bulletin Board 都被 lock 了，那麼當 client 發出 `post` request 會得到錯誤、發出 `pull` request 則不會得到任何 record。不論何種， process 都可以繼續運行，不會有 starvation 的情況發生。

<br>

# 3.
當初在寫 server.c 時，我就碰上了 `fcntl` 的 `F_GETLK` 沒有反應的情況，後來我才發現，在同一個 process 內，即便對 record 上了鎖，`F_GETLK` 仍會返回可以上鎖的狀態，因為 `F_GETLK` 是檢查有沒有其他 process 對 record 上了鎖。

後來我決定用一個很原始(笨)的方法：
開一個陣列 `record_lock[RECORD_NUM]`，只要對哪個 record 上了鎖，就讓對應 index 變成 1，解鎖之後再變成 0。

將這個陣列與 `F_GETLK` 並用，一個用來檢查自己上的鎖，一個用來檢查別人上的鎖。如此一來，就可以在自己寫入之前確認有沒有人正在寫入這段 `record`，避免同時寫入造成檔案資料不完整。

更 : 我後來才在 dc 上看到可以用 `SETLK` 返回 -1 這個方法，我好ㄅ。

<br>

# 4.

我好像不小心在前一題就回答了。
在多個 process 要同時存取 BulletinBoard 時，在寫入之前先用 F_GETLK 來確認有沒有其他 process 對 record 上了鎖。如果有，就代表現在有其他 process 正在存取；如果沒有，我們就對該 record 上鎖，防止其他 process 存取。