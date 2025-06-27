B11902038 鄭博允

## 基本設定
每個 service 都有一個 struct `self` 來儲存自己的資訊，並且用 linked list 的方式來記錄 child 的資訊，每個 child node 裡面會儲存 child 的資訊。使用 linked list 的好處有兩點 :
- 每次 spawn 新的小孩時，就會接在 list 的尾端，確保我們 traverse 的順序是從舊到新
- 每次 kill 小孩時，只要將小孩前後接起來就好，順序不會被更動

---

## 指令處理流程
由於每個指令都要由指定的 service 來執行 (e.g. `spawn Manager A` 要由 `Manager` 執行),
因此在真正執行指令之前，要先 traverse 來尋找 target service，具體而言，一個 service 處理指令的流程如下：
```c
if(self is target){
	/* execute command */
	write() "success" to parent;
} else {
	/* traverse: */
	cur = child_list.head;
	result = 0;
	while(cur != NULL){
		write() command to cur.write_fd;
		read() respond from cur.read_fd;
		if(respond == "success"){
			result = 1;
			break;
		}
		cur = cur->next;
	}
	
	/* deal with traverse result */
	if(result == 1){
		write() "success" to parent;
	} else {
		write() "failed" to parent;
	}
}	
```
根據不同指令，回傳給 parent 的訊息可能不同。以 `kill` 為例，就會回傳 kill 的 child 數量。

---

## 控制執行順序
為了讓輸出順序正確，我們需要確保 service 之間的執行順序，而我利用的是 pipe 的 block 機制，也就是 `read()` 一個空的 pipe 會 blocking。
1. 例如在 traverse 的時候，我們在每一輪在 write() 後會接著一個 read() 來等待 child 回傳訊息，而不會直接跑到下一輪去呼叫下一個 child。
2. 以 `spawn` 為例，為了確保 Manager 會在 service2 建立後才 output，我讓 service1 在 `fork()` service2 後直接去 `read()` service2，因此 service1 會 block 直到 service2 建立並回傳初始訊息，接著 service1 才會往上傳成功訊息給 Manager。
3. 而也有一些比較麻煩的情況，像是 `exchange` 的 target 有兩個，我們沒辦法讓其他 service blocking 直到 target1 做完，因為它有可能就是另一個 target2。因此我們在第一次尋找 target 的 traverse 只會通知 target「可以做 `exchange` 了」並立即回傳，不會等到他們做完 `exchange` 才回傳。而之後會再由 Manager 發送另外一個 finish command 來確認兩個 target 是否已經完成 `exchange`。順帶一提，FIFO 都是由 Manager 來建立與刪除。






