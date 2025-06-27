B11902038 資工一 鄭博允

---

# Network Administration

## 看個影集也會不小心洩漏密碼？！

### 1. 
封包攔截
![[School/Course Homeworks/NASA/assets/nasa hw1/wireshark1.png|675]]
<br>

右下角放大圖，可以看到輸入的帳號及密碼
![[School/Course Homeworks/NASA/assets/nasa hw1/wireshark2.png|525]]

### 2. 
<mark style="background: #FFF3A3A6;">無法找到含有帳密的封包</mark>，由於該網站採用 HTTPS，傳輸時會採用 TLS 等加密方式來加密。我們雖然可以攔截到以 TLS 方式傳送的封包，但因為內容經過加密，所以無法看出帳密是什麼。
<br>
<br>

## 爆炸好快樂

> Reference
> https://hackmd.io/@Q3RbyuA6R5uOnWh9lFegaQ/HkgLb4Vj4?type=view

<br>

先用內建的搜尋工具，在 `Package details` 內尋找 "mostimportant" 字串
( 在 `Package bytes` 內尋找會出現很多符合的封包，所以應該不是答案 )

![[School/Course Homeworks/NASA/assets/nasa hw1/wireshark3.png]]

找到了唯一一個符合的封包
![[School/Course Homeworks/NASA/assets/nasa hw1/wireshark6.png]]

接著查看封包內容，發現裡面有著 jpg (jpeg) 檔
![[School/Course Homeworks/NASA/assets/nasa hw1/wireshark4.png]]

將封包以 Hex Dump 的形式複製到文字編輯器 (sublime text)
利用 ctrl + shift + L 將前面的行號去除

上網查詢得知 jpg 檔的編碼格式為
``` C
FF DB FF ......
...............
......... FF D9
```

複製到十六進位編輯器 (HxD)，並去除前面及後面不必要的資訊
![[School/Course Homeworks/NASA/assets/nasa hw1/wireshark5.png]]

最後將檔案重新命名為 jpg 檔，打開後得到隱藏資訊
![[School/Course Homeworks/NASA/assets/nasa hw1/gift.jpg|325]]
<br>
<br>

## 好玩遊戲每天要玩

我一開始還真沒頭緒，去查了 DNS data exfiltration 到底是什麼，
大致上就是將訊息隱藏在 DNS 封包裡面，問題是網路上的藏法超級多種，
為此我還去查了 DNS 的封包格式，結果我找了好久沒找到。

突然我發現了這些封包的傳送其實是有規律的，
都是依照著
``` c
google.com -> youtube.com -> x.sao-p.net -> x.bocchi.rock
```
的順序進行查詢
接著我就發現了 `x.sao-p.net` `x.bocchi.rock` 的頭字母一直在變

利用 display filter 對 dns query 的內容進行過濾
```c
dns.qry.name matches "sao"
```
![[School/Course Homeworks/NASA/assets/nasa hw1/dns1.png]]

由於回傳的 dns response 封包會阻礙閱讀
後面再加上指定 destination ip 的過濾條件
```c
dns.qry.name matches "sao" and ip.dst = 8.8.8.8 
```
![[School/Course Homeworks/NASA/assets/nasa hw1/dns2.png]]

然後就可以發現隱藏的訊息為
`FLAG{5u74_8a_5u70_5U70r1__mumumumumumumumu}`

至於 bocchi 的訊息也是用同樣的方式來過濾 
![[School/Course Homeworks/NASA/assets/nasa hw1/dns3.png]]

會得到
`NOTFLAG{rrrrrrrRRRRRRRrrrrrrrRRRRRRRrrrrrr}`
<br>
<br>

## 這麼多的網路協定要是能全部都認識的話該有多好

```ad-sim
title:ICMP request & reply

> Reference
> https://ithelp.ithome.com.tw/articles/10301220

ICMP request 
![[icmp request.png]]

ICMP reply
![[icmp reply.png]]

<mark style="background: #FFF3A3A6;">第三層(網路層)協定</mark>。

ICMP 是一種「 錯誤偵測與回報機制 」，需要依靠 IP 來進行傳送，
當 IP 封包在傳送時偵測發生錯誤，便會將它轉給 ICMP，
接著 ICMP 會將錯誤訊息回報給原傳送端。

舉例來說，當使用 `ping` 指令時，實際上是發出一個 ICMP request，
對方如果接收到則會回傳一個 ICMP reply。
```


```ad-sim
title:DNS query & response
> Reference
> https://aws.amazon.com/tw/route53/what-is-dns/

DNS query
![[dns query.png]]

DNS response
![[dns response.png]]

<mark style="background: #FFF3A3A6;">第五層(應用層)協定</mark>。

DNS 負責 domain 和 IP address 之間的轉換。
當使用者發出 DNS query 給 DNS server 後，DNS server 會回傳 DNS response (封包內含對應的 domain 或 IP address) 給使用者，即為 。
```


```ad-sim
title:ARP request & reply

> Reference
> https://blog.downager.com/2013/07/03/%E7%B6%B2%E8%B7%AF-%E6%B7%BA%E8%AB%87-ARP-Address-Resolution-Protocol-%E9%81%8B%E4%BD%9C%E5%8E%9F%E7%90%86/

ARP request
![[arp request.png]]

ARP reply
![[arp reply.png]]

<mark style="background: #FFF3A3A6;">第三層(網路層)協定</mark>。

ARP 被用來以 IP address 來查詢對應的 MAC address。
當使用者要透過 IP address 來查詢某裝置的 MAC address 時，會發送 ARP request 廣播到所屬網段，該網段的裝置接收到廣播後，如果要查詢的 IP address 和自己一樣，則回傳 ARP reply (封包內含該裝置的 IP / MAC address)；否則不予理會。 

```
 

```ad-sim
title:DHCP Discover & Offer & Request & Ack

> Reference
> https://zh.wikipedia.org/zh-tw/%E5%8A%A8%E6%80%81%E4%B8%BB%E6%9C%BA%E8%AE%BE%E7%BD%AE%E5%8D%8F%E8%AE%AE

DHCP Discover
![[dhcp discover.png]]

DHCP Offer
![[dhcp offer.png]]

DHCP Request
![[dhcp request.png]]

DHCP Ack
![[dhcp ack.png]]

<mark style="background: #FFF3A3A6;">第五層(應用層)協定</mark>。

DHCP 負責在使用者與伺服器之間協議動態 IP 位址的分配。
DHCP 的運作分成四個步驟：

當使用者要連上網路，卻發現沒有指定靜態 IP 時，會發出 **DHCP Discover** 廣播到網路上，詢問 DHCP 伺服器在哪裡。
↓
當 DHCP Server 收到 DHCP Discover 後，會尋找空閒的 IP 並包裝成 **DHCP Offer** 回傳給使用者。
↓
使用者收到 DHCP Offer 後，發出 **DHCP Request** 廣播告知該 DHCP server 以接受這個 offer；同時告知同網段的其他 DHCP server 它已經接受了一個 offer，讓其他 DHCP server 收回原來的 offer 以提供給其他使用者。
↓
當 DHCP Server 收到 DHCP Request 後，會回應一個 **DHCP Ack** 給使用者，其中包含租期以及其他所有組態資訊，告知使用者可以使用該 IP。
```

---

# System Administration

## UNIX Basics
### I. Basic Permissions

> Reference
> https://ithelp.ithome.com.tw/articles/10218228

```ad-command
title: 權限 

| 權限 | 檔案   | 目錄                              |
|:---- |:------ |:--------------------------------- |
| r    | 可讀取 | 可 `ls`                           |
| w    | 可修改 | 可 `mv` `rm` `cp` 目錄內檔案 |
| x    | 可執行 | 可 `cd`                           |
```

1. **ls dir1**
	<font color="#BE7AA0"> **FALSE** </font>
	`ls` 指令需要 `r` 權限，由於對 `dir1` 只有 `x` 權限，
	故此題指令不可執行。

2. **ls dir1/dir1A**
	<font color="#7AA0BE"> **TRUE** </font>
	可以想成是 `cd` 進去 `dir1` 後再 `ls dir1A`
	因為對 `dir1` 有 `x` 權限所以可以進到目錄內，
	而對於 `dir1A` 有 `r` 權限所以可以查看目錄內容。

4. **cd dir1**
	<font color="#7AA0BE"> **TRUE** </font>
	如前所述，有 `x` 權限即可 `cd`。

6. **cd dir1/dir1A**
	<font color="#BE7AA0"> **FALSE** </font>
	由於對 `dir1A` 沒有 `x` 權限所以無法 `cd`。

8. **echo 'howdy' > dir1/dir1A/file1**
	<font color="#BE7AA0"> **FALSE** </font>
	上述指令是對 `echo 'howdy'` 的輸出進行 `>` 重定向到 `file1` ，因此需要有 `w` 修改檔案的權限。 
	雖然對 `file1` 有 `w` 權限可以對其進行修改，但是對於 `dir1A` 沒有 `w` 權限，所以不能對其內部的檔案(也就是 `file1` )進行修改。

9. **echo 'howdy' > dir1/dir1B/file2**
	<font color="#BE7AA0"> **FALSE** </font>
	與上題相反，雖然對 `dir1B` 有 `w` 權限可以對其內部的檔案進行修改，但是對於 `file2` 沒有 `w` 權限所以不能進行修改。

10. **echo 'howdy' > dir2/file3**
	<font color="#7AA0BE"> **TRUE** </font>
	對於 `dir2` `file3` 都有 `w` 權限所以可以進行修改。

8. **echo 'howdy' > dir2/link1**
	<font color="#BE7AA0"> **FALSE** </font>
	```ad-command
	title: Symbolic link
	`symbolic link` 是一種檔案類型，有點類似捷徑的概念，可以連結到任意檔案或目錄。
	
	可以用 `ln -s <linked file> <link>` 來建立連結。	
	 
	對於連結所做的操作，等同於直接操作源檔案，
	因此若是更改連結的權限 ( `chmod` )，等同於更改源檔案的權限，而該連結的權限則不會被更改。
	```
	
	此操作等同於 `echo 'howdy' > dir1/dir1B/file2`。
	即便對 `link1` 具有 `rwx` 權限，但是對於 `file2` 沒有 `w` 權限所以不能進行修改。

9. **rm dir1/dir1B/file2**
	<font color="#7AA0BE"> **TRUE** </font>
	```ad-command
	title: rm
	要 `rm` 一個檔案與對該檔案的權限無關，
	而是與該檔案所在目錄的 `w` `x` 權限有關。
	( 要能夠"進去"目錄並"修改"目錄中的檔案 ) 
	```
	因為對於 `dir1B` 具有 `wx` 權限，所以可以 `rm` 裡面的檔案 ( 也就是 `file2` )。
	由於對於 `file2` 沒有 `w` 權限，要移除檔案時，系統會告知
	`remove write-protected regular empty file 'file2'?`
	此時若回答 `y` 則會成功移除檔案。

10. **rm dir3/link2**
	<font color="#7AA0BE"> **TRUE** </font>
	此題指令只會對 `link2` 做 `rm` 而已，而非對 `file2`，而對於連結所在目錄 `dir3` 具有 `wx` 權限，所以此題指令可成功執行。

<br>

### II. ACL

> Reference
> https://officeguide.cc/linux-acl-access-control-list-setfacl-getfacl-command-tutorial/

1. 選擇 `b11902113` 作為好朋友
2. 建立目錄 `chatroom/`

	```batch
	mkdir chatroom/
	```
	移除 `group` `other` 的所有權限
	```batch
	chmod 700 chatroom/
	```
	賦予 `b11902113` 所有權限 (可以`ls`、`cd`、新增與刪除檔案)
	```batch
	setfacl -m u:b11902113:rwx chatroom/
	```
	當前 `chatroom/` 權限為
	![[School/Course Homeworks/NASA/assets/nasa hw1/chatroom1.png]]
	
	避免 `b11902113` 無法進入目錄 `b11902038/`
	確認當前目錄 ( `b11902038/` ) 的權限
	![[School/Course Homeworks/NASA/assets/nasa hw1/b11902038-1.png|500]]
	不須修改

3. 使 `b11902113` 的 `w` 權限繼承
	```batch
	setfacl -m d:u:b11902113:w chatroom
	```
	當前 `chatroom/` 權限為
	![[School/Course Homeworks/NASA/assets/nasa hw1/chatroom2.png]]

4. 設立 acl mask，使所有人 (除了自己) 的最高權限為 `r--`
	```batch
	setfacl -m m:r chatroom
	```
	再另外移除自己的 `wx` 權限
	```batch
	chmod u-wx chatroom
	```
	當前 `chatroom/` 權限為
	![[School/Course Homeworks/NASA/assets/nasa hw1/chatroom3.png]]

<br>

### III. Linux / Unix 雜項

> Reference (照題號分)
> [P1]
> https://linux.vbird.org/linux_basic_train/centos7/unit05.php
> [P2]
> https://www.hy-star.com.tw/tech/linux/permission/permission.html#sbit
> [P3]
> https://ithelp.ithome.com.tw/articles/10157552
> [P5]
> https://babygoat.github.io/2019/04/25/journalctl%E4%BD%BF%E7%94%A8%E7%AD%86%E8%A8%98/
> [P6]
> https://blog.csdn.net/Solomon1558/article/details/51763751
> [P7]
> https://magiclen.org/vimrc/

1. SUID 全名為 Set User ID，只針對執行檔起作用，
	可以讓使用者在執行該程式的過程中具備"該程式擁有者" 的權限。
	( 使用者必須對該程式有 `x` 權限 )

2. `tmp2/` 的權限為：
	```vim
	drwxrwxrwt 35 root root 4096
	```
	可以發現 `other` 的 `x` 權限被改為 `t` (sticky) 權限了，
	當目錄被設定 `t` 權限時，目錄下的檔案只有擁有者可以刪除或改名。

1.  
	Debian 以穩定為主，但因此版本更新不頻繁；
	Arch 採取滾動更新，所有安裝都要一步步打指令，對新手不友善。

4. Alpine Linux 是一個基於 BusyBox 工具集的 Linux 發行版。

5. 
	使用指令
	```batch
	journalctl -b -1 -p warning -x
	```
	`-b` 列出第幾次開機的日誌 ( `-1` 就是前一次 )
	`-p` 列出指定級別以上的紀錄，共有 8 個級別如下：
	```c
	0 emerg / 1 alert / 2 crit / 3 err
	4 warning / 5 notice / 6 info / 7 debug
	```
	`-x` 列出額外的解釋訊息 ( 不適用所有類型的訊息 )

6. 
	`~/.bashrc`：包含個別使用者的 shell 資訊，在使用者登入以及開啟新的 shell 時被讀取並執行。
	
	`~/.bash_profile`：同樣包含個別使用者的 shell 資訊，默認情況下會直接引用 `~/.bashrc` 的資料，在使用者登入時被讀取並執行**一次**。

7. (a) 設定行號
	```vim
	set number
	```
	(b) 游標行 highlight
	
	`cterm` 調整字體：`none` `bold` `underline` `reverse`
	`ctermbg` 調整背景顏色， `ctermfg` 調整文字顏色
	
	```vim
	hi CursorLine cterm=none ctermbg=none ctermfg=Magenta	
	```
	(c) 自動縮排
	```vim
	set ai
	```
	(d) 啟動游標
	```vim
	set mouse=a
	```
	 (e) 用 4 個空白鍵取代 tab
	```vim
	set tabstop=4
	```

<br>

## Shell Scripting

> Reference:
> https://en.wikipedia.org/wiki/Digital_Signature_Algorithm
> https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
> https://en.wikipedia.org/wiki/Modular_multiplicative_inverse
> https://unix.stackexchange.com/questions/365510/how-to-avoid-wrapping-in-bc-output
> https://blog.gtwang.org/linux/linux-bc-command-tutorial-examples/
> https://gist.github.com/jimratliff/d735a57eef05b650d4a17f10b7da64d9
> http://www.herongyang.com/Cryptography/DSA-Introduction-Algorithm-Illustration-p23-q11.html
> https://blog.csdn.net/huangjin0507/article/details/45045537