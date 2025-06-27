b11902038 資工三 鄭博允

---

# 1 Baby shark
## 1.
### (a)
![|650](School/Course%20Homeworks/Computer%20Networks/assets/1-1.jpeg)

### (b)
圖中的高峰大約是 $3.17$ MB/s，因此 transmit rate limit 應該是 $3.17$ MB/s = $3,170,000$ Bytes/s = $25,360,000$ bps = $25,360$ Kbps

## 2.
使用 wireshark 內建功能 IPv4 Statistics > All Addresses，發現 IP `140.112.28.152` 在所有封包的 source/destination 內的出現率是 96.20%，可見該 IP 應該就是 user IP address。 

## 3.
Transmission Control Protocol (TCP) 和 User Datagram Protocol (UDP) 兩者皆為傳輸層的通訊協定，簡單來講，TCP 是可靠的傳輸，UDP 是快速的傳輸。

TCP 會先將資料切成小封包並給予編號，再依序傳送封包。接收端會藉由編號確認收到的封包是否依照正確的順序，並發送確認信號給傳送端，此時傳送端才會發出下一個封包。反之，順序錯誤或是封包遺失，傳送端就必須重新傳送。

UDP 不會幫封包編號，以串流的方式發送封包，傳送端會連續不斷發送封包，無視順序錯誤或是封包遺失等問題。因為兩端之間的通訊很少，因此速度比 TCP 還要快，但是出錯率就比較高。

### (a)
使用 filter `tcp.port == 1080 && ip.src == 140.112.28.152`，尋找使用 tcp port 1080 並且 source 是 user IP 的封包，查看封包內容後得到訊息:
**Baby Shark, doo-doo, doo-doo, doo-doo.**

### (b)
使用 filter `udp.port == 2330 && ip.dst == 140.112.28.152`，尋找使用 udp port 2330 並且 destination 是 user IP 的封包，查看封包內容後得到訊息:
**Even a baby shark can use wire shark**

## 4.
IPv4 封包截圖:
![|600](School/Course%20Homeworks/Computer%20Networks/assets/1-2.png)

IPv6 封包截圖:
![|600](School/Course%20Homeworks/Computer%20Networks/assets/1-3.png)

可以發現兩者的 header 有部分欄位是不同的
- **封包長度**: (IPv4) Total Length、(IPv6) Payload Length
- **通訊協定**: (IPv4) Protocol、(IPv6) Next Header

## 5.
### a.
封包的 response (answer) 裡面總共有兩筆 records。
第一筆的 types 是 CNAME，用來做 domain name 的轉換，將 zh.wikipedia 轉換成其代表的標準網域 dyna.wikimedia.org
第二筆的 types 是 A，就是單純的從 hostname 轉成 IP address，回傳 dyna.wikimedia.org 的 IP address **103.102.166.224**

### b.
zh.wikipedia.org 的 IP address 是 **103.102.166.224**

---

# 2 The course permission code
## 1.
隨便點一個封包然後 Follow > TCP Stream 之後可以發現，伺服器的回應都來自 port 6000，因此 server 的 port 應該就是 **6000**。

根據 stream 的資訊來看，流程大略是
```
220 e995e0e51528 ESMTP
EHLO localhost
250-e995e0e51528 Nice to meet you, [172.17.0.1]
MAIL FROM:<AC@pang.com>
250 Accepted
RCPT TO:<thevoiceofgg3be0@csie.com>
250 Accepted
DATA
250 OK: message queued
QUIT
221 Bye
```
因此推測應該是使用 SMTP。

SMTP 常用的 server port 有 **25** 和 **587**，port 25 用於沒有安全防護的 SMTP，現今大多使用比較安全的 SMTPS 搭配 port 587。

## 2.
以下是郵件的標頭:
```
From: AC@pang.com
To: thevoiceofgg3be0@csie.com
Subject: Course Enrollment
```

## 3.
以下是郵件的內容:

Hi Chenchen,
Thank you for signing up for "Computer Network for League of Legends".
Due to the capacity of the classroom, only some of the students can get the permission code.
We are glad to tell you that you have successfully passed the course qualification check.
Here's your course permission code: "5Zyw542E6LGs5oiw6ZqK".
Please join the course before the 3rd week of this semester. Otherwise, this permission code will no longer be valid.
On the other hand, DO NOT distribute your permission code to other students
Enjoy this course and see you next week!
Best, regards.
Prof AC.

因此，學生想要加選的課程是「**Computer Network for League of Legends**」，而授權碼是 **5Zyw542E6LGs5oiw6ZqK**。

## 4.
雖然 server 提供了 `250 STARTTLS` 的選項，該選項可以使用 TLS 來加密原有的 SMTP 內容。不過 client 似乎沒有使用 STARTTLS，這也是為什麼我們可以直接看到明文的郵件內容。

沒有使用 STARTTLS 的缺點就是內容不會經過加密，只要拿到封包就能直接看到裡面的內容，造成安全性上的疑慮。

---

# 3 The problem sheet of midterm exam
## 1.
根據 Follow > TCP Stream 顯示的封包資訊:
username 是 **cnta**
password 是 **dMTy6C4UiQ4**

## 2.
根據 Follow > TCP Stream 顯示的封包資訊:
server 接收 FTP requests 的 port 是 **4000**
而 client 總共向 server 傳送了 4 個檔案，以及 server 接收該檔案的 port
- funny.jpeg - port **20005**
- mail.pcapng - port **20004**
- midterm.txt - port **20010**
- treasure.txt - port **20001**

## 3.
**treasure.txt** with **12** questions

---

# 4 The path to the destination
## 1.
![|500](School/Course%20Homeworks/Computer%20Networks/assets/1-4.png)

`traceroute` 使用了 IP 封包的 Time to live (TTL) 機制，每個 IP 封包都有一個整數的 TTL 值，每經過一個 router 都會 -1，如果直到 TTL=0 時封包都還沒有抵達目的地，則回傳 ICMP TIME_EXCEEDED 的錯誤。

因此，`traceroute` 會先向目的地發送 TTL=1 的 UDP 封包，在接收到 ICMP TIME_EXCEEDED 錯誤後 (因為第一個 router 會將 TTL -1)，就能從錯誤訊息中獲得 router 的 IP address。在收到錯誤後，`traceroute` 會將 TTL +1 並繼續發送封包，就這樣不斷增加 TTL 直到封包送達目的地為止，每一輪都會發送三個(同 TTL)封包。另外，`traceroute` 預設如果發送了 30 輪的封包都沒有到達目的地就會結束。

然而，當 UDP 封包到達目的地後，不會回傳 ICMP TIME_EXCEEDED 錯誤，因為此時已經到達目的地了。因此 `traceroute` 會將目的地的 port 設成幾乎不會被用到 (沒有人 listen) 的數字 (通常會用 33434)，接著目的地主機就會回傳一個 ICMP PORT_UNREACHABLE 錯誤。

## 2.
![|500](School/Course%20Homeworks/Computer%20Networks/assets/1-5.png)

可以發現 2. 的後面的輸出都是 `*`，並且最後並沒有成功到達目的地，這可能是因為某一個 router 的防火牆阻止了 ICMP 訊息回傳，或著是直接過濾掉了有著不常見 port 號的封包。當 `traceroute` 超過 5 秒沒有收到回傳時，就會輸出 `*`，並重新發送同樣 TTL 的封包直到 30 輪為止，這也是為什麼後面的輸出全都是 `*`。

---

# 5 Dig out the domain information
## 1.
![|500](School/Course%20Homeworks/Computer%20Networks/assets/1-6.png)

csie.ntu.edu.tw 的 IP address 是 **140.112.30.26**

## 2.
![|500](School/Course%20Homeworks/Computer%20Networks/assets/1-7.png)

amazon.com 的 IP address 有 **54.239.28.85**、**205.251.242.103**、**52.94.236.248**

在同個 domain 使用不同的 IP address 有幾個優點:
- 負載均衡 - 由於可以透過不同的 IP address 存取，可以讓流量不會全部集中在某些伺服器上面
- 穩定性 - 即便有某些 IP address 不能存取，還是能透過剩下的 IP address 存取
- 提升存取速度 - 每個 IP address 的地理位置不同，因此在世界各地都可以透過距離最近的 IP address 進行存取 (其實就是 CDN)

---

# 6 Back to elementary school
## 1.
![|625](School/Course%20Homeworks/Computer%20Networks/assets/1-8.png)

## 2.
No. 因為上一題的網址是 http，因此內容都是明文的，而此題提供的網址是 https，因此雖然還是可以找到封包，但內容都有經過加密。

---

# 7 Yet another curl?
## 1.
![|500](School/Course%20Homeworks/Computer%20Networks/assets/1-9.png)

`-X`: 用來指定請求類型
`-d`: 用來指定 POST 要傳送的資料

## 2.
![|500](School/Course%20Homeworks/Computer%20Networks/assets/1-10.png)

<div style="page-break-after:always;"></div>

# Reference
1 Baby shark
- https://www.networkacademy.io/ccna/ipv6/ipv4-vs-ipv6

2 The course permission code
- https://www.cloudflare.com/learning/email-security/smtp-port-25-587/
- https://sendgrid.com/en-us/blog/what-is-starttls

4 The path to the destination
- https://linux.die.net/man/8/traceroute
- https://en.wikipedia.org/wiki/Traceroute
- https://stackoverflow.com/questions/54988796/why-does-traceroute-expect-destination-unreachable-at-the-final-hop-instead-of

7 Yet another curl?
- https://gist.github.com/subfuzion/08c5d85437d5d4f00e58