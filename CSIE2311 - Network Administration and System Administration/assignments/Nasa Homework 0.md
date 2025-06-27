資工一 鄭博允 B11902038

<br>
<br>

## Network Administration
### True/False

1. <font color="#7AA0BE"> **TRUE** </font>
	LTE 俗稱 3.9G，算是 3G 到 4G 的過渡版本，傳輸速度沒有達到 4G 標準，LTE-A 才符合 4G 標準。

2. <font color="#7AA0BE"> **TRUE** </font>
	IP address 用於辨認主機在網路中的位置，MAC address 用於唯一辨認一張網卡，兩者都能用來唯一地辨認一個網路上的裝置。
   
3. <font color="#BE7AA0"> **FALSE** </font>
	目前 IPv4 耗盡為 RIR 級別的枯竭，代表各區域(洲)的網管機構無法再獲得新的 IPv4 位址。依地區(國家)而異，大部分的 ISP 還是有 IPv4 能夠分配，因此即便有新的裝置也未必要使用 IPv6。況且 IPv4 到 IPv6 的轉換目前還在進行中，並非所有網路服務都有支援 IPv6。
   
4. <font color="#7AA0BE"> **TRUE** </font>
	使用 VPN 的目的通常是為了隱藏真實 IP，因此正常情況下使用者發出的封包的 source IP 在第三者看來應該不會是其真正的 IP，而會是 VPN 伺服器的 IP。
   
5. <font color="#BE7AA0"> **FALSE** </font>
	NAT 是一種轉換 IP 位址的技術，可以將內網不同主機發出的封包轉換成唯一的外網 IP。即便沒有 public IP，外部仍可以透過 NAT 連接到內網。
   
6. <font color="#BE7AA0"> **FALSE** </font>
	DHCP 純粹用來分配 IP，裝置先向 DHCP 要求一個尚未使用的 IP，並用該 IP 連接網路，使用完畢後再將其還給 DHCP，因此裝置發出的封包並不會經過 DHCP server。
   
7. <font color="#7AA0BE"> **TRUE** </font>
	"舊"台大課程網使用的協定為 HTTP，而 HTTP 為明文傳輸，內容皆未經過加密，因此只要成功擷取封包就能夠知道內容。
   
8. <font color="#7AA0BE"> **TRUE** </font>
	兩個 VLAN 間必須透過網路層(也就是用 IP 位址)來通訊，沒有路由器便無法在不同 IP 間轉送封包。
   
9. <font color="#7AA0BE"> **TRUE** </font>
	當裝置還沒有 IP 時，發出 DHCP request 時會使用 0.0.0.0 作為源地址。
   
10. <font color="#7AA0BE"> **TRUE** </font>
	即便沒有 DHCP、DNS、NAT、VPN 等服務，在已知 IP 的狀況下，我們還是能夠連線到一個 server。


### Short Answer

1. &emsp;
	**(a) Gateway**
	中文為閘道器，負責連接兩種不同協定網路的裝置，讓資料在不同協定間轉換，可以避免發生資料漏損。

	**(b) Switch**
	中文為交換器，藉由記錄裝置的 MAC address，採用報文交換的方式在不同裝置間轉發資料。

	**(c) Port Forwarding**
	中文為通訊埠轉發，是 NAT 的一項功能。所有設置在同一內網的裝置，在 NAT 的作用下，外部 IP 是一樣的，然而不同裝置的 port 是不同的。透過 port forwarding，來自外網的需求才能被轉發到內網的不同裝置。
   
2. &emsp;
	將位址換成二進制
	
	| Address       | Binary     |
	| ------------- | ---------- |
	| <font color="#B9AC7E">64.128.64.147</font> | <font color="#7AA0BE">~.10010011</font> |
	| <font color="#B9AC7E">64.128.64.151</font> | <font color="#7AA0BE">~.10010111</font> |
	| <font color="#B9AC7E">64.128.64.155</font> | <font color="#7AA0BE">~.10011011</font> |
	| <font color="#B9AC7E">64.128.64.156</font> | <font color="#7AA0BE">~.10011100</font> |
	
	可以看出從左邊數來第4位開始 (從右邊數來第28位) 都是一樣的，
	因此 subnet 為 **64.128.64.144/28**

3. &emsp;
	**TCP/IP model**

	```ad-sim 
	title: 應用層 Application Layer
	應用程式利用此層來做交流，所有與交換應用程式專用的資料有關的協定都在此層。
	
	e.g. HTTP 用來交換 HTML 資料
	```

	```ad-sim 
	title: 傳輸層 Transport Layer
	負責將網路層的封包傳輸到對應的 port，同時確保傳送可靠性等問題。
	
	e.g. TCP 確認端到端可靠性後，才令資料依規定順序傳送
	```

	```ad-sim
	title: 網路層 Network Layer
	負責處理將封包傳送的路徑選擇，在此層定義封包需要送往哪一個 IP address。
	
	e.g. IP 根據來源主機和目的主機的 IP address 來傳送資料
	```

	```ad-sim 
	title: 連結層 Link Layer
	負責讓封包可以在不同裝置的網路層間傳輸，在此層定義來源／目的的 MAC address。
	
	e.g. 乙太網路
	```

	```ad-sim 
	title: 物理層 Physical Layer
	為連結層提供資料傳送的物理媒介。
	
	e.g. 光纖
	```
   
4. &emsp;
	**(a)**
	
	<font color="#B9AC7E">**TCP** 全名為 Transmission Control Protocol</font>
	<font color="#7AA0BE">**UDP** 全名為 User Datagram Protocol</font>
	兩者皆為傳輸層的通訊協定，
	簡單來講，TCP 是可靠的傳輸，UDP 是快速的傳輸。
	
	**(b)**
	```ad-sim
	title: TCP 說明
	color: 185, 172, 126
	TCP 會先將資料切成小封包並給予編號，再依序傳送封包。接收端會藉由編號確認收到的封包是否依照正確的順序，並發送確認信號給傳送端，此時傳送端才會發出下一個封包。反之，順序錯誤或是封包遺失，傳送端就必須重新傳送。
	```
	
	```ad-sim
	title: UDP 說明
	color: 122, 160, 190
	UDP 不會幫封包編號，以串流的方式發送封包，傳送端會連續不斷發送封包，無視順序錯誤或是封包遺失等問題。因為兩端之間的通訊很少，因此速度比 TCP 還要快，但是出錯率就比較高。
	```
	
	**(c)**
	
	|      | TCP  |  UDP   |
	|:----:|:----:|:------:|
	| 優點 | 可靠 |   快   |
	| 缺點 |  慢  | 不可靠 |
	
	**(d)**
	```ad-sim
	title: TCP 舉例 
	color: 185, 172, 126
	在傳送 email 時，會採用 TCP，因為 email 要求的是資料能夠正確地到達，傳輸時間並沒有那麼重要。
	```
	
	```ad-sim
	title: UDP 舉例
	color: 122, 160, 190
	在使用串流音樂服務 (Spotify) 時，會採用 UDP，因為這種服務的重點在於快速，以及在網路條件不好時能仍使用，而不太在乎資料的完整性。
	```

5. &emsp;
	**(a) CSMA/CD**
	
	全名為 Carrier Sense Multiple Access with Collision Detection。
	
	傳送端在傳送封包前會先偵測頻道是否空閒，若是空閒，則傳送封包。
	
	傳送封包的同時對頻道進行檢測，若是發生碰撞，則發出訊號告知其他節點發生碰撞。碰撞後等待隨機時間再重新傳送封包，嘗試 16 次仍失敗則放棄傳送。
	
	<br>
	
	**(b) CSMA/CA**
	
	全名為 Carrier Sense Multiple Access with Collision Avoidance。
	
	傳送端在傳送封包前會先偵測頻道是否空閒，若是空閒，則等待隨機時間後再偵測一次。
	
	若頻道依然空閒，則傳送封包，反之則再重複上述過程。
	
	此外，傳送端要傳送封包前會先傳送一個很小的 RTS 訊息給接收端，接收端接收到 RTS 後會回應 CTS 訊息給傳送端，並告知其他節點這段時間不能傳送封包，傳送端接收到 CTS 後才會傳送。
	
	<br>
	
	**(c)**
	
	|                            | CSMA/CD                                                                                              | CSMA/CA                                                              |
	| -------------------------- |:---------------------------------------------------------------------------------------------------- |:-------------------------------------------------------------------- |
	| <font color="#7AA0BE">優點</font> | <font color="#7AA0BE">頻道空閒時，因為傳送前不用花費時間等待，傳輸效率較佳。</font>                         | <font color="#7AA0BE">碰撞發生率較低。</font>                               |
	| <font color="#BE7AA0">缺點</font> | <font color="#BE7AA0">當頻道忙碌時，碰撞問題會增加，傳輸效率會驟降。必須不斷偵測，裝置的負荷量較大。</font> | <font color="#BE7AA0">傳送封包前必須花費較多等待時間，傳輸效率較差。</font> |

### Command Line Utilities

1. &emsp;
    使用 google DNS 來查找 https://dns.google/
    
    ## 找出對應的IP address
    **(a)**
    
    ![](https://i.imgur.com/xFfyFqY.png)
    
    IP address : **140.112.8.116**
    
    **(b)**
    
    ![](https://i.imgur.com/Dx7K2pc.png)
    
    IP address : **140.112.30.26**

    ---
    ## 找出對應的domain name
    **(a)**
    
    ![](https://i.imgur.com/GmV1huK.png)
    
    domain name : **pc09.cr.liberal.ntu.edu.tw / cool.ntu.edu.tw**
    
    **(b)**
    
    ![](https://i.imgur.com/u6MhGNL.png)
    
    domain name : **linux5.csie.ntu.edu.tw**



2. &emsp;
    **(a)**
    獲得的 IP address : **140.112.150.168**
    <br>
    
    **(b)**
    ```ad-command
    `dig [domain]` 查詢該網域的詳細資訊
    `dig +trace [domain]` 查看 DNS 從最上層到最下層經過的每個節點 (即 delegation path)
    ```
    
    利用 `dig` 指令查詢，
    可以得知目前使用的 DNS server 為 **140.112.254.4**。
    
    ![[School/Course Homeworks/NASA/assets/nasa hw0/NA 2b-1.png]]
    
    接著，利用 `dig +trace` 指令，得知查詢過程的 delegation path。
    
    ![[School/Course Homeworks/NASA/assets/nasa hw0/NA 2b-2.png| 400]]
    ![](https://i.imgur.com/pxQwImp.png)
    ![](https://i.imgur.com/PTtAzFL.png)
    ![](https://i.imgur.com/fhZZv3H.png)
    ![](https://i.imgur.com/AtLngDC.png)
    ![](https://i.imgur.com/vOmdAo9.png)
    ![](https://i.imgur.com/uUdYFfP.png)
    
    此 query 的 delegation path :
    **192.203.230.10**  (e.root-server.net) →
    **163.28.1.10** (f.dns.tw) →
    **203.73.24.24** (b.twnic.net.tw) →
    **140.112.2.2** (ntu3.ntu.edu.tw) →
    **140.112.30.13** (csman.csie.ntu.edu.tw)
    <br>
    
    **(c)**
    同樣利用 `dig` 指令來查詢，
    得知目前使用的 DNS server 為 **192.168.0.1**
    
    ![](https://i.imgur.com/7m6j1DD.png)
    
    DNS server 與 **(b)** 不同是由於連上 vpn 後，使用的會是 vpn server 所指定的 DNS；而沒有連上 vpn 時使用的會是 ISP 預設的 DNS。
    <br>

    **(d)**
    ```ad-command
    `tracert [domain/IP]` 追蹤封包傳達到指定目的地所經的路徑，可用來確認網路連線狀況
	```
    
    利用 `tracert` 來追蹤路徑
    
    ![](https://i.imgur.com/4yEGTB0.png)
    
    因為我家的無線路由器有開啟 DNS Relay，所以電腦裡看到的 DNS server 會是路由器的 IP，查到的 routing path 就變成電腦 → 路由器了。

---
## System Administration
### 1. 星球權限掌握

先依照題目給的帳號密碼登入
然後使用 `sudo su` 變更權限成 root
用 `cd ~` 移動到家目錄
用 `ls` 查看當前目錄的檔案，發現 `treasure_box-1`
用 `./` 打開檔案

![[School/Course Homeworks/NASA/assets/nasa hw0/flag1.png| 300]]

```ad-flag
title: **NASA {P1_I'm a good root}**
```

```ad-command
`sudo su` 輸入使用者密碼，獲取 root 權限

`cd` 移動到指定位置

`ls [OPTIONS]` 查看當前目錄的所有檔案 
-- `-l`可看到檔案詳細資訊
-- `-a`可看到隱藏的檔案

`./` 在當下目錄執行程式
```
<br>

### 2. 遠端操控

用 `systemctl start sshd` 開啟 ssh 服務

![](https://i.imgur.com/bSy7GeB.png)

用 `systemctl status sshd` 查看 ssh 目前的狀態，
發現開機自動開啟的功能為 `disabled`，並利用 `systemctl enable sshd` 開啟這項功能，再查看一次狀態，設定成功。

![](https://i.imgur.com/Fz28gB7.png)

接著要設定 port forwarding 才能夠遠端連線
先查看 VirtualBox 的虛擬 ip 為 192.168.56.1

![](https://i.imgur.com/sGnZiuA.png)

用 `ip address` 查看此虛擬主機 `Mars` 的 ip 為 10.0.2.15

![](https://i.imgur.com/grj5EA1.png)

回到 VirtualBox 設定 port forwarding

![](https://i.imgur.com/whfDqPo.png)

打開 cmd，用 `ssh musk@192.168.56.1` 遠端連線進虛擬主機中

![](https://i.imgur.com/APVZOM5.png)

回到虛擬主機中，打開 `treasure_box-2`

![](https://i.imgur.com/BlRCg9y.png)

```ad-flag
title: **NASA {P2_GoodRemote}**
```

```ad-command
`systemctl [OPTIONS] [SERVICE]`
-- `start` 啟動系統服務
-- `status` 查看系統服務狀態
-- `enable` 開啟開機自動啟動功能

`ip address` 查看主機 ip 位址

`ssh [USER]@[HOST IP]` 以指定身分 ssh 連線進入主機
```

<br>

### 3. 新手大禮包

由於還在 root，先用 `exit` 離開 root 權限

![](https://i.imgur.com/DulauUc.png)

接著用 `tar xvf gifts_from_SA.tar` 解包
因為跑出太多東西，指令被擠上去了，沒截到圖，所以就不放了

用 `cd gifts_from_SA` 移動到資料夾裡，發現 `flag_3` 檔案
本來想用 `./` 直接執行，發現好像不能這樣做
於是改用 `cat flag_3` 把檔案內容印出來

![](https://i.imgur.com/wI0cQCk.png)

```ad-flag
title: **NASA {P3_I Love SA}**
```

```ad-command
`exit` 退出目前的 shell

`tar [OPTIONS] [.tar]`
-- `cvf` 打包
-- `xvf` 解包

`cat [FILE]` 查看檔案內容
```
<br>

### 4. 星球資訊

用 `uname -a` 查看內核版本

![](https://i.imgur.com/U1Fei3T.png)

```ad-flag
title: **NASA {P4_Linux Mars 6.1.4-arch1-1}**
```

```ad-command
`uname [OPTIONS]`
-- `-a` 印出主機內核的詳細資訊
-- `-n` 印出顯示於網路上的主機名稱
```
<br>

### 5. 這是我的星球

用 `sudo hostnamectl set-hostname MuskPlanet` 修改 hostname

![](https://i.imgur.com/Ml39V6u.png)

用 `uname -n` 確認 hostname

![[School/Course Homeworks/NASA/assets/nasa hw0/flag5-2.png| 300]]

移到 `gifts_from_SA` 找到 `p5`
在裡面執行 `treasure_box-5`

![](https://i.imgur.com/wR5XOEp.png)

```ad-flag
title: **NASA {P5_AHOY My planet}**
```

```ad-command
`hostnamectl set-hostname [NAME]` 更改主機名稱
```
<br>

### 6. 強制更名

我不確定這樣算不算直接動到 `/etc/passwd`
除了 `usermod` 和 `chfn` 我在網路上找不到其他的方法
本來想說 `chfn` 就不用 sudo 
但是 `login.defs` 的 `CHFN_RESTRICT` 被禁了 
最後還是用了 sudo

用 `chfn -f` 更改 full name 

![](https://i.imgur.com/UIPK1vH.png)

執行 `treasure_box-6` 

![[School/Course Homeworks/NASA/assets/nasa hw0/flag6-2.png| 400]]

```ad-flag
title: **NASA {P6_Tesla Fxxk}**
```

```ad-command
`chfn -f ["NAME"] [USER]` 更改指定使用者的全名
```
<br>

### 7. 臥底

用 `useradd` 新增使用者 `nasa`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag7-1.png| 400]]

用 `passwd` 設定 `nasa` 的密碼

![[School/Course Homeworks/NASA/assets/nasa hw0/flag7-2.png| 400]]

用 `groupadd` 新增群組 `spy`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag7-3.png| 350]]

用 `usermod --gid` 將 `musk` 的主要群組更改為 `spy`
(後來發現好像應該用 `usermod -a -G` 才不會讓使用者退出原本群組)

![](https://i.imgur.com/o6cG63L.png)

用同樣的方法更改 `nasa` 的群組

![](https://i.imgur.com/sm8Jcbp.png)

在 p7 下找到 `openDoor` 

![[School/Course Homeworks/NASA/assets/nasa hw0/flag7-6.png| 350]]

用 `chown` 更改檔案的擁有者和群組
再用 `ls -l` 查看檔案的詳細資訊，確認更改成功

![](https://i.imgur.com/jdrm1jP.png)

最後執行 `openDoor`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag7-8.png| 300]]

```ad-flag
title: **NASA {P7_Door opened}**
```

```ad-command
`useradd [USER]` 新增使用者

`passwd [USER]` 更改使用者密碼

`groupadd [GROUP]` 新增群組

`usermod [OPTIONS]`
-- `-g` `--gid` 設定使用者的主要群組
-- `-G` `--groups` 設定多個使用者的副群組，加上 `-a` 可以避免使用者被移出原群組

`chown [USER] [FILE]` 更改檔案的擁有者
`chown [:[GROUP]] [FILE]` 更改檔案的所在群組
```
<br>

### 8. 辣個男人

用 `man pacman` 查看 pacman 的詳細資訊

![[School/Course Homeworks/NASA/assets/nasa hw0/flag8-1.png| 300]]
![](https://i.imgur.com/1lndF53.png)

然後就會看到名稱是 pacman - package manager utility
(全名應該是指這個吧?)

```ad-flag
title: **NASA {P8_man pacman_pacman - package manager utility}**
```

```ad-command
`man [COMMAND]` 查看指令的使用方法、說明 

根據 Arch linux 官方對 pacman 的介紹，
pacman 可以用來管理各種軟體包，包括安裝、刪除、更新等
基本的使用方法：
* 安裝 `pacman -S <package>`
* 刪除 `pacman -R <package>`
* 查詢 `pacman -Q <package>`
* 更新 `pacman -U <package>`
```
<br>

### 9. 中文才是王道！

用 `localectl set-locale` 改變系統的區域設置

![](https://i.imgur.com/nGsRZ3h.png)

執行 `treasure_box-9` 

![[School/Course Homeworks/NASA/assets/nasa hw0/flag9-2.png| 350]]

```ad-flag
title: **NASA {P9_Taiwan No.1!}**
```

```ad-command
`localectl set-locale [VARIALBE]=[LOCALE]` 更改區域設定，變數 `LANG` 為主語言環境

順帶一提 `localectl status` 可以查看目前的系統區域
```
<br>

### 10. 誰說這個星球很無聊？

用 `sl` 叫出小火車

![[School/Course Homeworks/NASA/assets/nasa hw0/flag10-1.png| 250]]
![](https://i.imgur.com/Eth9x5G.png)

```ad-flag
title: NASA {P10_sl}
```

後面加上 `lolcat` 變成彩色

![[School/Course Homeworks/NASA/assets/nasa hw0/flag10-3.png| 350]]
![](https://i.imgur.com/fCTbdco.png)

```ad-flag
title: NASA {P10_sl | lolcat}
```

用 `cowsay -l` 確認能召喚出的寵物

![](https://i.imgur.com/1eiPnY7.png)

用 `cowsay -f ` 叫出 dragon-and-cow

![](https://i.imgur.com/qz943Gb.png)

```ad-flag
title: NASA {P10_cowsay -f dragon-and-cow}
```

一樣後面加上 `lolcat` 變成彩色

![](https://i.imgur.com/YFKCBW2.png)

```ad-flag
title: NASA {P10_cowsay -f dragon-and-cow | lolcat}
```

```ad-command
`sl` 召喚小火車

`cowsay [OPTIONS]` 
-- `-l` 查看能召喚出的所有動物
-- `-f [動物] [要講的話]` 召喚指定動物出來然後講話

`lolcat` 把東西變彩色(任何東西)

另外 `|` 叫做管線，可將左邊指令的輸出變成右邊指令的輸入
```

### 11. 竊取機密文件

先用 `mkdir` 建立 `hide/` 
用 `ls` 確認建立成功

![[School/Course Homeworks/NASA/assets/nasa hw0/flag11-1.png| 400]]

用 `ls -ld` 查看 `hide/` 的權限

![[School/Course Homeworks/NASA/assets/nasa hw0/flag11-2.png| 400]]
```ad-info
title: 解讀權限

| d    | rwx    | r-x          | r-x      |
| ---- | ------ | ------------ | -------- |
| 類型 | 擁有者 | 同群組使用者 | 其他用戶 |

第一個字母代表類型
`d` 代表目錄、 `-` 代表檔案

| 權限 | 檔案   | 目錄                              |
| ---- | ------ | --------------------------------- |
| r    | 可讀取 | 可 `ls`                           |
| w    | 可修改 | 可修改目錄 / 新增刪除目錄內的檔案 | 
| x    | 可執行 | 可 `cd`                           |
`-` 代表沒有該權限
```

根據題目敘述，應該將 `hide/` 的權限改成 `rwx --x ---`
利用 `chmod` 更改權限，再用 `ls -ld` 確認修改成功

![[School/Course Homeworks/NASA/assets/nasa hw0/flag11-3.png| 400]]

接著用 `mv` 把 `secret` 移到 `hide/` 底下
用 `cd` 把自己也移到 `hide/` 裡面

![[School/Course Homeworks/NASA/assets/nasa hw0/flag11-4.png| 450]]

執行 `secret` 

![[School/Course Homeworks/NASA/assets/nasa hw0/flag11-5.png| 350]]

```ad-flag
title: NASA {P11_good spy}
```

```ad-command
`mkdir` 在當前目錄建立新的目錄

`chmod [OPTIONS] [FILE]` 更改檔案或目錄權限
`OPTIONS` 的基本格式為 `[ugo][[-+][rwx]]`
-- `u` `g` `o` 分別代表 user、group、others，不指定則視為所有人
-- `-` `+` 代表移除 / 新增某權限
-- `r` `w` `x` 代表權限

`mv` 移動檔案和目錄到指定位置
```

### 12. 製造大檔案

#### 方法一：

用指令執行十次 `fallocate`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag12-6.png]]

```ad-command
為了方便，就直接解釋整行指令
`seq 0 9 | xargs -i fallocate -l 1M {}` 

`seq 0 9` 會依序產生整數 0~9 

`xargs -i` 
會將 stdin 接收到的字串(也就是 0~9)
轉換成後面指令 (`fallocate`) 的指定參數 (`{}`) 

`fallocate -l 1M [FILENAME]`
產生指定大小為 1M 的空白檔案，此處檔案名稱被換成 `{}`

結果就會產生名稱為 0~9 的十個大小為 1M 的空白檔案
```

#### 方法二：

用指令執行十次 `dd`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag12-2.png]]

```ad-command
這行指令的原理和上一行一樣，只是把 `fallocate` 換成 `dd`

`dd if=/dev/zero of={} bs=1M count=1`

`if=/dev/zero` 會從 `/dev/zero` 讀取輸入，而 `/dev/zero` 在被讀取時會提供無限的空字元，因此就會產生空白檔案

`of={}` 將輸入寫入 `{}`(這裡為 0~9) 作為輸出

`bs=1M` 設定輸入 / 輸出的資料塊大小為 1M (意即一次輸入 1M 並輸出 1M)

`count=1` 複製 1 個資料塊 (意即複製 1 個 1M 的資料塊到輸出)

結果一樣會產生名稱為 0~9 的十個大小為 1M 的空白檔案
```

#### 方法三：

用 `fallocate` 產生名稱為 `0` 大小為 1M 的空白檔案
然後用 `cp` 複製九次

![[School/Course Homeworks/NASA/assets/nasa hw0/flag12-10.png| 400]]
~ 以此類推
最後就會產生名稱為 0~9 的十個大小為 1M 的空白檔案

```ad-command
`cp [FILE1] [FILE2]` 在當前目錄複製 `FILE1` 並將名稱改為 `FILE2`
```

#### 結果：
用 `ls -l` 查看成功與否 (我就不放三次圖片，反正結果一樣)

![[School/Course Homeworks/NASA/assets/nasa hw0/flag12-3.png]]

執行 `treasure_box-12`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag12-4.png| 400]]

```ad-flag
title: NASA {P12_nice large file}
```

另外，在嘗試不同方法時用 `rm` 刪除原本檔案

![[School/Course Homeworks/NASA/assets/nasa hw0/flag12-5.png]]

### 13. Bocchi The Rock!

執行小波奇

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-1.png]]

先用 `ctrl z` 將小波奇暫停並丟到後台，這樣才可以做事

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-2.png]]

用 `jobs` 查看小波奇的狀況，
得知目前為 `stopped` 暫停狀態，以及 `PID` 為 894

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-3.png]]

用 `kill` 終止小波奇發癲

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-4.png]]

接著用 `cat keygen | ./getkey` 
將 `keygen` 的內容做為輸入，執行 `getkey`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-5.png]]

得到網址 https://www.csie.ntu.edu.tw/~b09902109
用 `wget` 下載圖片

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-6.png]]

為了找出隱藏在圖片中的秘密，
我盯著圖片十分鐘，發覺了一個事實：
~~喜多醬真的好可愛~~

我 google 了一下，知道了隱寫術 (Steganography) 這個東西
網路上很多都是額外下載套件來達成隱寫的目的，
但因為前面有說 vm 裡的套件就可解出所有答案，
所以應該就只剩下直接把圖片解壓縮的方法

我用 `mv` 將 `bocchi.jpg` 重新命名為 `bocchi.gz`
照理來說這樣要可以解壓縮，並出現神秘檔案，
但是它成功地失敗了

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-10.png]]
![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-11.png| 400]]

之後我也嘗試了把它改成 `tar` `tar.gz` 等等，花了整個下午，用盡了各種解壓縮的方法，還是沒辦法成功

於是，我只好採用暴力破解法
我先在 windows 把 `bocchi.jpg` 以二進制的模式打開，理所當然的會出現一堆英文數字
因為 flag 通常會含有 `NASA` 這四個字元，因此我以 ascii 的規則將其轉換成十六進制，分別為 `4e41 5341`
然後直接在 `bocchi.jpg` 裡尋找這串文字，還真被我給找到了

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-12.png]]

然後我就抓個兩行，將其換回 ascii 字元，得到了 flag 13

```ad-flag
title: NASA {P13_GuitarHeroBocchiDesu}
```
---
補充：
後來我發現應該可以試試看用 `grep` 把 `NASA` 給抓出來

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-15.png| 400]]

但好像不行，於是我改用 `cat bocchi.jpg | less`
一頁一頁查找，其實很快就找到了，畢竟其他部份都是亂碼

![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-13.png| 400]]
![[School/Course Homeworks/NASA/assets/nasa hw0/flag13-14.png]]

```ad-command
`jobs -l` 查看目前所有 process 的狀態

`kill [OPTIONS] [PROCESS]`
-- `-1` 重啟 process
-- `-9` 強制停止 process
-- `-15` 正常停止 process

`wget [圖片網址]` 從網路上下載圖片 

`less [FILE]` 一頁一頁的查看檔案內容
-- `ctrl f` 下一頁
-- `ctrl b` 上一頁
-- `q` 離開
```

### 14. 加密文件

(這題的指令又臭又長，為了方便，一行一行解釋)

![[School/Course Homeworks/NASA/assets/nasa hw0/flag14-2.png]]

```ad-command
`cat storybook | sed 's/[^[:alnum][:space]]//g' > test`

將 `storybook` 作為 `sed` 的輸入，並將 `sed` 的輸出存入 `test`

`sed 's/[pattern]/[replace]/g'`
`s` 代表替換 substitute
`g` 代表全部 global
若沒有 `replace` 則默認為把 `pattern` 刪除

`[^list]` 為反向選擇，`list` 為不要選擇的字元
`[:alnum:]` 代表 0~9, a~z, A~Z
`[:space:]` 代表任何會產生空白的字元

所以此行指令即為
刪除 `storybook` 中所有"非"大小寫英文數字空格的字元，並將其存入 `test`
```

先用一次 `grep` 確認 `test` 裡有 pacman 這串文字

![[School/Course Homeworks/NASA/assets/nasa hw0/flag14-3.png]]

在用一次 `grep` 後面接上 `sed` 將字元做替換

![[School/Course Homeworks/NASA/assets/nasa hw0/flag14-4.png]]

```ad-command
`grep -n pacman test | sed 'y/0123456789abcdefghijklmnopqrstuvwxyzA/A{Dhi:<oP1sS4gNTcuy_@kln#%are+*f&t^>}/'`

`grep -n [pattern] [FILE]`
在 `FILE` 中尋找 `pattern`
`-n` 顯示目標所在行數

 `sed 'y/[list1]/[list2]/'`
 會將 `list1` 的字元依序替換成 `list2` 的字元
 (兩個 list 長度要一樣)

所以此行指令即為
在 `test` 中尋找 `pacman` 所在行數，並對該行內容做字元替換
```

```ad-flag
title: NASA {P14_This@large#file%is+such*an&asshole^Dont:you<think<think>>so}
```
雀食 : )

### 15. 國家機器動得很厲害

先建立一個新的 `findflag.sh`

![[School/Course Homeworks/NASA/assets/nasa hw0/flag15-1.png| 400]]

![[School/Course Homeworks/NASA/assets/nasa hw0/flag15-2.png| 400]]

```ad-command
建立一個 `find()` 函式

內容是在 `p15/${1}/*` 找出所有長度小於等於 25 的行，並印出來。
至於為什麼是 25，因為我發現大部分的長度都是 26，很合理。

`${1}` 為函式的輸入變數，這裡代表 `p15` 底下的目錄
`*` 代表該目錄底下的所有檔案
`${#line}` 計算每一行的長度
`-le` 小於等於

然後重複執行 `find()` 十次
```

![[School/Course Homeworks/NASA/assets/nasa hw0/flag15-4.png| 400]]

```ad-flag
title: NASA {P15_Do_You_Find_Me_}
```

### 16. 來自SA 的嘲諷

廣播是定時發出的，因此有可能與 `crontab` 有關
根據 google 到的資訊，系統預設執行腳本設定檔的地方有兩個 `/etc/crontab` `/etc/cron.d/*`
移動到 `/etc/cron.d/*` 裡發現了兩個檔案
`0hourly` 似乎是預設的，用 `cat` 查看 `minute` 的內容，好像就是亂源

![[School/Course Homeworks/NASA/assets/nasa hw0/flag16-3.png]]

根據內容，系統會每五分鐘寄信給 `root` ，接著 `root` 會執行廣播

![[School/Course Homeworks/NASA/assets/nasa hw0/flag16-1.png]]

直接用 `rm` 讓這個檔案消失

![[School/Course Homeworks/NASA/assets/nasa hw0/flag16-6.png| 300]]

等了一下後，家目錄底下出現了 flag 16

![[School/Course Homeworks/NASA/assets/nasa hw0/flag16-2.png| 300]]

```ad-flag
title: NASA {P16_don't laugh me!}
```

```ad-flag
title: 被定期執行的檔案為 `/etc/cron.d/minute`
```

對於在背後檢查的程式，我完全沒有頭緒，
所以我直接用 `grep` 在 `/etc` 裡尋找跟 flag16 有關的檔案，
結果還真的被我找到了

![[School/Course Homeworks/NASA/assets/nasa hw0/flag16-8.png]]

查看 `AutoExec.sh` 的內容，應該就是這個了吧

![[School/Course Homeworks/NASA/assets/nasa hw0/flag16-9.png]]

```ad-flag
title: 檢查程式為 `/etc/systemd/system/AutoExec.sh`
```

---

喔耶我終於寫完了  -v-b