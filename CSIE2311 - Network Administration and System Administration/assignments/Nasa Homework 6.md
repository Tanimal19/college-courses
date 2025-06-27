B11902038 資工一 鄭博允

---

# Network Administration
## 1. SSID / BSSID
> Reference :
[1] https://blog.csdn.net/qq_43804080/article/details/105506982
[2] https://blog.csdn.net/reekyli/article/details/108765419
[3] https://blog.csdn.net/jerry81333/article/details/52952300


### 1.
**SSID (Service Set IDentifier)** : 無線網路的識別名稱
**BSSID (Basic Service Set IDentifier)** : 無線網路中每個 AP 的網卡的 MAC address

舉例來說，"csie" 這個 wifi 只有一個 SSID = "csie"，但 "csie" 下可能有多個 AP，因此我們用不同的 BSSID 來區別它們。
<br>

### 2.
對於同一個 AP 來說，可以同時擁有多個 SSID，每個 SSID 代表不同的無線網路；同一台 AP 也可以同時擁有多個 BSSID，當它需要支援多個 SSID 時，必須對每一個 SSID 都分配一個虛擬網卡，每張網卡都有自己的 BSSID。
<br>

### 3.
evil twin attack 的原理如下：
由於我們在連接 wifi 時是用 SSID 來識別的，所以攻擊者只要另外建立一個名稱一樣的 wifi 就可以欺騙使用者，一旦使用者連上了錯誤的 wifi，攻擊者可以導引使用者到虛假的網站，竊取使用者的資料。

我們可以利用身分驗證來防止這種攻擊，在登入 wifi 使用身分驗證，使攻擊者難以只用 SSID 來偽造相同的 wifi。
<div style="page-break-after:always;"></div>

## 2. PSK/EAP/PEAP
> Reference :
[1] https://www.intel.com.tw/content/www/tw/zh/support/articles/000006999/wireless/legacy-intel-wireless-products.html
[2] https://zh.wikipedia.org/zh-hant/%E6%89%A9%E5%B1%95%E8%AE%A4%E8%AF%81%E5%8D%8F%E8%AE%AE
[3] https://zh.wikipedia.org/zh-hant/WPA

### 1.
**PSK (pre-shared ey)** : 預共用密鑰，簡單來說就是使用者和 AP 之間先達成一個共識：「我之後會用這個密碼登入喔」「好喔」，之後使用者在登入該無線網路時只要使用該密碼就可以。這種模式的安全性通常取決於密鑰的保存與共享方式，還有密鑰本身的強度。

**EAP (Extensible Authentication Protocol)** : 可延伸的驗證通訊協定，它不算是一種協定，而是一個標準框架，允許各家網路廠商在框架下開發不同的驗證方法。要實現 EAP 必須架設額外的驗證伺服器，來進行用戶端的驗證。

**PEAP (Protected Extensible Authentication Protocol)** : 是一種 EAP 類型，PEAP 會在用戶端和驗證伺服器之間建立一個安全通道，用來傳輸驗證資料，可以增加傳輸的安全性。PEAP 只使用伺服器端的憑證來驗證用戶端，由於用戶端不需要配置憑證，因此可以簡化 wifi 的實施與管理。

綜合比較：
PSK 容易實施，常被用在家庭或是小型網路，安全性較弱。而 EAP 的安全性較強，但成本較高，常用在企業或大型網路，PEAP 是一種安全性更高的 EAP。
<br>

### 2.
PSK 更適合用於 personal，因為它只要在 AP 上做簡單的設定即可，成本低且一實現。不適合用於 enterprise network 是因為它的安全性不夠強，只要密碼被知道就完蛋了。
<div style="page-break-after:always;"></div>

## 3. Connect to WiFi with terminal
> Reference :
[1] https://www.windowscentral.com/how-connect-wi-fi-network-windows-10

使用系統 : windows 11 
方法如圖所示
![[School/Course Homeworks/NASA/assets/nasa hw6/1.png]]
<div style="page-break-after:always;"></div>

## 4. csie/csie-5G
> Reference :
[1] https://www.tp-link.com/tw/support/faq/499/
[2] https://www.asus.com/hk/support/FAQ/1044838/
[3] https://www.intel.com.tw/content/www/tw/zh/products/docs/wireless/2-4-vs-5ghz.html

### 1.
2.4GHz 和 5GHz 分別代表了無線電波的兩個頻段，比較如下：

|                  | 2.4GHz | 5GHz |
| ---------------- |:------:|:----:|
| 速度             |  較慢  | 較快 |
| 覆蓋率(傳播距離) |  較廣  | 較短 |
| 非重疊頻道       |  較少  | 較多 |
| 支援裝置         |  較多  | 較少 |

理論上來說，5GHz 的網速會比 2.4GHz 來的快，這是因為支援的 wifi 標準不同，不是因為數字比較大。 

綜合來說，2.4GHz 的網速比較慢、可用的頻段也比較少，由於很多裝置使用 2.4GHz 所以時常阻塞，但可以傳輸的更遠、穿透力也最好／5GHz 的理論速度比 2.4GHz 快上許多，但相對的穿透力及傳輸距離就比不上 2.4GHz。
<br>

### 2.
我們應該優先連接 csie-5G 而非 csie，首先，5G wifi 的速度比較快，再來是因為 csie 同時支援 5G 和 2.4G，一旦連上可能會在兩個頻段之間切換，若太過頻繁可能會影響網速，所以單純的使用 csie-5G 會來的較佳。
<div style="page-break-after:always;"></div>

## 5. AP location
放在 R103-rear 應該會較佳，因為該位置的覆蓋率比較大也比較平均，如果放在 R-103 front，教室左後方的訊號會不太好。
![[School/Course Homeworks/NASA/assets/nasa hw6/2.png|375]]
<div style="page-break-after:always;"></div>

# System Administration
## Web Terminology
> Reference :
[1] https://tenten.co/insight/dev/apache-nginx-comparison/
[2] https://developer.mozilla.org/zh-TW/docs/Learn/Common_questions/Web_mechanics/What_is_a_web_server
[3] https://ithelp.ithome.com.tw/articles/10216821
[4] https://ithelp.ithome.com.tw/articles/10267249
[5] https://ithelp.ithome.com.tw/m/articles/10291291
[6] https://www.hksilicon.com/articles/2171039
[7] https://zh.wikipedia.org/zh-tw/%E4%BB%A3%E7%90%86%E6%9C%8D%E5%8A%A1%E5%99%A8#%E5%8A%9F%E8%83%BD
[8] https://zh.wikipedia.org/zh-tw/%E5%8F%8D%E5%90%91%E4%BB%A3%E7%90%86
[9] ChatGPT

<div style="page-break-after:always;"></div>

### 1.
Apache 和 Nginx 最主要的差別在於它們的基本架構，Apache 為多執行緒的伺服器，每一個請求都會建立新的執行緒來處理；而 Nginx 則是事件觸發的伺服器，使用非阻塞I/O模型來處理請求，一個執行緒可以處理多個請求。因此，Nginx 的資源消耗和處理效能都贏過 Apache。

兩者在處理靜態／動態內容時也有著不同的表現，Nginx 比較擅長處理靜態內容；Apache 可以在伺服器內處理動態內容，而 Nginx 則是會外包給其他進程處理。

Apache 的發展歷史比 Nginx 還要久，也因此擁有較豐富的擴充資源，bug 也相對少，比起 Nginx 來的更加穩定。 
<br>

### 2.
**Static Web Server** : 不管任何請求，網站只會傳送一樣的東西給使用者，例如已經編寫好的文件 (HTML、CSS 等)。
**Dynamic Web Server** : 會根據不同請求而有不同回饋，例如可以向使用者發送個人偏好的訊息。
<br>

### 3.
網頁渲染：將網頁內容轉換成使用者可以觀看、互動的形式。

**CSR (Client-side Rendering)** : 一開始瀏覽器就會跟伺服器要全部資料並存在用戶端，這樣之後切換頁面就不用再和伺服器拿資料了。
缺點是初始載入較久(要下載 JS 檔)；優點是切頁面很快。

**SSR (Server-side Rendering)** : 每次用戶發送請求時，瀏覽器才跟伺服器要資料，此時伺服器會產生 HTML 頁面並回傳。
缺點是每次請求都需要等待；優點是有利於 SEO (搜尋引擎優化，讓你的網頁在搜尋引擎的排序更前面)，因為伺服器在傳送給瀏覽器的初始頁面就包含了資料，讓搜尋引擎可以獲得正確的資料做排序。

**SSG (Static Site Generation)** : 在建立階段就將頁面渲染好，也就是說伺服器基本上不用再根據使用者請求做渲染，只要回傳一開始就渲染好的 HTML 即可，適合用在靜態網頁。
缺點是不能更新網頁內容，一旦有變動就要重新編譯；優點是渲染好的檔案可以被伺服器 cache (因為內容不會變)，加快下次的讀取速度。

**ISR (Incremental Static Regeneration)** : 算是一種 SSG 的擴展，在建立階段時不會渲染全部的檔案，保留部分不做渲染，等到有請求或是過期，才會產生新的特定檔案。
優點是保留 SSG 的特性，但解決了 SSG 不能更新網頁內容的問題。
<div style="page-break-after:always;"></div>

### 4.
Proxy 可以被看成是一個或多個用戶端的「代理」，負責和伺服器之間的溝通，用戶端向伺服器發送請求時會先經過 proxy，再由 proxy 轉發給伺服器。使用 proxy 有許多好處 :
- 加快存取速度：proxy 可以 cache 之前的請求內容，當使用者下次發出相同請求時，就不用再連線到伺服器。
- 增加安全性： 因為所有連線都會經過 proxy，它可以充當防火牆的角色，過濾流量、阻止攻擊等。
- ~~科學上網~~ 突破區域限制：~~中國~~ (部分國家) 會限制民眾連到特定的網站，而我們可以透過 proxy 來變更 IP，來繞過特定限制。 
<br>

### 5.
Reverse Proxy 同樣也是「代理」，不過是伺服器的代理，也就是說用戶端發送請求時，實際上是發送給 proxy，而 proxy 再轉發給背後的其他伺服器，因此我們可以只和 proxy 溝通而獲得其背後的不同資源。
使用 reverse proxy 的好處如下 : 
- 對用戶端隱藏伺服器(叢集)的 IP : 用戶端只會知道 reverse proxy 的 ip，而不會知道其背後還有其他的伺服器
- 附載均衡 : 可以平均的分配請求給各個伺服器，而不會造成單一伺服器負荷過大。
- 加快存取速度 : 和一般的 proxy 一樣，reverse proxy 也可以 cache 之前的請求內容，收到相同請求時便可以直接回覆。
<div style="page-break-after:always;"></div>

## Web Server Configurations
> Reference :
[1] https://www.ibm.com/docs/zh-tw/ibm-mq/9.1?topic=concepts-public-key-infrastructure-pki
[2] https://zh.wikipedia.org/zh-tw/%E5%85%AC%E9%96%8B%E9%87%91%E9%91%B0%E5%9F%BA%E7%A4%8E%E5%BB%BA%E8%A8%AD
[3] https://zh.wikipedia.org/zh-tw/%E8%87%AA%E5%8B%95%E6%86%91%E8%AD%89%E6%9B%B4%E6%96%B0%E7%92%B0%E5%A2%83
[4] https://certbot.eff.org/instructions?ws=nginx&os=debiantesting
[5] https://www.sunzhongwei.com/certbot-install-certificate-error-requested-nginx-plugin-does-not-appear-to-be-installed
[6] ChatGPT

<br>

### 1. Basic Setups

在 VM 上安裝 nginx
```bash
sudo apt update
sudo apt install nginx
```

啟動 nginx service
```bash
sudo systemctl enable nginx
sudo systemctl status nginx # 確認狀態
```

連上 VM 的 ip 應該會看到 nginx 的預設頁面
![[School/Course Homeworks/NASA/assets/nasa hw6/8.png|500]]

<div style="page-break-after:always;"></div>

### 2. Firewall Settings

安裝套件 iptables
```bash
sudo apt update
sudo apt install iptables
```

配置設定，允許 port 22、80 的連線並拒絕其他連線
```bash
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -A INPUT -j DROP
```

保存設定
```bash
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

如果成功應該可以看到下列輸出
![[School/Course Homeworks/NASA/assets/nasa hw6/3.png]]

我們可以在主機上用 netcat (nc) 來側特定端口的連線
```bash
nc -v [VM ip] [port] 
```

結果如下圖，port 22、80 是可以連線的，而 port 20 則無法連上
![[School/Course Homeworks/NASA/assets/nasa hw6/4.png]]

<div style="page-break-after:always;"></div>

### 3. The Main Page

建立 index.html
```bash
sudo vim /var/www/html/index.html
```

重啟 nginx service
```bash
sudo systemctl restart nginx
```

網站截圖
![[School/Course Homeworks/NASA/assets/nasa hw6/5.png|500]]

<div style="page-break-after:always;"></div>

### 4. User Directory
建立新的目錄 `/var/www/html/b11902038`
然後將 index.html 移到新建的目錄內
重啟 nginx service

網站截圖
![[School/Course Homeworks/NASA/assets/nasa hw6/6.png|500]]

<div style="page-break-after:always;"></div>

### 5. Let’s Encrypt
**(a)** 
PKI (Public Key Infrastructure) 是一個用於管理和驗證數位憑證的架構，通常會包含「憑證管理中心 (CA)」及「註冊管理中心 (RAs)」。
TLS (Transport Layer Security) 則是一個用來保護網路通訊安全的協定。

我們可以把 TLS 視為一種 PKI 的應用，在 TLS 握手過程中，服務器和客戶端會使用PKI來驗證對方的身份，並交換數位憑證以確保通信的安全性。
<br>

**(b)**
ACME (Automatic Certificate Management Environment) 是一種通訊協定，用途是自動化網站的數位憑證管理過程。

傳統的更新憑證需要手動操作，複雜且耗時，ACME 可以將這些流程自動化，使更新憑證更加方便快速。

<div style="page-break-after:always;"></div>

**(c)**
此題利用分配的 VM 來進行

安裝 nginx 等步驟與前面幾題相同

安裝 certbot 以及相關套件
```bash
apt install certbot python3-certbot-nginx
```

然後直接執行下面指令取得憑證
```bash
certbot --nginx
```

接著會要你輸入一些資訊 email 等
其中要輸入 domain name 時要特別注意，因為 VM 已經有預設的 domain name 了
我們先透過 `ip a` 來查看 VM 的 ip，然後用 dns (這裡用 google dns) 來查詢 domain name
結果如圖
![[School/Course Homeworks/NASA/assets/nasa hw6/9.png]]

我們再將查到的 domain name 填入，接著 certbot 就會幫你把其他事都做完

之後我們就可以用 https 連上網站了
![[School/Course Homeworks/NASA/assets/nasa hw6/7.png]]

我不確定 nginx 的設定檔指的是哪個，但我觀察 `/etc/nginx/nginx.conf` 前後是相同的 

<div style="page-break-after:always;"></div>

### 6. Secret
建立 secret.html 檔
```bash
sudo vim /var/www/html/secret.html
```

然後到 nginx 的設定檔 `/etc/nginx/sites-available/default` 加入以下內容
```py
location /secret.html {
	allow 192.168.28.0/24;  # 允許連線
	deny all;               # 拒絕連線
	root /var/www/html;     # 指定根目錄
}
```

重啟 nginx service 之後應該可以看到 (如果你是 192.168.28.0/24 的話) 以下網頁
![[School/Course Homeworks/NASA/assets/nasa hw6/10.png|600]]

<div style="page-break-after:always;"></div>

## Reverse Proxy
> Reference :
[1] https://docs.nginx.com/nginx/admin-guide/web-server/reverse-proxy/
[2] https://nginx.org/en/docs/http/ngx_http_proxy_module.html?&_ga=2.254072840.1636511183.1685529903-2054018530.1684991829#proxy_set_header
[3] https://noob.tw/nginx-reverse-proxy/
[4] https://dotblogs.com.tw/eric_obay_talk/2018/08/03/113020
[5] https://blog.csdn.net/bigtree_3721/article/details/72820594


首先在 VM 上安裝 nginx (步驟和前面一樣)
假設我們的 public ip 是 10.98.188.3

接著我們要修改設定檔 `/etc/nginx/sites-enable/default`
(或是在同個目錄下建立新的設定檔，但記得要 `unlink` 原本的設定檔)

修改 `server` 區塊如下

```perl
server {
    listen 80;
    server_name 10.98.188.3;

	# 到 hostA 的路徑
    location /hostA {
		proxy_pass http://10.217.44.28;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
    }
	
	# 到 hostB 的路徑
    location /hostB {
        proxy_pass http://10.217.44.6;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
		proxy_set_header X-Forwarded-Proto $http_x_forwarded_proto;
    }
	
	# 其他設定 
}

```
<br>

`proxy_pass` 代表要轉送的位址 (也就是內部 host 的位址)

`proxy_set_header` 可以設定變數來記錄某些資訊，並將這些資訊一併轉送給內部 host，這裡我們定義了三個變數
- `X-Real-IP` 記錄原始請求的 IP
- `X-Forwarded-For` 紀錄原始請求經過的所有 proxy 的 IP (如果沒有那會和 `X-Real-IP` 一樣)
- `X-Forwarded-Proto` 紀錄原始請求的 protocol header

<div style="page-break-after:always;"></div>