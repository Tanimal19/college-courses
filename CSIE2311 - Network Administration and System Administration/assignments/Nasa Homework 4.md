B11902038 資工一 鄭博允

---

# DNS & DHCP
## 1.  Build DNS and DHCP server
> Reference : 
[1] ChatGPT
[2] Lab7

使用 virtualbox 建立兩台 VM
<mark style="background: #ADCCFFA6;">**server**</mark> : debian 11  (iso : https://www.debian.org/download)
網路設置 NAT ／ 內部網路(internet)

<mark style="background: #FFB8EBA6;">**client**</mark> : ubuntu 20.04 (iso : https://releases.ubuntu.com/focal/)
網路設置 內部網路(internet)

### 1. On <mark style="background: #ADCCFFA6;">**server**</mark>

**事前網路設定：**
NAT 的那張直接用 DHCP，內部網路的那張照下圖設定
![[School/Course Homeworks/NASA/assets/nasa hw4/server-1.png|375]]

設定完後用 `ip a` 確認網卡資訊
![[School/Course Homeworks/NASA/assets/nasa hw4/server-2.png|475]]
<div style="page-break-after:always;"></div>

**架設 DNS server **

```bash
# 安裝所需套件
sudo apt update
sudo apt install bind9 isc-dhcp-server
```

設定以下檔案

/etc/bind/named.conf.local

![[School/Course Homeworks/NASA/assets/nasa hw4/named.conf.local.png|450]]

/etc/bind/zones/b11902038.com

![[School/Course Homeworks/NASA/assets/nasa hw4/b11902038.com.png|525]]

/etc/bind/zones/b11902038.rev

![[School/Course Homeworks/NASA/assets/nasa hw4/b11902038.rev.png|525]]

```bash
# 開啟 bind9 服務
sudo systemctl start bind9
# 或是重啟
sudo systemctl restart bind9
```
<div style="page-break-after:always;"></div>

**架設 DHCP server **

在 /etc/dhcp/dhcpd.conf 加入以下文字

![[School/Course Homeworks/NASA/assets/nasa hw4/dhcpd.conf.png|425]]

在 /etc/default/isc-dhcp-server 中將網卡設定成 內部網路 的那張
```bash
INTERFACESv4="enp0s8"
```

```bash
# 開啟 dhcp 服務
sudo systemctl start isc-dhcp-server
# 或是重啟
sudo systemctl restart isc-dhcp-server
```

<br>
<br>
<br>

### 2. On <mark style="background: #FFB8EBA6;">**client**</mark>

`ip a` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-1.png]]

<div style="page-break-after:always;"></div>

`dig www.b11902038.com` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-21.png]]

`dig www.b11902038.com @192.168.5.1` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-22.png]]
<div style="page-break-after:always;"></div>

`dig google.com` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-31.png]]

`dig google.com @192.168.5.1` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-32.png]]
<div style="page-break-after:always;"></div>

`dig -x 1.2.3.4` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-41.png]]

`dig -x 1.2.3.4 @192.168.5.1` 截圖

![[School/Course Homeworks/NASA/assets/nasa hw4/client-42.png]]
<div style="page-break-after:always;"></div>

## 2. Short Answer

1. $\quad$
	> Reference : https://www.cloudflare.com/zh-tw/learning/dns/dns-cache-poisoning/
	 
	```
	[Round 1]
	Average Latency (s):  0.217678 (min 0.000142, max 3.772244)
	
	[Round 2]
	Average Latency (s):  0.024628 (min 0.000218, max 2.400520)
	
	[Round 3]
	Average Latency (s):  0.016325 (min 0.000176, max 3.029017)
	```

	從測試結果可以發現前兩次的查詢速度較慢，第三次的查詢則快上許多，這有可能是因為 **DNS cache (快取)** 的機制。該機制會記錄使用者之前查詢過的結果，以便下次做相同查詢時可以快速回應，快取會持續留存直到其 TTL 值歸零或是由使用者自行清除。
	 
	一台剛建好的 DNS server，裡面還沒有任何快取，因此前幾次查詢必須經過許多伺服器而速度較慢，到了第三次因為已經有快取，所以查詢速度變快許多。
	
	---
	 
	**DNS cache poisoning (快取中毒)**  即是利用此種機制產生的攻擊，攻擊者偽裝成其他的 DNS server 並回應錯誤的資訊給使用者的 DNS，錯誤的資訊便會因為快取機制留存一段時間，只要 cache 沒有被清除，使用者就會一直被導引到錯誤的網址。
	
	 要防止該攻擊可以利用 **DNSSEC**，該機制會透過公開金鑰來檢查 DNS 的來源是否合法，以防止攻擊者偽造訊息。

<div style="page-break-after:always;"></div>

2. $\quad$
	> Reference: https://tw.godaddy.com/zh/help/what-factors-affect-dns-propagation-time-1746
	
	**DNS propagation time** 是指當使用者修改了 DNS 紀錄後 (像是更改域名、ip)，該更新傳播到全球所花費的時間，通常會是數小時到一天。
	
	**TTL** 是指一筆紀錄停留在 DNS cache 的時間，即便修改了紀錄，在舊紀錄的 TTL 尚未到期前 DNS server 仍會回應舊的資訊，直到舊紀錄消失才會套用更新後的資訊。長的 TTL 可能會使 DNS propagation time 變久；短的 TTL 雖然可以加快更新速度，但可能會增加查詢時間，因為每一筆相同查詢可能都要重新連接許多伺服器。

 <br>

 3. $\quad$
	 > Reference : https://zh.wikipedia.org/zh-tw/DNS_over_HTTPS
	
	 **DNS-over-HTTPS (DoH)** 即使用 HTTPS 協定來進行 DNS 查詢，優點是可以避免查詢過程資料被竊聽或是修改，缺點是必須經過更多次的數據傳遞而增加查詢時間。

<Br>

4. $\quad$
	> Reference : ChatGPT
	
	 我們可以將 DHCP server 和 client 配置在不同子網段，只要透過 **DHCPR** 即可。DHCPR 又稱 DHCP Relay (中繼)，當一個子網段內沒有 DHCP server，但是有 DHCPR 時，DHCPR 會做為中繼站將 client 的請求轉發給位於其他網段的 DHCP server 並取得回應再傳回給 client。 

 <br>

 5. $\quad$
	 > Reference : https://zh.wikipedia.org/zh-tw/DHCP_snooping
	
	 DHCP snooping 是一種安全機制，確保使用者永遠是從合法的 DHCP server 取得 ip，假如該 ip 是由非信任的 DHCP server 回應，則會丟棄該回應。簡單來講就是一種防火牆？

<div style="page-break-after:always;"></div>

# LDAP

## Basic Setup

### 1. 
就架 VM，沒什麼好寫的

安裝 OpenLDAP
```bash
apt install -y slapd ldap-utils
apt install ldapvi
```

### 2.
> Reference : Lab8

編輯 suffix.ldif 
```bash
# suffix.ldif
dn: olcDatabase={1}mdb,cn=config
changetype: modify
replace: olcSuffix
olcSuffix: dc=nasa,dc=csie,dc=ntu
```

套用設定
```bash
ldapmodify -Y EXTERNAL -H ldapi:/// -f suffix.ldif
```

---

編輯 rootdn.ldif
```bash
# rootdn.ldif
dn: olcDatabase={1}mdb,cn=config
changetype: modify
replace: olcRootDN
olcRootDN: cn=admin,dc=nasa,dc=csie,dc=ntu
```

套用設定
```bash
ldapmodify -Y EXTERNAL -H ldapi:/// -f rootdn.ldif
```

<div style="page-break-after:always;"></div>

編輯 base.ldif
```bash
# base.ldif
dn: dc=nasa,dc=csie,dc=ntu
dc: nasa
objectClass: top
objectClass: domain
 
dn: cn=admin,dc=nasa,dc=csie,dc=ntu
cn: admin
objectClass: organizationalRole
description: admin account
 
dn: ou=people,dc=nasa,dc=csie,dc=ntu
ou: people
objectClass: organizationalUnit
 
dn: ou=group,dc=nasa,dc=csie,dc=ntu
ou: group
objectClass: organizationalUnit
```

套用設定
```bash
ldapadd -D cn=admin,dc=nasa,dc=csie,dc=ntu -W -H ldapi:/// -f base.ldif
```

---

`ldapsearch` 截圖
![[School/Course Homeworks/NASA/assets/nasa hw4/3-1.png]]

<div style="page-break-after:always;"></div>

### 3.
> Reference : 
[1] ChatGPT
[2] https://www.openldap.org/doc/admin23/tls.html

使用 openssl 建立 ssl 憑證和私鑰 (輸入相關資訊)
```bash
sudo openssl req -newkey rsa:2048 -nodes -keyout /etc/ssl/private/ldap-server.key -out /etc/ssl/certs/ldap-server.csr
```

為憑證簽名
```bash
sudo openssl x509 -req -in /etc/ssl/certs/ldap-server.csr -out /etc/ssl/certs/ldap-server.crt -signkey /etc/ssl/private/ldap-server.key -days 3650
```

在 /etc/ldap/slapd.d/cn=config/olcDatabase={1}mdb.ldif 加入以下文字
```bash
olcTLSCACertificateFile: /etc/ssl/certs/ca-certificates.crt
olcTLSCertificateFile: /etc/ssl/certs/ldap-server.crt
olcTLSCertificateKeyFile: /etc/ssl/private/ldap-server.key
```

啟用 ldaps 監聽，更改 /etc/default/slapd
```bash
SLAPD_SERVICES="ldap:/// ldapi:/// ldaps:///"
```

將驗證開啟，在 /etc/ldap/ldap.conf 加入下行
```bash
TLS_REQCERT allow
```

更改憑證、私鑰的權限為 600
```bash
sudo chmod 600 /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ldap-server.crt /etc/ssl/private/ldap-server.key 
```

重啟服務
```bash
sudo systemctl restart slapd
```

---

截圖
![[School/Course Homeworks/NASA/assets/nasa hw4/3-2.png]]

<div style="page-break-after:always;"></div>

## Client 

## 1. 
> Reference : https://blueskyson.github.io/2021/09/16/virtualbox-arch-linux/

架 VM，沒什麼好寫的

## 2.

安裝 OpenLDAP
```bash
sudo pacman -S openldap
```

編輯 /etc/openldap/ldap.conf
將 BASE 替換成 server 的 suffix、URI 替換成 server 的 ip
```bash
BASE dc=nasa,dc=csie,dc=ntu
URI ldap://192.168.0.201
```

`ldapsearch` 截圖
![[School/Course Homeworks/NASA/assets/nasa hw4/3-3.png|500]]

<div style="page-break-after:always;"></div>

### 3.
> Reference : 
[1] ChatGPT
[2] https://www.openldap.org/doc/admin23/tls.html

在 server 端的 /etc/ldap/slapd.d/cn=config/olcDatabase={1}mdb.ldif 新增下行
```bash
olcTLCVerifyClient: demand
```

重啟服務
```bash
sudo systemctl restart slapd
```

在 client 端的 /etc/openldap/ldap.conf 新增下列文字
```bash
URI ldaps://192.168.0.201
TLS_CACERT /etc/ssl/certs/ca-certificates.crt
TLS_REQCERT demand
```

重啟服務
```bash
sudo systemctl restart nscd
```

連線失敗截圖
![[School/Course Homeworks/NASA/assets/nasa hw4/3-4.png]]

<div style="page-break-after:always;"></div>

### 4.
> Reference :
[1] ChatGPT
[2] https://dic.vbird.tw/linux_server/unit07.php

安裝 sssd
```bash
sudo pacman -S sssd
```

新增 /etc/sssd/sssd.conf
```bash
[sssd]
domains = nasa.csie.ntu
services = nss, pam

[nss]

[pam]

[domain/nasa.csie.ntu]
id_provider = ldap
auth_provider = ldap
ldap_uri = ldaps://192.168.0.201
ldap_search_base = dc=nasa,dc=csie,dc=ntu
ldap_tls_reqcert = demand
ldap_tls_cacert = /etc/ssl/certs

[ssh]
authorized_keys_command = /usr/bin/sss_ssh_authorizedkeys
```

啟動服務 
```bash
sudo systemctl restart sssd
```

ssh 連線失敗
```bash
ssh 192.168.0.201
ssh: connect to host 192.168.0.201 port 22: Connection refused
```


卡在這題很久之後，我已經不知道自己到底在做什麼設定了，
查了很多文件完全搞不懂，後面的東西我也沒有做，這次作業只能做到這了，
感謝助教看到這裡，辛苦了。
