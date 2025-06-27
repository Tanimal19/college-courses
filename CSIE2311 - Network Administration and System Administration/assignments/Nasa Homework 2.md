B11902038 資工一 鄭博允

---
# Network Administration

## 1. IPerf3
> Reference :
> https://www.cnblogs.com/linyu51/p/14133379.html

**前置作業**
筆電(win11)：去[官網](https://iperf.fr/iperf-download.php) 下載 IPerf3，然後將 iperf.exe 和 cygwin1.dll 複製到 `%systemroot%` 目錄下

手機(ios)：直接安裝 [APP](https://apps.apple.com/fr/app/he-net-network-tools/id858241710)
<br>

**指令操作**
打開 cmd，先用 `ipconfig` 得到筆電的 ip address，
接著輸入 `iperf3 -s` 將筆電作為 server 端。

打開 APP，輸入剛才的筆電 ip address，
接著將傳輸設定調整為 100M (總傳輸量)、TCP，然後就可以開始進行傳輸了。

兩台連接 csie 的裝置
![[School/Course Homeworks/NASA/assets/nasa hw2/iperf3_csie_1.png|525]]
![[School/Course Homeworks/NASA/assets/nasa hw2/iperf3_csie_2.png]]

一台連接 csie 與一台連接 csie-5G 的裝置
![[School/Course Homeworks/NASA/assets/nasa hw2/iperf3_csie_csie_5G_1.png|525]]
![[School/Course Homeworks/NASA/assets/nasa hw2/iperf3_csie_csie_5G_2.png|625]]

兩台連接 csie-5G 的裝置
![[School/Course Homeworks/NASA/assets/nasa hw2/iperf3_csie_5G.png|625]]

看來要兩台都連上 5G 才有用呢 :(
<br>

## 2. NeVeR_LosEs’ PC
### 1.

```bash
dig www.google.com AAAA
```
![[School/Course Homeworks/NASA/assets/nasa hw2/googleipv6.png|450]]

可以得知 www.google.com 的 ipv6 為 ` 2404:6800:4012::2004`

<br>

### 2.

> Reference :
> https://www.freedesktop.org/software/systemd/man/systemd.service.html
> https://docs.python.org/3/library/http.server.html
> https://www.runoob.com/linux/linux-comm-ip.html
> https://serverfault.com/questions/842542/why-are-ipv6-addresses-flushed-on-link-down
> https://sysctl-explorer.net/net/ipv6/keep_addr_on_down/
> https://www.lightnetics.com/topic/13002/bash-systemd-command-not-found/2
> https://pimylifeup.com/ubuntu-add-to-path/
> https://www.lightnetics.com/topic/13002/bash-systemd-command-not-found/2
> https://pimylifeup.com/ubuntu-add-to-path/

先照著提示看看指令的內容
![[School/Course Homeworks/NASA/assets/nasa hw2/systemctl1.png]]

這個 service 執行失敗，於是我查看檔案內容
```bash
cat /etc/systemd/system/never-loses-website.service
```
我不知道他在寫些什麼，所以我上網查了 `.service` 的格式，問題似乎是出在
```bash
ExecStart=/start-website
```
所以我又去查看該檔案內容，看起來是執行檔，就順便執行了一下
```bash
cat /start-website
```
![[School/Course Homeworks/NASA/assets/nasa hw2/systemctl3.png]]

我根據指令內容去查詢了用法，好像是 `python3 -m http.server` 少了一個參數，而這個參數是由 `ip -f inet6 addr show enp0s3` 輸出的，於是我嘗試單獨執行該指令，照理來說應該會出現網卡 enp0s3 的 ipv6 位址資訊，結果什麼都沒出現，於是我決定直接查看整張網卡的資訊
```bash
ip addr show enp0s3
```
結果發現網卡根本沒有 inet6 的欄位，我 google 了很久，但找不到能夠不需 root 權限就能讓 ipv6 的位址出現的方法。


於是我決定重頭開始，又重新輸入了一次指令，然後它突然就成功了？
![[School/Course Homeworks/NASA/assets/nasa hw2/systemctl4.png]]

我不知道要怎麼辦了，問題到底出在哪？

我又想到，題目所指的 systemd 服務會不會是單純指 `systemd` 這條指令呢？實際打了發現真的不能用，應該是沒有加入環境變數的問題，所以用指令將它加進去就好了吧
```bash
export PATH="/usr/lib/systemd:$PATH"
```
好，現在可以用 `systemd` 了，但好像還是沒有成功。

不寫了，不寫了，我要去被 DSA 虐了。

<br>


## 3. Bocchi’s バイト
### 1. Day 0: 孤獨面試
> Reference :
> https://ithelp.ithome.com.tw/articles/10230879
> [Lab 4 slides](https://docs.google.com/presentation/d/1XmyCq87cnAmbj8V9UOAm6LqeM1J1pxmzkBV2PC19g8w/edit#slide=id.g21b499ff8ba_0_0)


1. 設定主機名稱
	```
	Switch1(config)# hostname Bocchi
	Switch2(config)# hostname Rock
	```

2. 設定 Switch1 的 Enable 密碼
	```
	Bocchi(config)# enable password Bocchi
	Bocchi(config)# service password-encryption
	```

3. 設定 Switch1 的 Console 密碼
	```
	Bocchi(config)# line console 0
	Bocchi(config-line)# password Bocchi
	Bocchi(config-line)# login
	```

4. VLAN 相關設定
	```
	// 建立 vlan 10,20
	Bocchi(config)# vlan 10
	Bocchi(config)# vlan 20
	
	// 將 FastEthernet 0/1 設置在 vlan 10
	Bocchi(config)# interface FastEthernet 0/1
	Bocchi(config-if)# switchport mode access
	Bocchi(config-if)# switchport access vlan 10
	
	// 將 FastEthernet 0/2 設置在 vlan 20
	Bocchi(config)# interface FastEthernet 0/2
	Bocchi(config-if)# switchport mode access
	Bocchi(config-if)# switchport access vlan 20
	
	// 將 FastEthernet 0/1-2 設置為 trunk
	Bocchi(config)# interface range GigabitEthernet 0/1-2
	Bocchi(config-if)# switchport mode trunk
	Bocchi(config-if)# switchport trunk allowed vlan 10,20,99
	
	// 建立 port-channel 1
	Bocchi(config)# interface port-channel 1
	Bocchi(config-if)# switchport mode trunk
	Bocchi(config-if)# switchport trunk allowed vlan 10,20,99
	
	// 將 GigabitEthernet 0/1-2 加入 port-channel 1
	Bocchi(config-if)# interface range GigabitEthernet 0/1-2
	Bocchi(config-if-range)# channel-group 1 mode active
	
	---
	
	// 對 Rock 做一樣的事情
	Rock(config)# interface FastEthernet 0/1
	Rock(config-if)# switchport mode access
	Rock(config-if)# switchport access vlan 10
	
	Rock(config)# interface FastEthernet 0/2
	Rock(config-if)# switchport mode access
	Rock(config-if)# switchport access vlan 20
	
	Rock(config)# interface GigabitEthernet 0/1-2
	Rock(config-if)# switchport mode trunk
	Rock(config-if)# switchport trunk allowed vlan 10,20
	
	Rock(config)# interface port-channel 1
	Rock(config-if)# switchport mode trunk
	Rock(config-if)# switchport trunk allowed vlan 10,20,99
	
	Rock(config-if)# interface range GigabitEthernet 0/1-2
	Rock(config-if-range)# channel-group 1 mode active
	```

5. 設定 Admin
	```
	// 建立 vlan 99
	Bocchi(config)# vlan 99
	
	// 將 FastEthernet 0/24 設置在 vlan 99
	Bocchi(config)# interface FastEthernet 0/24
	Bocchi(config-if)# switchport mode access
	Bocchi(config-if)# switchport access vlan 99
	
	// ssh 連線設定
	Bocchi(config)# ip domain-name Bocchi.com
	Bocchi(config)# ip address 192.168.0.100 255.255.255.0
	Bocchi(config)# crypto key generate rsa
	Bocchi(config)# ip ssh version 2
	
	Bocchi(config)# line vty 0 15
	Bocchi(config-line)# login local
	Bocchi(config-line)# transport input ssh
	
	Bocchi(config)# username Bocchi secret Bocchi
	
	---
	
	// 對 Rock 做一樣的事情
	Rock(config)# ip domain-name Rock.com
	Rock(config)# ip address 192.168.0.101 255.255.255.0
	Rock(config)# crypto key generate rsa
	Rock(config)# ip ssh version 2
	
	Rock(config)# line vty 0 15
	Rock(config-line)# login local
	Rock(config-line)# transport input ssh
	
	Rock(config)# username Bocchi secret Bocchi
	```
	
<br>

### Day 2: 喜多風暴

1. 找出電腦名稱
	```
	C:\> ping 192.168.0.53
	C:\> arp -a
	```
	
	得到實體位址為 `0001.42b9.2d05`
	
	```
	C:\> telnet 192.168.0.250
	Core# ping 192.168.0.53
	Core# show mac addr
		0001.42b9.2d05   Fa0/1
	```
	
	可以知道該位址是接到 `Core` 的 `Fa0/1`，用同樣方法依序查找
	
	```
	Core# telnet 192.168.0.251
	Edge1# ping 192.168.0.53
	Edge1# show mac addr
		0001.42b9.2d05   Fa0/2
	
	Edge1# telnet 192.168.0.244
	LAb241# ping 192.168.0.53
	LAb241# show mac addr
		0001.42b9.2d05   Fa0/1
	```

	最後得知該電腦是接到 `Lab241` 的 `Fa0/1`，即為 **PC-PT 421-1**

2. 隔離失控電腦

	我的想法是在離失控電腦最近的 switch 上建立 ACL (Access Control List) 以阻斷所有來自 **PC-PT 421-1** (`192.168.0.53`) 的流量。
	然而 Packet Tracer 內的 Lab421 switch 不知道為什麼不能進入 Config 模式，而 Edge1 又沒有辦法使用 ACL (可以新增條件，但不能套用到 interface)。  
	因此以下只有指令，而沒有實作
	```
	Edge1(config)# access-list 10 deny host 192.168.0.53
	Edge1(config)# interface fa0/1
	Edge1(config-it)# ip access-group 10 in
	```

<br>

# System Administration
## 1. Short Answer
### **(a)** 
> Refernce
> https://www.mobile01.com/topicdetail.php?f=494&t=4683363

Btrfs 和 ZFS 都是支援 COW (copy-on-write) 的檔案系統，也就是說，他們都具有快照功能。它們最主要的差別在於支援的儲存容量，Btrfs 只適合在 1 ~ 4 顆 HD 的儲存系統上，而 ZFS 可以在大型的儲存系統上運作。
<br>

### **(b)**

> Reference
> https://www.youtube.com/watch?v=U-OCdTeZLac&t=150s

以下假設一顆硬碟容量為 1 ，有 n 顆。

**RAID 0**
將資料分散在所有硬碟，至少需要 2 顆硬碟，有效容量為 n。
優點：存取速度快 / 容量效率高
缺點：一顆硬碟損毀，資料就無法復原

**RAID 1**
將資料做鏡像，也就是複製到其他顆硬碟，至少需要 2 顆硬碟，有效容量為 1。
優點：資料有很多備份
缺點：容量效率極低

**RAID 5**
資料分散在所有硬碟，但每一顆硬碟會多儲存一組奇偶校驗和，
至少需要 3 顆硬碟，有效容量為 n-1 。
優點：就算有一顆硬碟毀損，資料還是可以復原
缺點：會少一顆硬碟的容量 / 無法承受同時兩顆硬碟毀損

**RAID 10**
即為 RAID 0 + RAID 1
將資料分散再分別做備份，至少需要 4 顆硬碟，有效容量為 n/2。
優點：兼具 RAID 0 的存取速度，還有 RAID 1 的備份
缺點：有效容量只有一半
<br>

### **(c)**
> Reference
> https://www.jianshu.com/p/c2b77d0bbc43
> https://zh.wikipedia.org/zh-tw/FUSE

一般的檔案系統 (Btrfs, exFAT, ext4 ...) 都是在 **Kernel** 中運作，而 FUSE 則是在 **Userspace** 中運作，它的優點在於讓使用者可以在不更動 **Kernel** 的狀況下建立自己的檔案系統，或是修改檔案系統。而缺點是效能的降低，這點要從它的運作方式來看，FUSE 會頻繁的在 **Kernel** 與 **Userspace** 間切換，而導致額外的開銷。

FUSE 的運作方式 (wiki)

<br>

## 2. 與其他作業系統共用檔案
> Reference
> https://www.linwei.com.tw/forum-detail/60/
> https://ithelp.ithome.com.tw/m/articles/10294967
> https://chat.openai.com/chat

根據題目，檔案系統必須符合以下要件：
Windows、Linux 原生支援讀寫
支援 4GiB 以上的檔案
在 Windows 上可以使用檔案日誌

![[School/Course Homeworks/NASA/assets/nasa hw2/CHATGPT.png]]

在 ChatGPT 的建議下，決定採用 **NTFS** 系統。
**FAT32** (無法支援 4GiB 以上的檔案)
**exFAT** (沒有檔案日誌)

```bash
# 安裝 ntfs-3g 附件包(才可以使用 mkfs.ntfs)
sudo pacman -S ntfs-3g

# 建立掛載點
sudo mkdir /mnt/usbdisk

# 格式化成 ntfs
sudo mkfs.ntfs /dev/sdj2

# 掛載
sudo mount /dev/sdj2 /mnt/usbdisk

# 查看 UUID
sudo blkid /dev/sdj2

# 修改 fstab (開機自動掛載)
sudo vim /etc/fstab

# 在 fstab 新增一行
UUID=18EB86C43578C04D   /mnt/usbdisk   ntfs   defaults   0 2
```

結果螢幕截圖：
![[School/Course Homeworks/NASA/assets/nasa hw2/2-1.png]]
![[School/Course Homeworks/NASA/assets/nasa hw2/2-2.png]]
<br>

## 3. 記憶體不足？
> Reference
> https://www.ltsplus.com/linux/linux-create-swap-space

```bash
# 建立 3GiB 的空白檔案
sudo dd if=/dev/zero of=/myswap bs=1M count=3072

# 格式化成 swap 檔案
sudo mkswap /myswap

# 啟用 swap
sudo swapon /myswap

# 開機自動啟用
sudo vim /etc/fstab
/myswap   swap   swap   defaults   0 0
```

結果螢幕截圖：
![[School/Course Homeworks/NASA/assets/nasa hw2/3-1.png]]
<br>

## 4. 空間不足
> Reference
> http://c.biancheng.net/view/920.html

```bash
# 查看目前 LV 的資訊 (主要是為了看路徑)
sudo lvdisplay

# 改變現有 LV 大小
sudo lvresize -L 1GiB /dev/nasahw2-main/course
```

結果螢幕截圖：
![[School/Course Homeworks/NASA/assets/nasa hw2/4-1.png]]
![[School/Course Homeworks/NASA/assets/nasa hw2/4-2.png]]
<br>

## 5. 建立加密分割區
> Reference
> https://cloud.tencent.com/developer/article/1836866
> https://chat.openai.com/chat

補：我做完  p8 後發現 `/etc/crypttab`  和 `/etc/fstab` 的部分有問題，所以在 p8 時有做修改，但為了維持整個作業的流暢度，我就沒有改掉這裡。

```bash
# 建立 LV
sudo lvcreate -L 800M -n homework nasahw2-main

# 將 LV 設定成加密區 (然後輸入 YES)
sudo cryptsetup luksFormat /dev/nasahw2-main/homework --key-file /home/elsa/lvm_key

# 開啟加密區並設定名稱 homework
sudo cryptsetup open /dev/nasahw2-main/homework homework --key-file /home/elsa/lvm_key

# 將加密區格式化為 ext4
sudo mkfs.ext4 /dev/mapper/homework

# 掛載
sudo mount /dev/mapper/homework /home/elsa/homework

# 修改 crypttab (開機自動讀取)
sudo vim /etc/crypttab
homework   /dev/nasahw2-main/homework   /home/elsa/lvm_key

# 修改 fstab (開機自動掛載)
sudo vim /etc/fstab
/dev/mapper/homework   /home/elsa/homework   ext4   defaults   0 2
```

結果螢幕截圖：
![[School/Course Homeworks/NASA/assets/nasa hw2/5-1.png]]
![[School/Course Homeworks/NASA/assets/nasa hw2/5-2.png]]

## 6. Extend then Snapshot
> Reference
> https://chat.openai.com/chat

### (a)

```bash
# 將 PV 加入 VG
sudo vgextend nasahw2-main /dev/sdd1
```

### (b)

```bash
# 建立快照 (-s)
sudo lvcreate -s -n backup -L 1GiB nasahw2-main/course

# 建立掛載點
sudo mkdir /mnt/backup

# 掛載
sudo mount /dev/nasahw2-main/backup /mnt/backup
```

### (c)
![[School/Course Homeworks/NASA/assets/nasa hw2/6-1.png]]

### (d)

```bash
# 打包(tar) 和壓縮(zstd)
sudo tar -I 'zstd -T0' -cvf /home/elsa/backup.tar.zst /mnt/backup/
```

使用 `tar` 命令將 `/mnt/backup/` 目錄下的所有檔案打包成壓縮檔 `/home/elsa/backup.tar.zst`。
由於要打包成 `.zst` 檔，用 `-I` 來指定使用的壓縮工具，`'zstd'` 代表使用的壓縮工具，而 `'-T0'` 代表使用最大 CPU 資源，加快壓縮的速度。
後面的選項 `-c` 表示建立壓縮檔案，`-v` 表示輸出詳細的處理資訊，`-f` 表示指定打包文件的文件名。

<br>

### (e)

```bash
# 卸載
sudo umount /mnt/backup

# 刪除 LV
sudo lvremove nasahw2-main/backup
```

<br>

## 7. Switch!
> Reference
> https://blog.csdn.net/mycms5/article/details/27218935
> https://www.joehorn.tw/tag/pvmove/

```bash
# 建立新 PV
sudo pvcreate /dev/sde1

# 將 PV 加入 VG
sudo vgextend nasahw2-secondary /dev/sde1

# 將 PV 的資料轉移
sudo pvmove /dev/sdf1

# 將 PV 從 VG 中移除
sudo vgreduce nasahw2-secondary /dev/sdf1

# 移除 PV
sudo pvremove /dev/sdf1
```

為了讓 `pvmove` 的動作更加明顯，使用 `sudo pvs` 確認，
可以看到`sdf1` 的 PFree 欄位容量增加了，而 `sde1` 的 PFree 欄位容量則減少了，可見資料有被成功轉移。
![[School/Course Homeworks/NASA/assets/nasa hw2/7-1.png|475]]

結果螢幕截圖：
![[School/Course Homeworks/NASA/assets/nasa hw2/7-2.png|475]]

<br>

## 8. Merge Request
> Reference
> https://www.thegeekdiary.com/how-to-merge-2-volume-groups-vgs-into-one-using-vgmerge-in-lvm/
> https://www.thegeekdiary.com/logical-volume-vg-lv-contains-a-filesystem-in-use-while-removing-lvm-filesystem/
> https://blog.roberthallam.org/2017/12/solved-logical-volume-is-used-by-another-device/comment-page-1/
> https://access.redhat.com/documentation/zh-tw/red_hat_enterprise_linux/6/html/logical_volume_manager_administration/dmsetup
> https://chat.openai.com/chat

先停用所有和 VG 相關的 LV，結果出現錯誤
```bash
sudo lvchange -a n nasahw2-main/course
Logical volume nasahw2-main/course contains a filesystem in use

sudo lvchange -a n nasahw2-main/homework
Logical volume nasahw2-main/homework is used by another device

sudo lvchange -a n nasahw2-secondary/videos
Logical volume nasahw2-secondary/videos contains a filesystem in use
```

錯誤 `contains a filesystem in use` 是因為還有檔案掛載其中，解決辦法：
```bash
# 找出和 nasahw2 的掛載
mount | grep /nasahw2

# 卸載
sudo umount /dev/nasahw2-main/course
sudo umount /dev/nasahw2-secondary/videos
```

而錯誤 `cis used by another device` 的解決辦法：
```bash
# 找出還在運作的 device
sudo dmsetup info -c

# 將 device 移除
sudo dmsetup remove homework
```

然後就可以順利執行了
```bash
sudo lvchange -a n nasahw2-main/course
sudo lvchange -a n nasahw2-main/homework
sudo lvchange -a n nasahw2-secondary/videos
```

接著將要合併進去的 VG (`nasahw2-secondary`) 停用
```bash
# 停用 VG
sudo vgchange -a n nasahw2-secondary

# 合併 VG
sudo vgmerge nasahw2-main nasahw2-secondary

# 重新啟用 VG 的 LV
sudo lvchange -a y nasahw2-main
```

接著要將檔案復原 (也就是重新掛載)
```bash
sudo mount /dev/mapper/nasahw2--main-course /home/elsa/course
sudo mount /dev/mapper/nasahw2--main-videos /home/elsa/videos
```

接著把電腦重開機試試看會不會自動掛載

......
沒有成功，開機的時候爆炸 (sulogin) 了
好，該怎麼辦？總之先把日誌叫出來
```bash
journalctl -b -1 -p warning 
```
![[School/Course Homeworks/NASA/assets/nasa hw2/damnifuckedup2.png]]

看起來問題是出在 `/dev/nasahw2-secondary/videos` 還有 `/dev/mapper/homework`

先解決第一個，到 `/etc/fstab` 裡，把
`/dev/nasahw2-secondary/videos` 改成 `/dev/nasahw2-main/videos`
因為 `nasahw2-secondary` 已經不存在了，難怪爆炸。

然後是第二個問題，我 `cd /dev/mapper` 後發現 `homework` 不見了，
好像是因為我前面把用 `dmsetup remove` 把它移除了 (所以前面好像做錯了？)
於是我只好把 p5 的步驟重作一遍，也就是
```bash
sudo cryptsetup luksFormat /dev/nasahw2-main/homework --key-file /home/elsa/lvm_key
sudo cryptsetup open /dev/nasahw2-main/homework homework --key-file /home/elsa/lvm_key
sudo mkfs.ext4 /dev/mapper/homework
sudo mount /dev/mapper/homework /home/elsa/homework
```

為了避免下次又出錯，我這次決定用 UUID，不用路徑了
```bash
sudo blkid /dev/mapper/homework
UUID="1d11b49b-c358-4dff-9739-d778bdc7c6b4" BLOCK_SIZE="4096" TYPE="ext4"

sudo blkid /dev/nasahw2-main/homework
UUID="ea982ca2-70f7-4579-b60a-56b0d45ffe36" TYPE="crypto_LUKS"
```

然後再修改 `/etc/crypttab` 和 `/etc/fstab`
```bash
sudo vim /etc/crypttab
homework   UUID=ea982ca2-70f7-4579-b60a-56b0d45ffe36   /home/elsa/lvm_key

sudo vim /etc/fstab
UUID=1d11b49b-c358-4dff-9739-d778bdc7c6b4   /home/elsa/homework   ext4   defaults   0 2
```

接著我就重開一次，
好耶，他又爆炸了！？

不知道為什麼每次 `/dev/mapper/homework` 都會消失，所以我決定祭出殺手鐧，掛載不了就別掛了，編輯 `/etc/fstab` 
```bash
/dev/mapperhomework   /home/elsa/homework   ext4
_netdev,x-systemd.device-timeout=1,defaults    0 0
```
其中 `_netdev` 將分割區視為外部硬碟，避免掛載不了就爆炸，而 `x-systemd.device-timeout=1` 則是當找不到 device 時，1 秒後就放棄。

這樣重新開機後就正常了 (只是要重新加密 `/dev/nasahw2-main/homework` )

我不確定這樣算不算成功，還是附上結果螢幕截圖：
![[School/Course Homeworks/NASA/assets/nasa hw2/8-2.png]]

<br>

## 9. RAID: Shadow Legends

### (a)
```bash
# 建立 RAID 0
sudo mdadm --create /dev/md/md0 --level=0 --raid-devices=3 /dev/sdg1 /dev/sdh1 /dev/sdi1
sudo mkfs.ext4 /dev/md/md0
sudo mkdir /mnt/raid0
sudo mount /dev/md/md0 /mnt/raid0

# 建立 RAID 1
sudo mdadm --create /dev/md/md1 --level=1 --raid-devices=3 /dev/sdg2 /dev/sdh2 /dev/sdi2
sudo mkfs.ext4 /dev/md/md1
sudo mkdir /mnt/raid1
sudo mount /dev/md/md1 /mnt/raid1

# 建立 RAID 5
sudo mdadm --create /dev/md/md5 --level=5 --raid-devices=3 /dev/sdg3 /dev/sdh3 /dev/sdi3
sudo mkfs.ext4 /dev/md/md5
sudo mkdir /mnt/raid5
sudo mount /dev/md/md5 /mnt/raid5

# 建立未使用RAID 的分割區
sudo mkfs.ext4 /dev/sdj1
sudo mkdir /mnt/vanilla
sudo mount /dev/sdj1 /mnt/vanilla
```

### (b)

```bash
dd if=/dev/zero of=/mnt/raid0/test bs=512M count=1 oflag=dsync
dd if=/dev/zero of=/mnt/raid1/test bs=512M count=1 oflag=dsync
dd if=/dev/zero of=/mnt/raid5/test bs=512M count=1 oflag=dsync
dd if=/dev/zero of=/mnt/vanilla/test bs=512M count=1 oflag=dsync
```

測速結果 (MB/s)：

|     | RAID 0 | RAID 1 | RAID 5 | Vanilla |
| --- |:------ |:------ |:------ |:------- |
| 1   | 482    | 244    | 238    | 415     |
| 2   | 448    | 236    | 243    | 469     |
| 3   | 467    | 244    | 250    | 466     |
| 4   | 496    | 236    | 241    | 487     |
| 5   | 482    | 241    | 235    | 547     |
| avg | 475    | 240.2  | 241.4  | 476.8   |

### (c)

根據觀察，RAID 1 和 RAID 5 相較於 RAID 0 慢了幾乎一半，這點是合乎預期的，因為 RAID 0 可以在三個分割區同時進行讀寫，而 RAID 1 要對檔案做鏡像、RAID 5 要計算校驗和，速度會比較慢是合理的。
然而和我的預想不同的是 RAID 0 和 Vanilla 的存取速度幾乎一模一樣，照理來說，RAID 0 的存取速度應該會和分割區的數量呈正比而倍數增長，也就是說速度應該會是只有一個磁碟的 Vanilla 的三倍。 

### (d)

首先，電腦硬體的規格以及虛擬機器的設定都會影響到讀寫的速度，這個部分是難以去計算的。
雖然我不是很明白 VM 內部的運作方式，而我的猜測如下：
我的電腦只有一顆磁碟(SSD)，VM 只是運用其中的一部份，對於 VM 來說，它只是用某種方式將磁碟分割成很多塊，然後將它們視為不同分割區，再組成 RAID。
因此在讀寫時雖然看起來是同時對不同分割區做處理，但實際上只有對同一顆硬碟做讀寫，也就是說，RAID 0 雖然看似是三個分割區同時做讀寫，實際上背後只有一顆磁碟在做讀寫，所以速度會和 Vanilla 差不多。而 RAID 1 和 RAID 5 因為要另外做鏡像和計算校驗和所以速度會更慢。
