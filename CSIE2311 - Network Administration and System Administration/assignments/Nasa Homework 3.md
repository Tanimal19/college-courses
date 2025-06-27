B11902038 資工一 鄭博允 

---
# Network Administration
## Short Answers
> Reference :
> https://docs.opnsense.org/manual/firewall.html
> https://www.reddit.com/r/PFSENSE/comments/6vyqw3/what_is_the_difference_between_the_interface_net/
> https://lin0204.blogspot.com/2017/01/blog-post_30.html

1. 兩者皆會拒絕流量，差別在於 Reject 會向發送端 (只限 TCP & UDP) 傳送訊息，而 Block 不會作出任何回應。
一般而言，Block 通常被使用在 WAN，當有未知流量傳入時，發送端不會知道防火牆的存在， 可以降低被攻擊的機率；Reject 通常被使用在 LAN，可以減少發送端等待回應的時間。

2. interface net 代表整個網段，像是 `192.168.1.0/24`；
interface address 代表單一的 IP address，像是 `192.168.1.1`

3. Stateful Firewall (狀態防火牆) 可以追蹤通過的封包資訊，透過比對通過的封包和狀態表來判斷是否合法。OPNSense 即為此種類型的防火牆。
Stateless Firewall (無狀態防火牆) 只依據使用者的設定來運行，不會追蹤封包資訊。
舉例來說，今天設定了一個只出不進的防火牆，牆內流量可以 出到牆外，反之則不行。如果是無狀態防火牆，我們會發現無法從牆內 `Ping` 到牆外，原因是 `Ping` 的回傳封包被擋下來了。而如果是狀態防火牆，它會記錄 `Ping` 封包的資訊並在回傳時讓其通過，我們便可以成功從牆內 `Ping` 到牆外。

<div style="page-break-after:always;"></div>

## OPNSense

> Reference :
https://hackmd.io/3sYCHRBNTIuMiJEIhLdckg#/
https://docs.opnsense.org/manual/aliases.html
https://kifarunix.com/how-to-enable-secure-shell-ssh-server-on-opnsense/
https://docs.opnsense.org/manual/firewall.html


### 1. 

依照 LAB 6 的方法建立 VLAN 5，
接著照下圖設定 vlan 的網域 : **Interfaces > [OPT5]**
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-1-1.png]]

照下圖設定 DHCP : **Services > DHCPv4 > [OPT5]**
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-1-2.png]]

我不太確定 DHCP lease 要包含 DNS server 是什麼意思，照理來說我們不能出借 DNS server 的 ip 給使用者吧？
所以我(擅自)認為這是指要將 8.8.8.8 和 8.8.4.4 作為預設的 DNS server，就把 [OPT5] 的 DNS servers 填入 8.8.8.8 和 8.8.4.4
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-1-3.png]]

vlan8 和 vlan99 也照同樣方式設定。
<div style="page-break-after:always;"></div>

### 2.
alias 的意思為「別名」，它有點像分組，把不同 ip、port 等等放在一起並給予一個 alias，之後只要引用這個 alias 就可以代指裡面的所有物件。

**Firewall > Aliases** : 點選 `add` 新增 alias，並依下圖設定 (以 GOOGLE_DNS 為例) 
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-2-1.png]]
完成後應該會長這樣：
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-2-2.png]]
<br>

### 3.

**System > Settings > Administration** : 依下圖設定
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-3-1.png]]

題目說要使用 ADMIN_PORTS 進行連線，但我找不到可以設定多個 port 的地方，用 alias 也不行，所以就用預設的 port 22。
為了測試，我後來有把 root user login 給勾起來。
<div style="page-break-after:always;"></div>

### 4.
依據題目要求，設定 firewall rules 如下：
允許 vlan99 到 192.168.1.1/24 (OPNSense) 的所有連線
允許 vlan99 到 GOOGLE_DNS 的所有連線
允許 vlan99 到 CSIE_WORKSTATIONS 的所有連線

![[School/Course Homeworks/NASA/assets/nasa hw3/na2-4-1.png]]

很遺憾地，我的 opnsense 好像壞了，雖然 LAN 是好的(vlan 可以互通)，但是它不能連到 WAN。
在 opnsense 本體上 `ping 8.8.8.8` 會得到 `No route to host`，
我查了一下，沒有找到解決辦法，似乎是版本問題？
所以我沒有辦法附上 `traceroute` 到CSIE_WORKSTATIONS 的截圖。
而 ssh 到 OPNSense 的截圖如下：

![[School/Course Homeworks/NASA/assets/nasa hw3/na2-4-3.png]]

<br>

### 5.
依據題目要求，設定 firewall rules 如下：
在 vlan5 interface 上允許所有從 vlan5 到 vlan8 的流量
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-5-1.png]]

在 vlan8 interface 上封鎖所有從 vlan8 到 vlan5 的流量
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-5-2.png]]

<br>

### 6.
**Firewall > Settings > Schedules** : 新增新的 schedule 如下
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-6-1.png]]

接著到 vlan5 interface 設定封鎖所有流量的 firewall rule 並套用 schedule，完成後如下：
![[School/Course Homeworks/NASA/assets/nasa hw3/na2-6-2.png]]
<div style="page-break-after:always;"></div>

---
# System Administration
## KVM & Virsh
### 1. 新增VM
> Reference :
https://blog.gtwang.org/linux/kvm-qemu-virt-install-command-tutorial/
https://linux.die.net/man/1/virt-install


建立 VM 要用的虛擬磁碟檔
```bash
qemu-img create -f qcow2 /tmp2/b11902038/ubuntu.qcow2 20G
```

建立 VM
```bash
virt-install \
  --name=b11902038 \
  --vcpus=2 \
  --memory=8192 \
  --disk path=/tmp2/b11902038/ubuntu.qcow2,format=qcow2 \
  --network user,mac=52:54:F8:90:20:38 \
  --cdrom=/tmp2/nasa-hw3/ubuntu.iso \
  --os-variant=ubuntu22.04 \
  --graphics vnc,listen=0.0.0.0
```

查看該 VM 的 VNC 連接埠
```bash
virsh vncdisplay b11902038
:0
```

打開 VNC Viewer 輸入 `linux1.csie.ntu.edu.tw:0` 然後連到 VM 裡面，
第一次會先做 ubuntu 的設定，一路跟著說明走並設定 username 與 hostname

<div style="page-break-after:always;"></div>

工作站 (linux1) 上 `virsh list` 的輸出畫面
![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-1-1.png|300]]

VM 內開機完成的畫面
![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-1-2.png]]

<div style="page-break-after:always;"></div>

VM 內登入後的畫面
![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-1-3.png]]

VM 內執行ip a 指令的輸出畫面
![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-1-4.png]]

<div style="page-break-after:always;"></div>

### 2. 使用 console 進入 VM
> Reference :
https://stackoverflow.com/questions/11845280/virsh-console-hangs-at-the-escape-character

先在 VM 裡開啟 serial console 的服務，並重新啟動 VM
```bash
systemctl enable serial-getty@ttyS0.service
systemctl start serial-getty@ttyS0.service
reboot
```

在工作站 (linux1) 輸入
```bash
virsh console b11902038
```
![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-2-1.png]]

<div style="page-break-after:always;"></div>

### 3. 建立快照
> Reference : 
https://godleon.github.io/blog/KVM/KVM-Template-And-Snapshot/
ChatGPT

VM 
```bash
# 安裝 qemu-guest-agent 插件
sudo apt-get update
sudo apt-get install qemu-guest-agent

# 檢查是否運作
systemctl status qemu-guest-agent
systemctl start qemu-guest-agent
```

工作站
```bash
# 建立快照
virsh snapshot-create-as b11902038 b11902038_snapshot --disk-only --atomic --quiesce

# 列出 VM 的所有快照
virsh snapshot-list b11902038
```

- `--disk-only` 只對 disk 作快照 
- `--atomic` 如果沒有成功建立快照，則會回復原本狀態
- `--quiesce` 讓尚未寫入 disk (記憶體中的資料) 一併進入到快照中；使用該參數必須先在 VM 上安裝 qemu-guest agent

![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-3-1.png]]
<div style="page-break-after:always;"></div>

### 4. 在不同工作站間搬移 VM
> Reference : 
https://blog.gtwang.org/linux/linux-scp-command-tutorial-examples/

將目錄複製到另一個工作站
```bash
scp -r /tmp2/b11902038 b11902038@linux2.csie.ntu.edu.tw:/tmp2/b11902038
```

ssh 到 linux2 然後檢查檔案是否順利搬移，接著開啟 VM
```
virsh start b11902038
```
![[School/Course Homeworks/NASA/assets/nasa hw3/sa1-4-1.png]]
<div style="page-break-after:always;"></div>

## Docker
> Reference : 
https://docs.docker.com/compose/install/linux/#install-the-plugin-manually

### 1. 安裝 Docker
```
sudo apt-get update
sudo apt-get install docker.io
sudo apt-get install docker-compose-plugin
```
![[School/Course Homeworks/NASA/assets/nasa hw3/sa2-1-1.png]]

### 2. 撰寫 Dockerfile

Dockerfile 如檔案所附

用所在目錄的 Dockerfile 建立名為 "sl" 的 image
```
docker build -t sl ./
```

接著執行 container
```
docker run --rm -it sl
```

![[School/Course Homeworks/NASA/assets/nasa hw3/sa2-2-1.png]]
<div style="page-break-after:always;"></div>

### 3. 部署網頁伺服器

```bash
# 從 git 下載檔案
git clone https://github.com/aoaaceai/nasa-hw3
cd nasa-hw3

# 在背景運行
docker-compose up -d
```

查看 container 的狀態
```
docker-compose ps
```
![[School/Course Homeworks/NASA/assets/nasa hw3/sa2-3-1.png]]

查看 logs
```
docker-compose logs
```
![[School/Course Homeworks/NASA/assets/nasa hw3/sa2-3-2.png]]
<div style="page-break-after:always;"></div>

## Kubernetes
### **Kubernetes Architecture**
> Reference :
https://cwhu.medium.com/kubernetes-basic-concept-tutorial-e033e3504ec0
https://medium.com/devops-mojo/kubernetes-architecture-overview-introduction-to-k8s-architecture-and-understanding-k8s-cluster-components-90e11eb34ccd

<br>

![[School/Course Homeworks/NASA/assets/nasa hw3/kubernetes.png|475]]

<br>

<mark style="background: #ADCCFFA6;">**Pod**</mark>
運作的最小單位，一個 Pod 對應到一個服務；一個 Pod 裡面可以有多個 Container，但一般而言只會有一個。
<br>

<mark style="background: #ADCCFFA6;">**Worker Node**</mark>
運作的最小硬體單位，一個 Node 對應到一台機器 (實體機 or 虛擬機)，包含三個元件：
- **kubelet** : 管理員，負責管理 Pods 的狀態並與 Master 溝通
- **kube-proxy** : 有點像 switch？管理 Pods 的 IP，確保和外界的溝通
- **Container Runtime** : 真正負責執行 container 的程式
<br>

<mark style="background: #ADCCFFA6;">**Master Node** </mark>
特化指揮官型的 Node，負責管理所有其他 Worker Node，包含四個元件：
- **API Server** : 管理整個 Kubernetes 所需 API 的接口／負責 Node 之間的溝通
- **etcd** : 存放 Cluster 的備份，故障時可以透過其還原狀態
- **Controller Manager** : 透過 API Server 來確認 Cluster 的狀態，是管理 controllers 的 controller？一個 controller 負責監控不同部位的狀態
- **Scheduler** : Pods 調度員，每當一個新的 Pod 被建立，它會根據每個 Node 的條件去協調出一個 Node 讓該 Pod 運行
<br>

<mark style="background: #ADCCFFA6;">**Cluster**</mark>
多個 Worker Node 和 Master Node 的集合。
<br>

使用 K8s 的好處有很多，其中一點是可以很容易的部署服務，由於每個服務都對應到一個 container ，因此可以很容易的做修改、擴張。
<br>
<br>

## **Build Application**
> Reference :
https://ithelp.ithome.com.tw/articles/10193509
https://docs.google.com/presentation/d/1qHpLB9eRDWm3m8h5UZAmXq3VlGxn7866RwpFRJdJhY0/edit#slide=id.p
ChatGPT

開啟 minikube
```bash
minikube start
```

建立 psql.yaml
```yaml
# psql.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: psql-b11902038-depl
spec:
  replicas: 1
  selector:
    matchLabels:
      app: psql-b11902038
  template:
    metadata:
      labels:
        app: psql-b11902038
    spec:
      containers:
      - name: postgres
        image: postgres:14-alpine
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          value: example
---
apiVersion: v1
kind: Service
metadata:
  name: psql-b11902038-svc
spec:
  selector:
    app: psql-b11902038
  ports:
    - name: postgres
      port: 5432
      targetPort: 5432
  type: ClusterIP
```

部署這個服務
```bash
kubectl apply -f psql.yaml
```

運行成功截圖：
![[School/Course Homeworks/NASA/assets/nasa hw3/sa3-2-1.png]]
![[School/Course Homeworks/NASA/assets/nasa hw3/sa3-2-2.png|425]]
<div style="page-break-after:always;"></div>

### **ConfigMap & Secret**
> Reference :
https://ithelp.ithome.com.tw/articles/10224066

1. ConfigMap 和 Secret 皆為 K8s 的檔案類型，用來將容器的配置資訊從設定檔中分離出來，因此我們可以在容器運行時進行修改，而不需要修改容器本身。也就是說，容器會透過讀取 ConfigMap (或是 Secret) 來獲取特定資訊。
兩者最主要的差別在於 ConfigMap 用來存放一般的資訊，而 Secret 用來存放需要加密的敏感資訊。
<br>

2. 

建立 secret.yaml
```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: psql-secret
type: Opaque
data:
  # 這裡為 b11902038 的 base64 encode
  POSTGRES_USER: YjExOTAyMDM4 
  POSTGRES_PASSWORD: YjExOTAyMDM4
```

部署這個文件
```bash
kubectl apply -f secret.yaml
```

修改 deployment 原本的 yaml 
```bash
kubectl edit deployment/psql-b11902038-depl
```

找到環境變數的位置 ( spec:containers:env: )，並改成以下文字：
```yaml
spec:
  containers:
  - env:
    -name: POSTGRES_USER
	  valueFrom:
        secretKeyRef:
		  key: POSTGRES_USER
		  name: psql-secret
	-name: POSTGRES_PASSWORD
	  valueFrom:
        secretKeyRef:
		  key: POSTGRES_PASSWORD
		  name: psql-secret
```

更改後的截圖，可以發現 user 和 password 被擋起來了
![[School/Course Homeworks/NASA/assets/nasa hw3/sa3-3-1.png]]
<div style="page-break-after:always;"></div>

### **CLI bad, I want GUI**

進入 container 
```bash
sudo docker exec -it minikube bash
```
![[School/Course Homeworks/NASA/assets/nasa hw3/sa3-4-1.png]]


現在是 4/23 的下午 3:37 
今天早上我的 minikube 突然爆掉了，沒有辦法 `minikube start`，
花了幾個小時爬文，卸載又重裝，還是修不好，也沒有辦法 `kubectl` 
我決定放棄，所以後面關於 K8s 的部分都沒有做。

<div style="page-break-after:always;"></div>

我還是把我原本的 yaml 檔放上來 (雖然沒有測試過)：
```yaml
# pgadmin.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pgadmin-b11902038-depl
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pgadmin-b11902038
  template:
    metadata:
      labels:
        app: pgadmin-b11902038
    spec:
      containers:
        - name: pgadmin4
          image: dpage/pgadmin4
          ports:
            - containerPort: 80
          env:
            - name: email
              value: "b11902038@nasa.com"
          volumeMounts:
            - name: pgadmin-storage
              mountPath: /var/lib/pgadmin
      volumes:
        - name: pgadmin-storage
          emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: pgadmin-b11902038-svc
spec:
  selector:
    app: pgadmin-b11902038
  ports:
    - name: http
      port: 80
      targetPort: 80
  type: NodePort
```
