b11902038 資工三 鄭博允

## 專案操作
建立 GCP project "Cloud Native"

<div style="page-break-after:always;"></div>

## Netwroking
### 建立一個 Auto Mode 的 VPC
![|500](School/Course%20Homeworks/Cloud%20Native/assets/1.png)

### 建立一個 VPC firewall rule
![|500](School/Course%20Homeworks/Cloud%20Native/assets/2.png)

<div style="page-break-after:always;"></div>

## Compute Engine
### 建立 VM instance
![|500](School/Course%20Homeworks/Cloud%20Native/assets/3-1.png)
![|500](School/Course%20Homeworks/Cloud%20Native/assets/3-2.png)

### 透過 GCE 頁面上的按鈕遠端 SSH 到 VM hw-vm 安裝 Nginx，並確認
Nginx 運行狀態
![|500](School/Course%20Homeworks/Cloud%20Native/assets/4-1.png)
![|500](School/Course%20Homeworks/Cloud%20Native/assets/4-2.png)
![|500](School/Course%20Homeworks/Cloud%20Native/assets/4-3.png)

### 編輯 VM hw-vm，設定網路標籤為 hw-allow-http，並使用瀏覽器訪問
VM hw-vm External IP
![|500](School/Course%20Homeworks/Cloud%20Native/assets/5.png)

### 切換到 VM hw-vm 監控頁面，確認 VM 運作情況
![|500](School/Course%20Homeworks/Cloud%20Native/assets/6.png)

### 將 VM hw-vm 關閉
![|500](School/Course%20Homeworks/Cloud%20Native/assets/7.png)

<div style="page-break-after:always;"></div>

## Cloud Storage
### 建立一個任意名稱的 bucket，將任一張圖片上傳到 bucket
![|500](School/Course%20Homeworks/Cloud%20Native/assets/8.png)

### 將 bucket 設定為公開，使用瀏覽器訪問圖片
![|500](School/Course%20Homeworks/Cloud%20Native/assets/9.png)

### 在 bucket 建立資料夾 hw_storage
![|500](School/Course%20Homeworks/Cloud%20Native/assets/10.png)

<div style="page-break-after:always;"></div>

## 清理資源
### 刪除 VM
![|500](School/Course%20Homeworks/Cloud%20Native/assets/11.png)

### 刪除 Bucket
![|500](School/Course%20Homeworks/Cloud%20Native/assets/12.png)
