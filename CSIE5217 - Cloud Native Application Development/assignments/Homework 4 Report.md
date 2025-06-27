
b11902038 資工三 鄭博允

## Docker Hub
創建一個 Docker Hub Repo，並且有一個名為 2025cloud 的專案:
專案內有兩個以上的 Container Image:
**docker hub** https://hub.docker.com/r/tanimal19/2025cloud
## README & Dockerfile
Github 專案內有 Dockerfile:
**repo** https://github.com/Tanimal19/CloudNative2025/tree/main/hw4

README 有清楚描述如何透過 docker build 打包你的應用程式:
**issue** https://github.com/Tanimal19/CloudNative2025/issues/6

README 有清楚描述如何透過 docker run 運行你的 Container Image:

## Github Action
Github Action 有辦法去自動執行 Docker Build:
Github Action 有辦法去自動執行 Docker Push 將產生好的 Image 給推到前述的 2025cloud repo:
**success action** https://github.com/Tanimal19/CloudNative2025/actions/runs/14789767439

產生一個 Pull Request，裡面故意寫壞 Dockerfile 讓他失敗，GitHub Action 要有辦法偵測壞掉:
**pull request** https://github.com/Tanimal19/CloudNative2025/pull/7

## 文件詳述
README 以圖文的方式描述目前專案自動化產生 Container Image 的邏輯，以及 Tag 的選擇邏輯:
**readme** https://github.com/Tanimal19/CloudNative2025/tree/main/hw4