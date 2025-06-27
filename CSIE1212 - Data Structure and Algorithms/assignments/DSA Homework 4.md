b11902038 資工一 鄭博允

---

# Problem 1 - Raffle Tickets
## 1.
> Reference : 
https://ithelp.ithome.com.tw/articles/10245793

### (a)
![[School/Course Homeworks/DSA/assets/DSAHW4/IMG_0038.jpg|500]]

hash function :
$$h_1(k) = 8k \mod 7$$

步驟流程：

1. $h_1(72) = 2$，2 號格子還沒滿，將 $72$ 放入 2 號格子 
<br>
2. $h_1(23) = 2$，2 號格子滿了，看下一格
	3 號格子還沒滿，將 $23$ 放入 3 號格子
<br>
3. $h_1(51) = 2$，2 號格子滿了，看下一格
	3 號格子也滿了，看下一格
	4 號格子還沒滿，將 $51$ 放入 4 號格子
<br>
4. $h_1(3) = 3$，3 號格子滿了，看下一格
	4 號格子也滿了，看下一格
	5 號格子還沒滿，將 $3$ 放入 5 號格子
<br>
5. $h_1(18) = 4$，4 號格子滿了，看下一格
	5 號格子也滿了，看下一格
	6 號格子還沒滿，將 $18$ 放入 6 號格子
<br>
 6. $h_1(40) = 5$，5 號格子滿了，看下一格
	6 號格子也滿了，看下一格(回到 0 號格子)
	0 號格子還沒滿，將 $40$ 放入 0 號格子
<br>
 7. $h_1(21) = 0$，0 號格子滿了，看下一格
	1 號格子還沒滿，將 $21$ 放入 1 號格子
<br>
 8. 結束

<div style="page-break-after:always;"></div>

### (b)
![[School/Course Homeworks/DSA/assets/DSAHW4/IMG_0039.jpg|500]]

hash function : 
$$h(k, i) = (h_1(k) + i \times h_2(k)) \mod 7$$
$$h_1(k) = 8k \mod 7$$
$$h_2(k) = 1 + ((2k-2) \mod 6)$$


步驟流程：

1. $h(72, 0) = 2$，2 號格子還沒滿，將 $72$ 放入 2 號格子 
<br>
2. $h(23, 0) = 2$，2 號格子滿了，$i = i + 1$
	$h(23, 1) = 5$ 5 號格子還沒滿，將 $23$ 放入 5 號格子
<br>
 3. $h(51, 0) = 2$，2 號格子滿了，$i = i + 1$
	$h(51, 1) = 0$ 0 號格子還沒滿，將 $51$ 放入 0 號格子
<br>
4. $h(3, 0) = 3$，3 號格子還沒滿，將 $3$ 放入 3 號格子
<br>
5. $h(18, 0) = 4$，4 號格子還沒滿，將 $18$ 放入 4 號格子
<br>
6. $h(40, 0) = 5$，5 號格子滿了，$i = i + 1$
	$h(40, 1) = 6$ 6 號格子還沒滿，將 $40$ 放入 6 號格子
<br>
 7. $h(21, 0) = 0$，0 號格子滿了，$i = i + 1$
	$h(21, 1) = 5$，5 號格子滿了，$i = i + 1$
	$h(21, 2) = 3$，3 號格子滿了，$i = i + 1$
	$h(23, 3) = 1$ 1 號格子還沒滿，將 $21$ 放入 1 號格子
<br>
 8. 結束

<div style="page-break-after:always;"></div>

## 2.
> Reference : All by my self

**<mark style="background: #ADCCFFA6;">演算法說明</mark>**
For any letter, the difference of ASCII value between the uppercase and lowercase is fixed (which is 23). Therefore we can derive the value of uppercase from lowercase, and vice versa.

Let's use $R_i$ to represent a rotation start from $s[i]$, for instance, $s$ = "John" and $R_1$ = "Ohnj". 
For each $R_i$ , $s[i]$ will be the only uppercase of the name, and the order stay unchanged, like the picture below. (Assume we count from tail to head)
![[School/Course Homeworks/DSA/assets/DSAHW4/IMG_0042.jpg|350]]
We can simply calculate the hash value of $R_{i}$ by "shifting" $R_{i-1}$, 
and the hash value of $R_{i}$ is 
$(\text {hash}[R_{i-1}] - s[i-1])\ /\ d\ - 23 + (s[i-1] + 23) \times d^{(L-1)} \mod Q$
<br>

**<mark style="background: #ADCCFFA6;">Pseudo Code</mark>**
```c
// Assume s[i] is the ASCII value of name[i]
// Assume string begin at 1

Name_Hash(s[], L, d, Q){
	
	hash[1:L] = 0
	
	for(i = L to 1)
		hash[0] = hash[0] * d + s[i] (mod Q)
	
	for(i = 2 to L)
		hash[i] = 
		(hash[i-1] - s[i-1])/d - 23 + (s[i-1] + 23) * (d^(L-1)) (mod Q)

	return hash[1:L]
}
```
<br>

**<mark style="background: #ADCCFFA6;">時間複雜度</mark>**
We have two *for* loops, each run $L$ times, so the time complexity of this algorithm is $2L = O(L)$.

<div style="page-break-after:always;"></div>

## 3.
> Reference : All by my self

在 840 人內抽 30 次的可能情況有 $840^{30}$ 種

先假設這 30 次內「沒有任何人被抽到兩次」，代表每次都是抽到不同人，
總共會有 $P^{840}_{30} = \large{840! \over 810!}$ 種情況

排除掉這些情況，代表可能有「任何人被抽到兩次以上」，共會有 $840^{30} - \large{840! \over 810!}$ 種情況，而發生這種情況的機率是
$${840^{30} - \large{840! \over 810!} \over 840^{30}} = 0.4078694$$

也就是說「任何人被抽到**兩次以上**」的機率是 0.4，遠大於觀眾所給的機率
當然，如果只是「任何人被抽到**兩次**」的機率會小於這個數字，但肯定也會大於 $2 \over 840^2$

因此，這些觀眾的申訴無效，駁回 !!!

<div style="page-break-after:always;"></div>

### 4.
> Reference : All by my self

在 (a) 小題，我們需要計算 $s_1$ ~ $s_n$ 以及 $key$ 的 hashing value，而 spurious hits 的情況是 $s_1$ ~ $s_N$ 中出現和 $key$ 一樣的 hashing value樣
我們可以先把問題轉換如下：
```ad-info
title: 
假設 $key$ 的 hashing value 是固定的，每次從 $0$ ~ $P-1$ 之間抽一個 hashing value 給字串 $s_i$，並重複 $N$ 次行動，
則「$s_1$ ~ $s_N$ 中出現和 $key$ 一樣的 hashing value」的機率為何？
```

由於 $key$ 的 hashing value 只有一個，因此每次抽出 hashing value 給字串 $s_i$ 時，剛好抽出和 $key$ 一樣的 hashing value 的機率是 $\large 1 \over P$

因為我們會執行 $N$ 次行動，所以「 $s_1$ ~ $s_N$ 中出現和 $key$ 一樣的 hashing value 的機率(spurious hits)是 $\large N \over P$

因此在 $N = 10^5, P = 10^9+7$ 的情況下，處理問題 (a) 時出現 spurious hits 的機率為 
$${10^5 \over 10^9 + 7} \approx 0.0001$$

---

在 (b) 小題，我們需要計算 $s_1$ ~ $s_n$ 的 hashing value，而 spurious hits 就是有兩個字串的 hashing value 是一樣的，也就是同個 hashing value 出現了兩次(以上)
同樣地，我們可以把問題轉換如下：
```ad-info
title:
每次從 $0$ ~ $P-1$ 之間抽一個 hashing value 給字串 $s_i$，並重複 $N$ 次行動
則「有任何 hashing value 出現**兩次以上** (spurious hits)」的機率為何？
```

在 N 個字串中，每個字串的 hashing value 都不一樣的機率為 $\large P! \over \large p^N (P-N)!$

與之相反的情形，也就是「有任何 hashing value 出現**兩次以上**(spurious hits)」的機率則是 $1 - {\large P! \over \large p^N (P-N)!}$

因此在 $N = 10^5, P = 10^9+7$ 的情況下，處理問題 (b) 時出現 spurious hits 的機率為 $$1 - { (10^9+7)! \over (10^9+7)^{10^5} (10^9+7-10^5)!} \approx 0.99326$$

---

因此我們可以發現在相同設定下，問題 (b) 的 spurious hits 發生機率比問題 (a) 高了許多 (幾乎一定會發生)，主要原因在於兩者對於 spurious hits 的發生定義不同。

<div style="page-break-after:always;"></div>

# Problem 2 - README
## 1.
> Reference : All by my self

For any node and its two childs, the maximum ratio of red nodes to black nodes is 2:1 (like picture below)
![[School/Course Homeworks/DSA/assets/DSAHW4/2-1.png|110]]
If we want to reach the maximum ratio, that is, every 3 nodes in the tree formed such situation. Thus, the maximum ratio is 2:1, but not every case can reach the maximum ratio.

RB tree with 15 nodes that reach the maximum ratio of red nodes to black nodes :
![[School/Course Homeworks/DSA/assets/DSAHW4/2.png]]

<div style="page-break-after:always;"></div>

## 2.
> Reference : All by my self

※ 假設簡單的問題放在左邊

**Explanation**

因為 RB tree 算是一種 binary search tree，所以對其做 inorder traversal 實際上就是對所有節點由小到大遍歷，也就是說當我們做 inorder traversal 時，每個節點的順位就代表著它是第幾小的節點。
下圖的數字代表該節點在 inorder traversal 中的順位 :
![[School/Course Homeworks/DSA/assets/DSAHW4/3-1.png]]

我們可以發現這個數字其實就相當於其左側的所有節點數量 + 1，因此我們在每個節點加入一個新的變數 `leftsize`，該變數代表著該節點左子樹的節點數量，我們就可以用這個數字來推算其順位。
下圖的數字代表該節點的 `leftsize` :
![[School/Course Homeworks/DSA/assets/DSAHW4/3-2.png]]
<div style="page-break-after:always;"></div>

**Update the tree**

要計算每個節點的 `leftsize`，我們需要知道該節點左子樹的節點數量，也就是說我們需要先跑過左子樹並把經過的節點數量記下來，再到節點更新 `leftsize`，而我們可以透過 inorder traversal 達成這個目的。

具體流程如下 :
我們對整棵樹做 inorder traversal，但用一個變數 `count` (初始為 0) 來記錄經過的節點數並回傳這個數值，然後經過節點本身時用 `count` 更新 `leftsize`。
```c
Inorder(node){
	count = 0
	if (node != NIL) {
		count += Inorder(node->left)
		node->leftsize = count
		count += 1                          // 也要算進自己
		count += Inorder(node->right)
	}
	return count
}
```

因為是 traversal，時間複雜度為 $O(n)$
<br>

**Find the k-th smallest**

在尋找的過程中，我們使用變數 `order` (初始為 0) 來代表目前所在節點的順位
每當碰到一個節點，我們會先將 `order` 加上該節點的 `leftsize + 1`，再開始比較，會有三種情況 :
1. 如果 `order` 等於 $k$，代表這就是我們要找的節點，直接結束
<br>
2. 如果要找的 $k$ 小於 `order`，代表要找的節點在左子樹，我們要往左邊的節點走，這時候我們要將 `order` 減掉 `leftsize + 1` (因為我們要去左子樹，`leftsize` 會重疊)
<br>
3. 如果要找的 $k$ 大於 `order`，代表要找的節點在右子樹，我們要往右邊的節點走，而這裡不用對 `order` 做更動 (因為我們要去右子樹，`leftsize` 要累加)

由於我們最多只會跑完整棵樹的高度，因此時間複雜度為 $O(\log n)$

<div style="page-break-after:always;"></div>

## 3.
> Reference : All by my self

**Modified Insertion**

基本上和原本的 insertion 是差不多的，只不過我們會在過程中調整 `leftsize` 以確保維持特性，調整如下 :

一開始，我們先尋找要插入的位置，在過程中的每個節點
- 如果「往左找」，代表新的節點會插入在左子樹，而 `leftsize` 加上 1
- 如果「往右找」，則我們不用更動 `leftsize`
<br>

接著進入 insertion fix 的部分
- 如果發生 Left-Rotate($T$, $x$)，將 $y$ 的 `leftsize` 加上 `x->leftsize + 1` (因為多了 $x$ 和它的左子樹)
- 如果發生 Right-Rotate($T$, $y$)，將 $y$ 的 `leftsize` 減去 `x->leftsize + 1` (因為少了 $x$ 和它的左子樹)

![[School/Course Homeworks/DSA/assets/DSAHW4/3-3.png]]
<div style="page-break-after:always;"></div>

**Modified Deletion**

基本上和原本的 deletion 是差不多的，只不過我們會在過程中調整 `leftsize` 以確保維持特性，調整如下 :

一開始，我們先尋找要刪除的節點，在過程中的每個節點
- 如果「往左找」，代表要刪除的節點在左子樹，而 `leftsize` 減去 1
- 如果「往右找」，則我們不用更動 `leftsize`
<br>

接著進入 deletion fix 的部分，這裡的更動和 insertion fix 一模一樣
<br>

由於只是多了一些常數的計算，所以 insertion 和 deletion 的時間複雜度都和原本一樣是 $O(\log n)$
<div style="page-break-after:always;"></div>

## 4.
> Reference : All by my self

**Explanation**

要讓 Souffle 可以處理最多的信，代表我們要從 $z_i$ 最小的信開始處理，直到 $E$ 小於 0 為止

假設我們將信件依照 $z_i$ 的大小排列，從 $z_1$ 到 $z_n$
那麼我們的目的就是要找到最大的 $k$ 讓 $\sum^k_{i=1} z_i \lt E$ 
也就是說處理完第 1 ~ k 封信後，再處理第 $k+1$ 封信後，$E$ 就會小於等於 0
此時 Souffle 可以處理的信件量即為 $k+1$

先假設每個節點裡面都有一個變數 `z_i` 來代表該節點的 $z_i$ 
和第二題一樣，我們為 RB tree 新增了一個變數 `leftsize`，用來代表該節點左子樹的節點數量，也就是信件數量
另外我們還要新增一個變數 `lefttotal` 來代表該節點左子樹所有節點的 `z_i` 總和

以下圖為例，紅色數字為該節點的 `z_i`、圓內的數字為該節點的 `lefttotal` (`leftsize` 和第二題一樣)
![[School/Course Homeworks/DSA/assets/DSAHW4/4-1.png]]
<br>

**Update the tree**

計算 `leftsize` 的方式與第二題相同，
計算 `lefttotal` 的方式也大同小異，只是我們這裡再另外使用 `count2` (初始為 0) 來計算 `z_i` 的總和

因為是 traversal，時間複雜度為 $O(n)$
<div style="page-break-after:always;"></div>

**Determine the maximum number of emails**
與第二題不同，現在我們要計算的是該節點之前的 `z_i` 總和

使用變數 `size` (初始為 0) 來代表目前節點的信件總數
使用變數 `total` (初始為 0) 來代表目前節點的 `z_i` 總和
使用變數 `max` (初始為 0) 來記錄目前能看的最多信件數

每當碰到一個節點 :
將 `size` 加上該節點的 `leftsize + 1`
將 `total` 加上該節點的 `lefttotal + z_i`

再開始比較，會有兩種情形:
1. `total` $\ge E$ :
代表如果由小到大讀了 `size` 封信，$E$ 會小於等於零，也就是說 Souffle 會扛不住，所以我們要往左邊找，讓 Souffle 不要讀那麼多信
我們要將 `total` 減去 `lefttotal + z_i`、將 `size` 減去 `leftsize + 1`
而我們不用對 `max` 做更新，因為 Souffle 不會讀那麼多的信
<br>

2. `total` $\lt E$ :
代表 Souffle 可以讀完 `index` 封信而且 $E$ 不會小於等於零，但我們不知道 Souffle 還能不能讀更多的信，所以我們要往右邊找，尋找 Souffle 的極限
我們不用對 `total` 和 `size` 做更新，但是要將 `max` 更新成 `index` 因為目前的 `index` 就是 Souffle 的極限

就這樣一路直到碰到 NIL，接著我們回傳 `max+1` 就是 Souffle 可以讀的最多信件數了

由於我們最多只會跑完整棵樹的高度，因此時間複雜度為 $O(\log n)$
<div style="page-break-after:always;"></div>

**Modified Insertion 2**

基本上和第三題的 insertion 是差不多的，只不過除了調整 `leftsize` 還要調整 `lefttotal`，調整如下 :

假設欲插入的節點為 `k`
一開始，我們先尋找要插入的位置，在過程中的每個節點
- 如果「往左找」，代表新的節點會插入在左子樹，而 `leftsize` 加上 1、`lefttotal` 加上 `k->z_i`
- 如果「往右找」，則我們不用更動 `leftsize`、`lefttotal`
<br>

接著進入 insertion fix 的部分
- 如果發生 Left-Rotate($T$, $x$)，將 $y$ 的 `leftsize` 加上 `x->leftsize + 1`、將 $y$ 的 `lefttotal` 加上 `x->lefttotal + x->z_i` (因為多了 $x$ 和它的左子樹)
- 如果發生 Right-Rotate($T$, $y$)，將 $y$ 的 `leftsize` 減去 `x->leftsize + 1`、將 $y$ 的 `lefttotal` 減去 `x->lefttotal + x->z_i` (因為少了 $x$ 和它的左子樹)
<br>

**Modified Deletion 2**

基本上和第三題的 deletion 是差不多的，只不過除了調整 `leftsize` 還要調整 `lefttotal`，調整如下 :

假設欲刪除的節點為 `k`
一開始，我們先尋找要刪除的節點，在過程中的每個節點
- 如果「往左找」，代表要刪除的節點在左子樹，而 `leftsize` 減去 1、`lefttotal` 減去 `k->z_i`
- 如果「往右找」，則我們不用更動 `leftsize`、`lefttotal`
<br>

接著進入 deletion fix 的部分，這裡的更動和 insertion fix 一模一樣
<br>

由於只是多了一些常數的計算，所以 insertion 和 deletion 的時間複雜度都和原本一樣是 $O(\log n)$

