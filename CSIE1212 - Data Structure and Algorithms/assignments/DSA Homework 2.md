B11902038 資工一 鄭博允

---

※ 作業中所有提到的陣列 $A$ 都是以 1 開始 (除非特別註明)

# Problem 1
## 1.
> Reference : All by myself

一棵樹只有一個根節點，其餘皆為子節點，
假設整棵樹有 $C = n_0 + n_1 + n_2 + ... + n_k = \sum_{i=0}^{k}n_i$ 個節點，則子節點總共有 $C-1$ 個。


所有 degree $k$ 的節點下的子節點數共為 $C = n_k \times k$。
假設每個 degree $k$ 的節點皆為子節點。(這裡不需要計算 degree $k$ 的節點本身，因為在計算其他節點的子節點時會被涵蓋進去)
我們可以得到所有子節點的數量為 $\sum_{i=0}^{k}{n_i \times i}$，並得到下式：
$$
\begin{aligned}
\sum_{i=0}^{k}{n_i \times i} &= C-1 \\\\
\sum_{i=0}^{k}{n_i \times i} &= \sum_{i=0}^{k}n_i - 1 \\\\
\sum_{i=0}^{k}{n_i \times i} - \sum_{i=0}^{k}n_i + 1 &= 0 \\\\
-n_0 + \sum_{i=0}^{k}{n_i(i-1)} + 1 &= 0 \\\\
1 + \sum_{i=0}^{k}{n_i(i-1)} &= n_0
\end{aligned}
$$

$Q.E.D.$
<div style="page-break-after:always;"></div>

## 2.
> Reference : All by myself

我們可以只用 inorder 和 preorder 就重建出一棵「唯一」的二元樹，由於任何子樹的 preorder 一定是是從根節點開始，而 inorder 可以用根節點來判斷左右子樹的內容物。

inorder traversal : 4, 26, 19, 22, 7, 15, 34, 11, 13, 8
preorder traversal : 22, 4, 19, 26, 11, 15, 7, 34, 13, 8

1. 依序從 preorder 提取數字 $x$ 
e.g. **22**

2. 在 inorder 裡面用數字 $x$ 將數列分成兩邊
e.g. <mark style="background: #ABF7F7A6;">4, 26, 19</mark>, **22**, <mark style="background: #FFB8EBA6;">7, 15, 34, 11, 13, 8</mark>

3. 將 $x$ 紀錄為根節點

4. 以左側的數字為左子樹的內容，並回到步驟一，如果沒有左側沒有數字，則前往下一步

5. 以右側的數字為右子樹的內容，並回到步驟一，如果沒有右側沒有數字，則前往下一步

6. 完成

最後完成圖：
![[School/Course Homeworks/DSA/assets/DSAHW2/1-2.png|400]]
<div style="page-break-after:always;"></div>

## 3.
> Reference : All by myself

我們沒有辦法只用 preorder 和 postorder 重建出一棵「唯一」的二元樹，由於我們無法判斷根位於子節點的右邊或左邊。

preorder traversal: 22, 4, 19, 26, 11, 15, 7, 34, 13, 8
postorder traversal: 26, 19, 4, 7, 34, 15, 8, 13, 11, 22

例如我們可以用上面的資訊建出和上一題不同的樹：
![[School/Course Homeworks/DSA/assets/DSAHW2/1-3.png|400]]

<br>

## 4.
> Reference : All by myself

![[School/Course Homeworks/DSA/assets/DSAHW2/1-4.png|400]]
<div style="page-break-after:always;"></div>

## 5.
> Reference : All by myself

要建一棵最矮的樹，也就是說，兩邊的節點越平衡(數量一樣)越好，因此我們可以假設根節點會位在陣列的中間，所有子樹都可以用相同的觀點來進行。

建立一個遞迴演算法，每一個遞迴會將陣列中間的數值 ($A[{n \over 2}]$) 設為根節點，然後依序將根節點左側 ($A[1:{n \over 2}-1]$)、右側 ($A[{n \over 2}+1 : n]$)，視為左右子樹的內容並呼叫下一個遞迴，最後回傳根節點。

```pseudo
\begin{algorithm}
\caption{}
\begin{algorithmic}
\PROCEDURE{Build-BST}{$A[1:n], n$}
	\STATE \\
	\IF{$n \le 1$}
		\STATE root = $A[1]$
		\RETURN root
	\ENDIF
	\STATE \\
	\STATE root = $A[{n \over 2}]$
	\STATE root.left = \CALL{Build-BST}{$A[1:{n \over 2}-1]$, ${n \over 2}-1$}
	\STATE root.right = \CALL{Build-BST}{$A[{n \over 2}+1 : n]$, ${n \over 2}-1$}
	\RETURN root
	\STATE \\
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
<br>

**時間複雜度分析：**
每個遞迴都會將一個數字設為根節點，花費 $O(1)$。
最終，陣列的每一個元素都會被設為某一子樹的根節點，而途中被設定過的元素不會再被訪問，因此遞迴總共會被執行 $n$ 次。
所以時間複雜度為 $O(n)$。
<div style="page-break-after:always;"></div>

## 6.
> Reference : All by myself

※ All inputs is not counted as extra-space.
※ Space of variables is ingnored.

<mark style="background: #ADCCFFA6;">**Pigeon-Store**</mark>

**Time complexity :**
**for** loop in Line 4 will run $P$ times, and every lap is $O(1)$, therefore the entire loop is $O(P)$.
**for** loop in Line 2 will run $M$ times, and every lap is $O(P) + O(P) = O(P)$, therefore the entire loop is $M \times O(P) = O(M \times P)$.
Remaining lines cost $O(1)$ and can be ingnored.
In the end, the total time complexity is $O(M \times P)$.
<br>

**Extra-space complexity :**
We allocate an $M$ by $P$ array $bugs$, therefore the extra-space complexity is $O(M \times P)$.
<br>

<mark style="background: #ADCCFFA6;">**Pigeon-Search**</mark>

**Time complexity :**
Worst case :  every $s$ and $bugs[j]$ differ only in the last letter, which means it will never go through Line 10 ~ 12.
**for** loop in Line 6 run $P$ times, and it's $O(P)$.
**for** loop in Line 4 run $M$ times, and every lap is $O(P)$,  therefore the entire loop is $O(M \times P)$.
**for** loop in Line 2 run $N$ times, and every lap is $O(M \times P)$, therefore the entire loop is $O(N \times M \times P)$.
Line 1 is $O(N)$ to initialize the array, but $O(N \times M \times P) + O(N) = O(N \times M \times P)$.
Remaining lines cost $O(1)$ and can be ingnored.
In the end, the total time complexity is $O(N \times M \times P)$.
<br>

**Extra-space complexity :**
We have an $M$ by $P$ array $bugs$ from **Pigeon-Store**, which is $O(M \times P)$.
We allocate an array $eat$ that is $O(N)$.
Therefore the extra-space complexity is $O(M \times P\ +\ N)$.
<div style="page-break-after:always;"></div>

## 7.
> Reference : All by myself

<mark style="background: #ADCCFFA6;">**Spotteddove-Store**</mark>

**Time complexity :**
**for** loop in Line 6 will run $P$ times, and every lap is $O(1)$, therefore the entire loop is $O(P)$.
**for** loop in Line 3 will run $M$ times, and every lap is  $O(P) + O(P) = O(P)$, therefore the entire loop is $O(M \times P)$.
Remaining lines cost $O(1)$ and can be ingnored.
In the end, the total time complexity is $O(M \times P)$.
<br>

**Extra-space complexity :**
Assume space of a $Node$ is $26$, each child is $1$.
Since each section of a bug (= a letter) has 26 possibilities, bug of length $P$ has at most $26^P$ types.

If $M \ge 26^P$ and there has $26^P$ different bugs, than some bugs will be repeated, and we can use a perfect 26-ary tree without root to store all of them, therefore the total space is $\sum^{P}_{i=1}{26^i} = O(26^P)$

If $M \lt 26^P$, if every bug is unique, than there will be $M$ different "paths" of length $P$ from $bugsRoot$, and the total space will be less than $M \times P$ since some part of the "path" might overlap, therefore the total space is $O(M \times P)$.
<br>

<mark style="background: #ADCCFFA6;">**Spotteddove-Search**</mark>

**Time complexity :**
Worst case :  every $s$ is matched, which means it will go through Line 14 every lap.
**for** loop in Line 6 run $P$ times, and it's $O(P)$.
**for** loop in Line 2 run $N$ times, and every lap is $O(P)$,  therefore the entire loop is $O(N \times P)$.
Remaining lines cost $O(1)$ and can be ingnored.
In the end, the total time complexity is $O(N \times P)$.
<br>

**Extra-space complexity :**
We have tree $bugsRoot$ from **Spotteddove-Store**, whose space complexity is :
$M \ge 26^P$ than $O(26^P)$ ; $M \lt 26^P$ than $O(M \times P)$
We allocate an array $eat$ that is $O(N)$
Therefore the extra-space complexity is :
$M \ge 26^P$ than $O(26^P + N)$ ; $M \lt 26^P$ than $O(M \times P + N)$
<div style="page-break-after:always;"></div>

## 8.
> Reference : All by myself

Pigeon 鴿子 ／ Spotted Dove 斑鳩

我會選擇鴿子演算法而非斑鳩演算法。~~因為我腦霧了~~
數據會說幹話，來看一下表格：

|        | time | extra-space |
| ------ | ---- | ----------- |
| Pigeon store |  $O(M \times P)$    |   $O(M \times P)$          |
|  Pigeon search      |  $O(N \times M \times P)$    | $O(M \times P\ +\ N)$            |
|  Spotteddove-Store      |  $O(M \times P)$    |   $O(26^P)$ ; $O(M \times P)$          |
|  Spotteddove-Search      |  $O(N \times P)$    |   $O(26^P + N)$ ; $O(M \times P + N)$          |

從上表可以得知，
時間方面，兩者在 store 是伯仲之間，鴿子演算法在 search 則是爆輸一波；
空間方面，斑鳩演算法在 $M$ 很大時比較有利，在其他情況則是和鴿子演算法平起平坐。

然而這只是理論方面，實際上真的會有鳥具有那麼多喜歡的蟲嗎($M$ 很大)？全世界的蟲真的有那麼多嗎？~~真的有閒人會幫鳥吃蟲寫演算法嗎？~~

從實現演算法的角度來看，
斑鳩演算法不太好實現，用了艱澀難懂的資料結構，不知道在~~供三~~
相較之下鴿子演算法是老嫗能解，用簡單的陣列就能實現。

總而言之，
**鴿子一定是大拇指的啦**
<div style="page-break-after:always;"></div>

# Promblem 2
## 1.
> Reference : All by myself

```c
[7, 3, 5, 0, 2, 8, 6, 1]
```
分割成 size-1 array
對每個 size-1 array 排序
```c
[7] [3] [5] [0] [2] [8] [6] [1]
```
將相鄰的 size-1 array 合併成 size-2 並排序
```c
[3, 7] [0, 5] [2, 8] [1, 6]
```
將相鄰的 size-2 array 合併成 size-4 並排序
```c
[0, 3, 5, 7] [1, 2, 6, 8]
```
將相鄰的 size-4 array 合併成 size-8 並排序
```c
[0, 1, 2, 3, 5, 6, 7, 8]
```
完成  
<br>

## 2.
> Reference : All by myself

從 $i = 1$ 開始，依序比對 $r[i]$ 是否大於 $r[j]$ ，$i \lt j \le 8$
得到 $r[1] \gt r[2], r[4], r[5], r[6], r[8]$，共發生 $5$ 次 reversion

接著從 $i = 2$，再依序比對 $r[i]$ 是否大於 $r[j]$ ，$i \lt j \le 8$
得到 $r[2] \gt r[4], r[5], r[8]$，共發生 $3$ 次 reversion

以此類推直到 $i = 8$，將每一輪的次數相加
最後得到 reversion 發生 $5 + 3 + 5 + 2 + 1 + 1 = 17$ 次
<div style="page-break-after:always;"></div>

## 3.
> Reference : All by myself

※有一些 pseudo code 看起來超過 30 行，那是因為加了空白行以方便閱讀，所以實際上沒有到 30 行

### <mark style="background: #ADCCFFA6;">**Explain**</mark>
This algorithm is a modified merge sort, for each recursion we merge two array $L[]$ and $R[]$. For every element in $R[]$, it's index (season) is bigger than $L[]$. Therefore, for any element $L[l]$, if it's value (rank) is bigger than any element $R[r]$, in other words, $L[l]$ is insert after $R[r]$ when merging, then we can infer that reversion is occur.

In conclusion, when merging $L[]$ and $R[]$, if some $R[r]$ is insert, that means every element in $L[]$ that haven't been insert will go through a reversion, so the number of reversions should be add on the current length of $L[]$ (since we did not actually remove element from $L[]$, we use $len$ to represent it).
<br>

```pseudo
\begin{algorithm}
\caption{Reversion-Count}
\begin{algorithmic}
\PROCEDURE{Main}{$A, n$}
\STATE Ans = \CALL{R-Merge-Sort}{$A, 1, n$}
\ENDPROCEDURE
\STATE \\
\PROCEDURE{R-Merge-Sort}{$A$, $head$, $tail$}
\STATE $reversion = 0$
\IF{$tail \le head$}
	\RETURN $reversion$
\ENDIF
\STATE \\
\STATE $mid = (head+tail)/2$
\STATE $reversion = reversion +$  \CALL{R-Merge-Sort}{$A$, $head$, $mid$}
\STATE $reversion = reversion +$  \CALL{R-Merge-Sort}{$A$, $mid+1$, $tail$}
\STATE \\
\STATE copy $A[head:mid]$ to $L[]$ 
\STATE copy $A[mid+1:tail]$ to $R[]$ 
\STATE $l=0, r=0$
\STATE $len$ = length of $L[]$
\STATE \\
\FOR{$i = head$ to $tail$}
	\STATE $A[i] =$ \call{smaller}{$L[l], R[r]$}
	\IF{$A[i]$ equal $L[l]$}
		\STATE $l = l + 1$
		\STATE $len = len -1$
	\ELIF{$A[i]$ equal $R[r]$}
		\STATE $r = r + 1$
		\STATE $reversion = reversion + len$
	\ENDIF
\ENDFOR
\STATE \\
\RETURN $reversion$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
<br>

### <mark style="background: #ADCCFFA6;">**Time Complexity**</mark>
Althought we add some new lines (18~24) to calculate the number of reversions, they only cost constant time, so the total time complexity is same as merge sort, which is $O(n\log n)$.
<br>

### <mark style="background: #ADCCFFA6;">**Extra-Space Complexity**</mark>
Since we only allocate at most $O(n)$, which is $L[n/2]$ and $R[n/2]$ at the first recursion, the extra-space complexity is $O(n)$.
<div style="page-break-after:always;"></div>

## 4.
> Reference : All by myself

### <mark style="background: #ADCCFFA6;">**Explain**</mark>
At first, we apply merge-sort and sort $A[]$ by their rank, so every $A[i].rank \ge A[i-1].rank$.

The algorithm *P-Merge-Sort* is nearly same as previous algorithm (*R-Merge-Sort*), but this time we compare $L[r]$ and $R[r]$ by their K/D.

Also, we reverse the statement : if some $L[l]$ is insert, that means every element in $R[]$ has bigger index (rank) and K/D than it, namely they can pair each other, so the number of pairs should be add on the current length of $R[]$.
<br>

```pseudo
\begin{algorithm}
\caption{Pair-Count}
\begin{algorithmic}
\PROCEDURE{Main}{$A, n$}
\STATE Apply Merge-Sort and sort $A$ by their rank //$O(n\log n)$
\STATE Ans = \CALL{P-Merge-Sort}{$A, 1, n$}
\ENDPROCEDURE
\STATE \\
\PROCEDURE{P-Merge-Sort}{$A$, $head$, $tail$}
\STATE $pair = 0$
\IF{$tail \le head$}
	\RETURN $pair$
\ENDIF
\STATE \\
\STATE $mid = (head+tail)/2$
\STATE $pair = pair +$  \CALL{P-Merge-Sort}{$A$, $head$, $mid$}
\STATE $pair = pair +$  \CALL{P-Merge-Sort}{$A$, $mid+1$, $tail$}
\STATE \\
\STATE copy $A[head:mid]$ to $L[]$ 
\STATE copy $A[mid+1:tail]$ to $R[ ]$ 
\STATE $l=0, r=0$
\STATE $len$ = length of $R[]$
\STATE \\
\FOR{$i = head$ to $tail$}
	\STATE $A[i] =$ \call{smaller}{$L[l].kda,\ R[r].kda$}
	\IF{$A[i]$ equal $L[l]$}
		\STATE $l = l + 1$
		\STATE $pair = pair + len$
	\ELIF{$A[i]$ equal $R[r]$}
		\STATE $r = r + 1$
		\STATE $len = len -1$
	\ENDIF
\ENDFOR
\STATE \\
\RETURN $pair$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
<br>

### <mark style="background: #ADCCFFA6;">**Time Complexity**</mark>
We do merge-sort at first, and we do *P-Merge-Sort* after, so the total time complexity is double of merge sort, which is $2 \cdot n\log n = O(n\log n)$.
<br>

### <mark style="background: #ADCCFFA6;">**Extra-Space Complexity**</mark>
In the first merge-sort, we use $O(n)$, in *P-Merge-Sort* we use $O(n)$, too.
Therfore the extra-space complexity is $O(n)$.
<div style="page-break-after:always;"></div>

## 5.
> Reference : https://www.geeksforgeeks.org/k-th-element-two-sorted-arrays/

### <mark style="background: #ADCCFFA6;">**Explain**</mark>
We assume array $A_1[]$ and $A_2[]$ begin at 1.

Assume the k-th element we want has value $key$.
If we know some number $A[i]$ is smaller than $key$, then we can assert that every element before $A[i]$ is smaller than $key$ and can be ingnored. 

For every recursion, we compare k/2-th element of two arrays, if $A_1[k/2] < A_2[k/2]$ then we discard elements before k/2-th element of the smaller one, which is $A_1[h_1:k/2]$.
Now we can turn it to a subtask : find (k-k/2)-th element in $A_1[k/2:]$ and $A_2[h_2:]$

To help understanding the algorithm, please refer to this example :
```c
A1[] = {1, 2, 4, 7, 10}
A2[] = {3, 5, 6, 8, 9}
k = 5

A1[2] < A2[2]
discard A1[1:2]

A1[] = {4, 7, 10}
A2[] = {3, 5, 6, 8, 9}
k = 3

A1[1] > A2[1]
discard A2[1:1]

A1[] = {4, 7, 10}
A2[] = {5, 6, 8, 9}
k = 2

A1[1] < A2[1]
discard A1[1:1]

A1[] = {7, 10}
A2[] = {5, 6, 8, 9}
k = 1

return 5
```

Since we can't really discard the elements in an array, we use $h_1$, $h_2$ to represnet the head of $A_1$ and $A_2$.
<br>

```pseudo
\begin{algorithm}
\caption{Find k-th element in two sorted array}
\begin{algorithmic}
\PROCEDURE{Main}{$A_1$, $A_2$, $n$, $k$}
\STATE Ans = \CALL{Find-K}{$A_1$, $A_2$, $1$, $1$, $k$, $n$}
\ENDPROCEDURE
\STATE \\
\PROCEDURE{Find-K}{$A_1$, $A_2$, $h_1$, $h_2$, $k$, $n$}
\IF{$h_1 \gt n$}
	\RETURN $A_2[h_2+k-1]$
\ELIF{$h_2 \gt n$}
	\RETURN $A_1[h_1+k-1]$
\ENDIF
\STATE \\
\IF{$k$ equal $1$}
	\RETURN \CALL{smaller}{$A_1[h_1],\ A_2[h_2]$}
\ENDIF
\STATE \\
\STATE $i =$ \CALL{floor}{$k/2$}
\IF{$A_1[h_1+i-1] \le A_2[h_2+i-1]$}
	\RETURN \CALL{Find-K}{$A_1$, $A_2$, $h_1+i$, $h_2$, $k-i$, $n$}
\ELSE
	\RETURN \CALL{Find-K}{$A_1$, $A_2$, $h_1$, $h_2+i$, $k-i$, $n$}
\ENDIF
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
<br>

### <mark style="background: #ADCCFFA6;">**Time Complexity**</mark>
Since every recursion we discard (ingnored) k/2 elements, in the end, we will go through $log k$ times recursions, and thus the total time complexity is $O(log k)$.
<br>

### <mark style="background: #ADCCFFA6;">**Extra-Space Complexity**</mark>
We only allcoate few variables, therfore the extra-space complexity is $O(1)$.
<div style="page-break-after:always;"></div>

## 6.
> Reference : All by myself

For line 3~8 in *Partition*, it make $macth[latest]$ be the largest element in the array. 
Therefore, line 10~12 will be go through from $j = oldest$ to $latest-1$, and since each lap $i = j$, it will always swap with itself.
At the end of *Partition*, the array will have maximum at $macth[latest]$, and the remained elements are still the same. Then *Partition* will return $i+1 = latest$.

In *NeonSort*, by above we know that *pi = latest*, so each lap it will do 
*NeonSort*($match, oldest, latest-1$),
which exactly put the maximum into $macth[latest-1]$ 
and
*NeonSort*($match, latest+1, latest$),
which literally does nothing.

In summary, for every lap in *NeonSort*, it will call *Partition* and put maximum into tail of non-sorted part. It's obvious that these code actually act like insertion sort and has time complextiy of $O(n^2)$ rather than $O(n log n)$.
<div style="page-break-after:always;"></div>

# Problem 3

## 1.
> Reference : All by myself

![[School/Course Homeworks/DSA/assets/DSAHW2/1.png]]
<br>

## 2.
> Reference : All by myself

![[School/Course Homeworks/DSA/assets/DSAHW2/2.png]]
<div style="page-break-after:always;"></div>

## 3.
> Reference : All by myself

對於 min heap 來說，每一個節點都會小於它的所有子節點，因此我們可以知道最小的數字一定會在樹根

先將最小的放在樹根，剩下 14 個數字，將它均分成兩組，分別代表左右子樹的內容，總共會有 $C^{14}_{7} = 3432$ 種組合

接著看到其中一棵子樹，總共有 7 個數字，將最小的放在樹根，剩下 6 個數字，將它均分成兩組，會有 $C^{6}_{3} = 20$ 種組合；由於左右子樹都要做分配，最後共有  ${(C^{6}_{3})}^2 = 400$ 種組合

繼續遞迴下去，子樹總共有 3 個數字，將最小的放在樹根，剩下 2 個數字，將它均分成兩組，總共會有 $C^{2}_{1} = 2$ 種組合；由於有 4 個 3 節點的子樹要做分配，最後會有 ${(C^{2}_{1})}^4 = 16$ 種組合

剩下 1 個數字，將最小的放在樹根，沒有子節點要做分配

最後答案會得到
$$
3432 \times 400 \times 16 = 21,964,800
$$
<div style="page-break-after:always;"></div>

## 4.
> Reference : All by myself

用陣列來表示 min heap，假設陣列從 1 開始，那麼對於每一個元素 $A[i]$，它會有左子節點 $A[2i+1]$ 和右子節點 $A[2i+2]$。

下圖數字為該節點在陣列中的 index :
![[School/Course Homeworks/DSA/assets/DSAHW2/tree.png|475]]

從上圖我們可以發現，對於 $A[1]$ 而言，它沒有辦法形成任何 inverse pair，因為 min heap 的性質使得它一定要小於所有子節點，也就是 $A[2:15]$。同樣的對於任何一個節點而言，它都沒有辦法和它的子樹內任何節點形成 inverse pair。

另外也可以發現樹的越左邊的節點其 index 會越小，因此要獲得最多的 inverse pair，我們要依照左到右的順序，盡可能的讓每一個節點擁有最大值。

我們假設陣列中有 $a_1 < a_2 < a_3 < ... < a_{15}$ 共 15 個不同的數字。要注意到，每一個節點的數字不能大於其子樹的任何節點，舉例來說 $A[2]$ 的最大可能值為 $a_9$、$A[4]$ 的最大可能值為 $a_{13}$。

(<mark style="background: #FF5582A6;">紅色部分</mark>)
現在我們分別賦予最左側的節點它們的最大值。

(<mark style="background: #ADCCFFA6;">藍色部分</mark>)
一旦賦予了最大值，其子樹內容的選擇就會被固定，舉例來說如果我們讓 $A[2] = a_9$，那麼其子樹的內容就一定會是 $a_{10} ~ a_{15}$，因此我們必須接著填完左子樹的內容。

(<mark style="background: #BBFABBA6;">綠色部分</mark>)
右子樹的內容剩下 $a_2 ~ a_8$ 可以分配，這時候 $A[3]$ 只能是 $a_2$ 因為它一定要小於其子樹，而剩下的點一樣依據左大右小的概念分配，最後會得到如下圖的樹：
![[School/Course Homeworks/DSA/assets/DSAHW2/treef.png|475]]

最後我們可以得到 inverse pair 的最大數量為 $53$
<div style="page-break-after:always;"></div>

## 5.
> Reference : B11902121 王哲淵 (想法提供)

要讓 $cost$ 越小，我們每一次合成都要用陣列中最小的兩個素材。

因此，我們先將陣列轉換成 min heap，然後每一回合都做 remove min 兩次，然後合成取出來的兩個素材，並把 $cost$ 加上該回合的花費，再將合成完的素材 insert 回去，重複動作直到 heap 只剩下一個元素。總共會進行 $n-1$ 次合成。

<mark style="background: #ADCCFFA6;">**時間複雜度**</mark>
將陣列轉換成 min heap : $O(n)$
每一回合做兩次 remove min 以及一次 insert : $\log n \times 3 = O(\log n)$
進行 $n-1$ 次回合 : $O(\log n) \times (n-1) = O(n \log n)$

總共 $O(n) + O(n \log n) = O(n \log n)$

<mark style="background: #ADCCFFA6;">**空間複雜度**</mark>
$O(1)$，因為我們直接使用原本的陣列來實現 heap。
<div style="page-break-after:always;"></div>

## 6.
> Reference : All by myself

每一個公會用 $G_1$ ~ $G_m$ 表示，公會 $i$ 的每一個成員用 $G_i[1]$ ~ $G_i[n]$

首先用每個公會的第一位成員 ($G_1[1]$ ~ $G_n[1]$) 建立一個有 m 個節點的 max heap

接著取出 heap 的最大值 $H[1]$ 放入 new，並從 $H[1]$ 所在的公會拿取下一個成員，將其插入 max heap，如下圖所示：
![[School/Course Homeworks/DSA/assets/DSAHW2/merge guild.png|475]]
就這樣直到每個公會的成員都被清空，因為總共要插入 $(n-1) \times m$ 個成員，總共會執行 $(n-1) \times m$ 次。

最後會剩下 max heap 裡的 m 個成員還沒被放入 new，我們依序提取最大值並放入 new 直到 heap 清空，到這裡為止演算法就完成了。

<mark style="background: #ADCCFFA6;">**時間複雜度**</mark>
1. 建立 max heap : $O(m)$
2. 執行 mn 次 remove max 和 insert : $mn \times (\log m + \log m) = O(mn\log m)$ 
3. 執行 m 次 remove max : $O(m \log m)$

總共 $O(mn\log m)$
<br>

<mark style="background: #ADCCFFA6;">**空間複雜度**</mark>
我們用了一個有 m 個節點的 max heap，為 $O(m)$

<div style="page-break-after:always;"></div>
