B11902038 鄭博允

---

# Problem 1
## 1.
> Reference : All by myself

We use $\pi(i)$ to represent the prefix function of element with index $i$ in string $s$. $1 \le i \le s$
To make $t$ be it's maximum, we need to make the overlap area of adjacent string, which is "same prefix and suffix of $s$", as long as possible. 
The length of "same prefix and suffix of $s$" is $\pi(s)$.
Here is the step of construct the string $t$ :
1. place $s$ at the begin of $t$
2. at $\pi(s)$ spaces in front of tail, place $s$
3. keep doing step 2 until the total length greater than $n$

<div style="page-break-after:always;"></div>

## 2.
> Reference : All by myself

**演算法說明 / 流程**
<mark style="background: #ADCCFFA6;">Pre-processing</mark> :
新增一個陣列 $M[]$，每一個 $M[n]$ 內記錄著 $s[1:n]$ 的 hash 值，
也就是 $M[n] = \sum^{n}_{i=0} k^{n-i} \cdot s[i] \mod Q$

而我們可以利用 $M[]$ 算出任意子字串的 hash 值：
$s[l:r] = \sum^{r-l}_{i=0} k^{r-l-i} \cdot s[l+i] \mod Q = (M[r] - k^{r-l} \cdot M[l]) \mod Q$

以下為計算 $M[]$ 的 pseudo code :
```c
m = 0
for i=1 to n
	m = (k*m + s[i]) mod Q
	M[i] = m 
```
<br>

<mark style="background: #ADCCFFA6;">Query (單次)</mark> : 
1. 檢查 $r_1$ 是否等於 $r_2$，以及  $l_1$ 是否等於 $l_2$
	皆相等則回傳 TRUE，繼續下一圈
	否則繼續下一步

2. 檢查 $r_1 - l_1$ 和 $r_2 - l_2$ 是否相等，
	不相等則回傳 FALSE，繼續下一圈
	相等繼續下一步
 
3. 計算 $s[l_1:r_1]$ 和 $s[l_2:r_2]$ 的值，比較兩者是否相等
	不相等則回傳 FALSE，相等則回傳 TRUE (假設沒有 collision)

以下為 Query 部分的 pseudo code :
```c
if (r1 == r2 && l1 == l2)
	return TRUE

if (r1-l1 != r2-l2)
	return FALSE

s1 = (M[r1] + k^(l1-r1) * M[l1]) mod Q
s2 = (M[r2] + k^(l2-r2) * M[l2]) mod Q

if (s1 == s2)
	return TRUE
else
	return FALSE
```
<br>

**時間複雜度**
Pre-processing 花費 $O(n)$，
處理每一個 query 只要花費 $O(1)$，處理 $q$ 個會花費 $O(q)$
因此總共為 $O(n+q)$

<div style="page-break-after:always;"></div>

## 3.
> Reference : All by myself

**演算法說明**
假設字串 $s$ 最多可以被切成 $S_1, S_2\ ...\ S_c$ 共 $c$ 個長度為 $\large n \over c$ 的相同 substrings
我們可以發現 $s$ 最長的相同前綴後綴為
$S_1 + S_2 +\ ...\ + S_{c-1}$ 和 $S_2 +\ ...\ + S_{c-1} + S_c$
因此可以得知 $\pi(n) = \large{c-1 \over c}\normalsize n$
並獲得 $c$ 的可能最大值  $c_{max} = \large{n \over n - \pi(n)}$
而一個 substring 的長度即為 $\large{n \over c_{max}}\normalsize = n - \pi(n)$

由於每個字串一定都能被切成 1 等分，所以 $c_{max}$ 的最小值為 2，$c_{max} = \large{n \over n - \pi(n)}\normalsize \ge 2$，我們可以得到 $2\pi(n) \ge n$

剩下所有可能的 $c$ 皆為 $c_{max}$ 的因數
舉例來說，如果 $s$ 可以被切成 8 個相同 substrings $b_1$，那它也可以被切成 1, 2, 4 個相同 substrings；但是不能被切成 3 個相同 substrings ，因為 3 不是 8 的因數，所以會破壞掉原本的 $b_1$，substrings 就不會相等。

<br>

**演算法流程**
1. 計算 prefix function

2. 檢查 substring 長度是否為 $n$ ，如果不是代表我們無法將 $s$ 平均劃分 ($c$ 不會是整數)；檢查 $2\pi(n) \ge n$
如果以上兩個條件有任何一個不成立，則 $c$ 只可能是 1，結束演算法

3. 前面兩個步驟皆成立，則所有可能的 $c$ 即為 $c_{max} = \large{n \over n - \pi(n)}$ 的所有因數

以下為 pseudo code :
```c
lps[] = Prefix-Function(s[], n)

// 檢查
sublen = n-lps[n]
if ( n > 2*lps[n] ) OR ( n % sublen != 0 )
	return c = 1

// 計算因數
cmax =  n/(n-lps[n])
for i=1 to cmax
	if cmax % i == 0
		i is factor of cmax

return all factors of cmax
```
<br>

**時間複雜度**
計算 prefix function 時花費 $O(n)$，計算因數花費 $O(n)$，
所以總共為  $O(n) + O(n) = O(n)$

<div style="page-break-after:always;"></div>

## 4.
> Reference : All by myself

**演算法說明 / 流程**
要找到字典序比 $key$ 還小的子字串 $s'$，我們只需要在 $s$ 裡面找到第一個比 $key$ 還小的字元即可。

這裡先介紹我們的 modified prefix function，從計算"最長的相同前綴後綴"變成計算"最長的比前綴小的後綴"，舉例來說 "CCBDC" 的 $\pi(3) = 2$ 因為 "CC" > "CB"，而 $\pi(2) = 0$ 因為 "C" 沒有小於 "C"

我們對每個字元的判斷則如下：
- 如果 $s[i]$ 比 $key[q+1]$ 小，我們就找到了，此時子字串的起始點會是 $i-q$
- 如果 $s[i]$ 和 $key[q+1]$ 相等，我們就繼續比下一個字元
- 如果 $s[i]$ 比 $key[q+1]$ 大，代表這不是我們要找的，要比對下一個子字串，但是我們可以不用從 $key$ 的第一個字元開始比。
以下圖為例，當 $i = 4, q = 3$ 時 $s[i]$ > $key[q+1]$
![[School/Course Homeworks/DSA/assets/DSAHW3/5.png|350]]
$\pi(q) = 2$，代表"最長的比前綴小的後綴"長度為 2
前綴為 $key[1:2]$ = "CC"、後綴為 $key[2:q]$ = "CB"，而 $key[1:2]$ > $key[2:q]$
由於 $s[2:i-1]$ 和 $key[2:q]$ 是一樣的，代表 $key[1:2]$ > $s[2:i-1]$，此時就找到子字串了，而起始點會是 $i-\pi(q) = 2$
另一方面，如果今天 $\pi(q) = 0$，代表沒有"最長的比前綴小的後綴"，因此要從 $key$ 的頭開始比，也就是說我們下一個要比的是 $s[i]$ 和 $key[1]$。
<br>

**時間複雜度**

計算 modified prefix function 的前半部分和計算一般的 prefix function 是一樣的，我們會先計算最長的相同的前綴後綴。然而，一旦遇到字典序比較小的後綴，我們就計算那一個後綴的長度，並把前面的都變成零，後面的依序加一。
時間複雜度最多為 $2m = O(m)$。

而比對的部分最多花上 $O(n)$，所以時間複雜度總共為 $O(n+m)$

<div style="page-break-after:always;"></div>

## 5.
> Reference : All by myself

假設 index 從 0 開始
如果 $s$ = "5554321"、$key$ = "555"
GPT's algorithm 會回傳 $3$, 但正確的答案應該是 $1$,
因為 $s[1:3] =$ "554" < "555"

這個演算法會出錯是因為它沒有每一個 $s$ 的 substring 都和 $key$ 比對到。
照理來說，在第 5 行的 while loop 裡面，每一輪$idx_s$ 應該只能加一，這樣才能每一個 substring 都比對到，但是在第 7 行的 while loop 裡面，$s[idx_s]$ 一旦和 $key$ 的任一個元素比對過後就會被跳過，導致有些 substring 會被忽略。
以上面的例子來說， $s[1]$ 在整個演算法中只有和 $key[1]$ 比對過，而沒有和 $key[0]$ 比對過。

<div style="page-break-after:always;"></div>

# Promblem 2
## 1.
> Reference : All by myself

```c
[152, 234, 57, 8, 601, 310]
// 以個位數排序
[310, 601, 152, 234, 57, 8]
// 以十位數排序
[8, 601, 310, 234, 152, 57]
// 以百位數排序
[8, 57, 152, 234, 310, 610]
```

<div style="page-break-after:always;"></div>

## 2.
> Reference : All by myself

**演算法說明 / 流程**
如果從 $B$ 的尾端開始跑，每一輪都把 $B[i]$  insert 進 $C$，
到了 $i=k$ 的時候，我們已經將 $B[k:N]$ 都 insert 到 $C$ 裡面了。
對於 $B[k]$ 來說，inversion 發生的條件是 $i>k$ $B[i]<B[k]$，
由於目前 $C$ 裡面的都是 index > $k$ 的元素，
因此 $C$ 裡面數值小於 $B[k]$ 的元素出現次數即是 inversion 發生的次數，
也就是 $sum(B[k]-1)$。

以下是 pseudo code :
```c
// Array begin at 0
Find-Inversion(B[], N, K){
	inversion = 0

	for i=N-1 to 0 
		insert(B[i])
		inversion += sum(B[i]-1)

	return inversion
}
```
<br>

**時間複雜度**
for 迴圈會跑 $N$ 次，每一輪會花費 $2\log K$，
所以總共為 $N \cdot 2 \log K = O(N \log K)$ 

<div style="page-break-after:always;"></div>

## 3.
> Reference : All by myself

**演算法說明 / 流程**
string $s$ with length $N$ has 
1 substring with length $N$
2 substrings with length $N-1$
3 substrings with length $N-2$
...
N substrings with length $1$
Therefore it has at most $\sum_{i=1}^{N} i$ different substrings.

If we don't want to count substrings that contain character $σ$ , we can simply divide the string into two strings by $σ$.
For example, if $s$ = "NTUSUCKS" and $σ$ = "U", we can divide $s$ into 3 substrings : "NT", "S", "CKS".
Therefore, the number of substrings that do not contain "U" is the sum of the number of substring's substrings, which is $3 + 1 + 6 = 10$

Here's the pseudo code :
```c
Substring-Count(s[], N, σ)
// array start at 1
len = 0
ans = 0

for i=1 to N
	if s[i] == 'σ' ; then
		len = 0
	else
		len ++
		ans += len

return ans
```

<br>

**時間複雜度**
Since there's only one $for$ loop that run $N$ times, the time complexity is $O(N)$

<div style="page-break-after:always;"></div>

## 4.
> Reference : All by myself

**演算法說明**
以 $s$ = "abbc" 為例，
遇到第一個 "b" 時，我們可以將 $s$ 分成 "ab" 和 "bbc"
![[School/Course Homeworks/DSA/assets/DSAHW3/1.png|300]]
如果將 "ab" 的 "b" 拿掉，會失去 "b", "ab" 2 種子字串 (失去的數量實際上就是子字串的長度)
如果將 "bbc" 的第一個 "b" 拿掉，會失去 "b", "bb", "bbc" 3 種子字串
由於前後的字串可以互相組合(第一個 "b" 會重複)，所以總共會失去 $2 \times 3 = 6$ 種子字串："b", "bb", "bbc", "ab", "abb", "abbc"

在下面的 pseudo code 裡面，我們用 `prev` 代表前面的子字串長度 ("ab")，`len[index]` 代表後面的子字串長度 ("bbc")。每次分割完後，前面的子字串就不會再被用到了，而後面的子字串因為還沒有全部檢查完，不確定還要不要再分割，所以會記錄下來。
我們用 `count[]` 來記錄子字串出現的數量，一開始都是 $N!$，因為還沒有任何字元被拔除。每一輪，字元對應的 `count[index]` 會被減去失去的子字串數量。

以下為 pseudo code :
```c
// array start at 1
set Cset[C] 
set every element of len[C] with N
set every element of count[C] with N!

for i=1 to N
	index = Cset[s[i]]
	prev = len[index] - (N-i)
	len[index] = N-i
	count[index] -= ( prev*(len[index]+1) )

return count[]
```
<br>

**時間複雜度**
計算 $N!$ 會花 $O(N)$，
初始化 `len[]` `count[]` 會花 $O(C)$，
for 迴圈會花 $O(N)$，
因此總共會花 $O(N) + O(C) + O(N) = O(C+N)$

<div style="page-break-after:always;"></div>

# Problem 3
## 1.
> Reference : All by myself

由於 random seed 固定，我們可以知道第幾次的 UNION 會是如何合併的。
cnt 越大，代表 FIND-SET 越多次，也就是每一次都要從 set 的尾端開始查，而且每次 UNION 都要讓大的 set 合併到小的 set，讓樹越高(不平衡)。

輸出結果：
![[School/Course Homeworks/DSA/assets/DSAHW3/2.png]]

測資：
```c
djs_union(1, 0);
djs_union(2, 0);
djs_union(3, 0);
djs_union(0, 4);
djs_union(0, 5);
djs_union(6, 0);
djs_union(7, 0);
djs_union(0, 8);
djs_union(9, 0);
djs_union(0, 10);
djs_union(11, 0);
djs_union(12, 0);
djs_union(0, 13);
djs_union(0, 14);
djs_union(0, 15);
```

<div style="page-break-after:always;"></div>

## 2.
> Reference : All by myself

每一次 UNION 都會是 ind 較小的 set 合併到 ind 較大的 set，根據上一題的做法，我們要讓 ind 較小的 set 同時也是數量較大的 set，也就是說我們要從 ind 最小的 set 開始合併，讓 ind 值最大的 set 最晚合併。
我們可以將隨機數列的前 16 個數值依大小排列，然後從 ind 最小的兩個 set 開始 UNION。

輸出結果(雖然看起來跟上一張很像，但真的不是同一張)：
![[School/Course Homeworks/DSA/assets/DSAHW3/3.png|475]]

測資：
```c
djs_union(10, 0);
djs_union(10, 6);
djs_union(10, 14);
djs_union(10, 15);
djs_union(10, 5);
djs_union(10, 11);
djs_union(10, 7);
djs_union(10, 2);
djs_union(10, 1);
djs_union(10, 13);
djs_union(10, 9);
djs_union(10, 8);
djs_union(10, 4);
djs_union(10, 12);
djs_union(10, 3);
```

<div style="page-break-after:always;"></div>

## 3.
> Reference : All by myself

**演算法說明**
這裡我們使用 tree + union-bysize + path compression 來實現 disjoint set。

我們用 `set[0:n-1]` 來代表 row 0 ~ n-1、用 `set[n:n+m-1]` 來代表 column 0 ~ m-1。
如果 row 0 和 row 4 互通、row 4 又和 row 2 互通，代表 row 2 也可以透過 row 4 跳到 row 0，所以只要在同一個 set 裡面的數字都是可以互通的。

假設在演算法的一開始，我們會先 MAKE-SET `set[0:n+m-1]`。
```c
for i=0 to n+m-1
	MAKE-SET(i)
```

每次的 INSTALL，我們會將兩個 rows (或 columns) 合併到同一個 set，代表它們可以互通，也就是 UNION(x, y)。
如果是要 UNION columns，我們要將 x、y 都加上 n 才會對上它們對應的 column。
```c
INSTALL(x, y, flag)
	if flag == 1
		x = n+x
		y = n+y
	
	UNION(x, y)
```

每次的 QUERY 都會做 4 次 FIND-SET，如果 row x 和 row 0 在同一個 set、column y 和 column 0 在同一個 set，代表它們可以互通。
```c
QUERY(x, y)
	if FIND-SET(x) == FIND-SET(0) && FIND-SET(n+y) == FIND-SET(n+0)
		return TRUE
	else
		return FALSE
```

<div style="page-break-after:always;"></div>

## 4.
> Reference : All by myself

**演算法說明**
先以 row 為例，假設有 row 0 ~ 5，而連接的狀態如下：
![[School/Course Homeworks/DSA/assets/DSAHW3/6.png|225]]
我們將每個 row 分成兩個 type : a (上圖紅色) 和 b (上圖藍色)，舉例來說 row 1 會有 $1a$ 和 $1b$ 兩種。如果我們使 type a 和 type b 間隔相連，會發現所有 type a 之間都需要 double-bounces 才能到達，所有 type b 亦然。因此我們只要在 UNION 兩個 rows 時確保它們是不同的 type，這樣只要在最後找到和 row 0 在同一個 set 而且 type 也相同的 rows，就是可以 double-bounces 的 rows。
以上圖為例，最後的 set 會變成 $[\ 0b, 1b, 2a, 3b, 4a, 5b\ ]$。

假如有特殊的情況，例如我們要 UNION 上圖的 2 和 4，兩者已經在同一個 set 而且 type 是一樣的，代表它們可以互相 double-bounces，現在又要將它們 UNION 代表它們也可以互相 bounce，所以我們只要將兩種 type 都合併在一起就行了，也就是說最後的 set 會變成 $[\ 0b, 1b, 2a, 2b, 3b, 4a, 4b, 5b\ ]$。

在實作部分，我們使用以下 index 來代表不同 rows 和 columns
`set[0:n-1]` : row $0_a$  ~ $(n-1)_a$
`set[n:2n-1]` : row $0_b$  ~ $(n-1)_b$
`set[2n:2n+m-1]` : column $0_a$  ~ $(m-1)_a$
`set[2n+m:2n+2m-1]` : column $0_b$  ~ $(m-1)_b$

和上一題一樣，假設我們一開始就先 MAKE-SET `set[0:2n+2m-1]`。
```c
for i=0 to 2n+2m-1
	MAKE-SET(i)
```

每次的 INSTALL，我們都會合併 x、y 的不同 type，舉例來說，要 INSTALL (1, 3) 的話，我們就會合併 $1_a, 3_b$ 和 $1_b, 3_a$。
這樣做的原因是為了確保每次都能夠合併不同的 type，而不會發生兩個 type 一樣的情況。舉例來說，假設只有兩個 set : $[\ 1_a, 2_b\ ]$ $[\ 3_a, 4_b\ ]$，如果要 INSTALL(1, 3)，那我們不能直接 UNION ($1_a, 3_a$)，因為 $1_a$ 和 $3_a$ 在同一個 set 代表 $1_a$ 和 $3_a$ 可以互相 double-bounces，而違反了規則。
如果我們每次都合併 x、y 的不同 type，我們就會有四個 set : $[\ 1_a, 2_b\ ]$ $[\ 3_a, 4_b\ ]$  $[\ 1_b, 2_a\ ]$ $[\ 3_b, 4_a\ ]$，如此一來我們就可以 UNION($1_a, 3_b$) 和 UNION($1_b, 3_a$) 而變成兩個 set :  $[\ 1_a, 2_b, 3_b, 4_a\ ]$ $[\ 1_b, 2_a, 3_a, 4_b\ ]$

```c
INSTALL(x, y, flag)
	
	if flag == 1 // UNION COLUMN 
		xa = 2n+x
		ya = 2n+y
		xb = 2n+m+x
		yb = 2n+m+y
	else // UNION ROW 
		xa = x
		ya = y
		xb = n+x
		yb = n+y
		
	UNION(xa, yb)
	UNION(xb, ya)
```

如果 (x, y) 可以只用 double-bounces 到達 (0, 0)，代表：
row $x_a$ 和 row $0_a$ (或是 row $x_b$ 和 row $0_b$) 在同一個 set 
**而且**
column $y_a$ 和 column $0_a$ (或是 column $y_b$ 和 column $0_b$) 在同一個 set 
```c
QUERY(x, y)
	if (FIND-SET(x) == FIND-SET(0) || 
		FIND-SET(x+n) == FIND-SET(0+n)) && 
	   (FIND-SET(2n+y) == FIND-SET(2n+0) || 
		FIND-SET(2n+m+y) == FIND-SET(2n+m+0))
		return TRUE
	else
		return FALSE
```

<div style="page-break-after:always;"></div>

## 5.
I didn't write it.