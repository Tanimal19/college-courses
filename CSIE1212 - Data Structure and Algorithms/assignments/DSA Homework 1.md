B11902038 資工一 鄭博允

---

# Problem 1

### 1.
> Reference : all by myself

**No.**
Consider $f(n) = 2n^3$

If $f(n) = O(n^3)$ , we have
$2n^3 \le cn^3\,$ for all $\, n \ge n_0$
$\quad2 \le c$
then we can pick any $c$ with $c \ge 2$ to satisfied the condition,
therefore $f(n)$ is belong to $O(n^3)$.

If $f(n) = O(n^2)$ , we have
$2n^3 \le cn^2\,$ for all $\, n \ge n_0$
$\quad2 \le {c \over n}$
$\,\,2n \le c$
if we pick $n = \max(n_0 , c)$ , then there's **no** $c$ satisfied the condition,
therefore $f(n)$ is not belong to $O(n^2)$.
<br>

### 2.
> Reference :
> https://yourbasic.org/algorithms/time-complexity-explained/

For line 2 is $m = l$, it's obvious that this algorithm will act like a
**sequential search** (rather than a **binary search**).

When the algorithm has longest running time , we called it **worst-case**.
For this algorithm, it's worst-case happened when it has to traverse the whole array to find the key, which is $key = A[n]$ (or there's no key in the array, same cost).

At worst-case, the **while** loop will run $n$ times, 
thus we can assume that time complexity of worst-case is  
$W(n) = an$ with $a > 0$

With common sense, $W(n) = O(n)$

> Proof :
If  $W(n) = O(n)$ , we have
$an \le cn\,$ for all $\, n \ge n_0$
then we can pick $c \ge a$ to satisfied the condition.

Sine worst-case time complexity gives a upper bound of time complexity of any other input with size $n$ , this algorithm has time complexity $O(n)$.  
<br>

### 3.
> Reference : all by myself

Consider $f(n) = n$ ,
with common sense, $f(n) = O(n)$ and thus $f(n) = O(n^2)$

But $\lim_{n \to ∞} ({f(n) \over n^2} = {n \over n^2} = {1 \over n}) = {1 \over ∞} = 0$
which is contrary to the statement.

Therefore the statement is not true.
<br>

### 4. 
> Reference : all by myself

If $\lg n = O(\sqrt{n})$ , we have
$\lg n \le c\sqrt{n}\,$ for all $\, n \ge n_0\quad$ ... [1]

Since
$\lim_{n \to ∞}\large{\lg n \over \sqrt{n}} \normalsize= \lim_{n \to ∞}\Large{{1\over n} \over {1 \over 2\sqrt{n}}} \normalsize= \lim_{n \to ∞}{ 2\over \sqrt{n}} = 0$
So when $n \to ∞$ , $\lg n$ will be smaller than $\sqrt n$ ,
and thus if we set $n_0$ big enough, the statement [1] will be true. 

Therefore $\lg n = O(\sqrt{n})$.
<br>

### 5.
> Reference : all by myself

$\sum_{i=1}^{n}{i^n} = 1^n + 2^n +\,... + (n-1)^n + n^n$

It's obvious that $n^n = O(n^n)$

> Proof :
If  $n^n = O(n^n)$ , we have
$n^n \le cn^n\,$ for all $\, n \ge n_0$
then we can pick $c \ge 1$ to satisfied the condition. 

Since $1^n < 2^n <\, ... < (n-1)^n < n^n$ when $n \ge 1$
by **Transitivity of Big-O**
we can know $1^n = O(n)\,...(n-1)^n = O(n^n)$

and by **Closure of Big-O**
we can get $1^n + 2^n +\,... + (n-1)^n + n^n = O(n^n)$

Therefore 
$\sum_{i=1}^{n}{i^n} = O(n^n)$
<br>

### 6.
> Reference : all by myself

At line 3 of  B11902777's solution,
we can't say
$|\lg(f(n)) − \lg(g(n))| = c ⇒ \lg({f(n) \over g(n)}) = c$

because it's a absolute value 
and $|\lg(f(n)) − \lg(g(n))|$ might be either $\lg({f(n) \over g(n)})$ or $\lg({g(n) \over f(n)})$

so the solution need to be fix as below :
```ad-quote
title:
$|\lg(f(n)) − \lg(g(n))| = O(1) ⇒ |\lg(f(n)) − \lg(g(n))| = c$,
for all $n > n_0$ , where $n_0, c > 0$ are constants.

$|\lg(f(n)) − \lg(g(n))| = c ⇒ \lg({f(n) \over g(n)}) = c$ or $\lg({g(n) \over f(n)}) = c$ 

$\lg({f(n) \over g(n)}) = c⇒{f(n) \over g(n)} = 2^c ⇒ f(n) = 2^cg(n)$
$\lg({g(n) \over f(n)}) = c⇒{g(n) \over f(n)} = 2^c ⇒ f(n) = \large{1 \over 2^c}\normalsize g(n)$

For ${1 \over 2^c} < 1$
Take $c' = {1 \over 2^c}$ , we have $f(n) ≤ c'g(n)$ for all $n > n_0$

$Q.E.D.$
```

<br>
<p style="page-break-after:always"></p>

# Problem 2

### 1.
> Reference : all by myself

```pseudo
\begin{algorithm}
\caption{Find Missing}
\begin{algorithmic}
\COMMENT{array begin at [1]}
\PROCEDURE{Find-Missing}{$A, l$}
	\STATE Set $find[2]$ = ${0,0}$
	\STATE \\
	\FOR{$i=0$ \TO $i=2$}
		\STATE Set $tail$ = $l$
		\STATE Set $head$ = 1
		\STATE \\
		\WHILE{$head \le tail$}
			\STATE $mid$ =ceil($head+tail \over 2$)
			\IF{$A[mid] > mid+i$}
				\STATE $tail = mid-1$
			\ELIF{$A[mid] \le mid+i$}
				\STATE $head = mid+1$
			\ENDIF
		\ENDWHILE
		\STATE \\
		\IF{$A[mid] == mid+i$}
			\STATE $find[i+1] = mid+1+i$
		\ELIF{}
			\STATE $find[i+1] = mid+i$
		\ENDIF
		\STATE \\
		\STATE $i = i+1$
	\ENDFOR
	\STATE \\
	\IF{$find[1] == find[2]$}
		\STATE $find[2] = find[2]+1$
	\ENDIF
	\STATE \\
	\RETURN $find$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
24 lines (without blanked)
<br>

### **Explanation**
For this problem, we can find that:
If a element is correct, it's index should fit stored ID.
Therefore, by comparing index ($mid$) and stored ID ($A[mid]$), 
we can know the missing number happend on left or right side of the element $mid$.
And we can apply this property to build a binary-search-liked algorithm. 
<br>

### **Time & Space Complexity**
In **while** loop, we narrow down the range by half for each loop. After $k$ loop we have the range with 
$${n \over 2^k}$$
and the loop will run until range equal to 1, so we can get the time complexity of **while** loop :
$${n \over 2^k} = 1\ ⇒\ k = \log n$$

Since there's two missing ID, we use **for** loop to run the search twice,
so the time complexity of this algoritm is
$$2\log n = O(\log n)$$
<br>
<p style="page-break-after:always"></p>

### 2.
> Reference : all by myself

```pseudo
\begin{algorithm}
\caption{Find Pair}
\begin{algorithmic}
\COMMENT{array begin at [1]}
\PROCEDURE{Find-Pair}{$A, l$}
	\STATE Set $now$ = $A[1]$
	\STATE Set $begin$ = 1
	\STATE Set $cursor$ = 1
	\STATE \\
	\FOR{$i=1$ \TO $i = l$}
		\IF{$A[i] \ge now\times2$}
			\STATE $cursor = begin$
			\STATE $begin = i$
			\STATE $now = now\times 2$
		\ENDIF
		\STATE \\
		\IF{$A[i]$ is $even$ \AND $cursor < begin$}
			\WHILE{$A[cursor] == 0$ \AND $cursor < begin$}
				\STATE $cursor = cursor + 1$
			\ENDWHILE
			\STATE \\
			\IF{$A[cursor]\times2 == A[i]$}
				\STATE $A[cursor] = 0$
				\STATE $A[i] = 0$
				\STATE $cursor = cursor + 1$
			\ENDIF
		\ENDIF
	\STATE $i = i+1$
	\ENDFOR
	\STATE \\
	\FOR{$i=1$ \TO $i = l$}
		\IF{}
			\RETURN 0
		\ENDIF
	\STATE \\
	\STATE $i = i+1$
	\ENDFOR
	\STATE \\
	\RETURN 1
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
29 lines (without blanked)
<br>

### **Explanation**
- Every number must fit in an only "pair list", for each pair list :
$1, 2, 4, ...$
$3, 6, 12, ...$
$5, 10, 20, ...$
we can found that *odd numbers* must be the first number of each pair list.

- Since the array is sorted, the double of $a+1$ which is $2a+2$ must appear after $2a$, and same does $a+2$ and it's double and so on.

Having these properties, we can cut the array into below form (c for $cursor$, b for $begin$)
![[School/Course Homeworks/DSA/assets/DSAHW1/findpair.png|500]]
For every number in section 1, their double (if exist) must in section 2.
Therefore, we can pair a section with it's previous section in sequence. On the other hand, we only look for *even numbers* since *odd numbers* always be the first number of each pair list.

At first **for** loop, we pair $begin$ with $cusor$ , if they are paired, we set both to 0 ; if $cursor$  is 0, we shift cursor to the next value ; otherwise, we keep go on until meeting the next sector. Once we meet the next section, we set $begin$ and $cusor$ to the first number of each section, then doing the same thing.

At second **for** loop, we check if every number is 0 , which means every number is being paired.
<br>

### **Time & Space Complexity**

Since $cursor$ never decrease, the **while** loop in the first **for** loop will run at most $n$ times in the entire **for** loop. And each **for** loop run at most $n$ times, so the time complexity of this algorithm is $3n+c = O(n)$ with some constant $c << n$.

The extra-space complexity is $O(1)$, since we only use several variables.  
<br>
<p style="page-break-after:always"></p>

## 3.
> Reference : all by myself

```pseudo
\begin{algorithm}
\caption{Merge}
\begin{algorithmic}
\COMMENT{$*$ for pointer}
\PROCEDURE{Merge}{$*head1$, $*head2$}
	\STATE Set $*now, *next1, *next2, *temp$
	\STATE \\
	\IF{$head1 \to value \le head2 \to value$}
		\STATE $now = head1$
		\STATE $next2 = head2$
	\ELSE
		\STATE $now = head2$
		\STATE $next2 = head1$
	\ENDIF
	\STATE $next1 = now \to next$
	\STATE $now \to next$ = NIL
	\STATE \\
	\WHILE{$next1 \ne$ NIL \AND $next2 \ne$ NIL}
		\IF{$next1 \to value \le next2 \to value$ \OR $next2 \to value ==$ NIL}
			\STATE $temp = next1 \to next$
			\STATE $next1 \to next = now$
			\STATE $now = next1$
			\STATE $next1 = temp$
		\ELIF{$next1 \to value \gt next2 \to value$ \OR $next1 \to value ==$ NIL}
			\STATE $temp = next2 \to next$
			\STATE $next2 \to next = now$
			\STATE $now = next2$
			\STATE $next2 = next1$
			\STATE $next1 = temp$
		\ENDIF
	\ENDWHILE
	\STATE \\
	\RETURN $now$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
26 lines (without blanked)
<br>
<p style="page-break-after:always"></p>

### **Explanation**
This algorithm work as below picture:
![[School/Course Homeworks/DSA/assets/DSAHW1/merge.png]]
Start from the smaller value in $head1$ and $head2$, compare it with the next value in the same list ($next1$) and the next value in the another list ($next2$), then let the smaller one point to $now$ and shift $now$ to the smaller one.
<br>

### **Time & Space Complexity**
Assume that the sum of length of list1 and length of list2 is $n$.
The **while** loop will run $n$ times, and the total time complexity is $n+c = O(n)$ with some constant $c << n$

The extra-space complexity is $O(1)$, since we only use several variables.
<br>
<p style="page-break-after:always"></p>

## 4.
> Reference : all by myself

```pseudo
\begin{algorithm}
\caption{Negation Sort}
\begin{algorithmic}
\COMMENT{$*$ for pointer}
\PROCEDURE{Sort}{$*head$}
	\STATE Set $*now = head$
	\STATE Set $*newhead = head$ 
	\STATE Set $*nlist =$ NIL
	\STATE Set $*temp$
	\STATE \\
	\WHILE{$now \to next \to next \ne$ NIL}
		\IF{$now \to next \to value < 0$ }
			\STATE $temp = now \to next$
			\STATE $now \to next = now \to next \to next$
			\STATE $temp \to next = nlist$
			\STATE $nlist = temp$
		\ELSE
			\STATE $now = now \to next$
		\ENDIF
	\ENDWHILE
	\STATE $now \to next \to next = nlist$
	\STATE \\
	\IF{$head \to value < 0$}
		\WHILE{$now \to next \ne$ NIL}
		\STATE $now = now \to next$
		\ENDWHILE
		\STATE $now \to next = head$
		\STATE $newhead = head \to next$
		\STATE $head \to next$ = NIL
	\ENDIF
	\STATE \\
	\RETURN $newhead$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
24 lines (without blanked)

### **Explanation**
Assume the list before sorted is
$18 → -15 → -11 → 8 → 7 → -4 → 2 → 1$
If we separate negative numbers from the list:
Original : $18 → 8 → 7 → 2 → 1$
Negative : $-15 → -11 → -4$
we named the list with negative numbers $nlist$, if we reverse $nlist$ and concatenate to the tail of original list, it will produce:
$18 → 8 → 7 → 2 → 1 → -4 → -11 → -15$
And that's exactly what we want.

We begin at head of list, if the next value is negative, we put it into head of $nlist$ ; otherwise, we move to next node.
In the first **while** loop, we only check the second to second last node, because the absolute value of last node is the smallest, so we can simply concatenate the head of $nlist$ to the last node.
Then, we check the head of original list: if it's negative, we concatenate it to the tail of the list, set $newhead$ to the next node of original head.
In the end, we return $newhead$.
<br>

### **Time & Space Complexity**
Assume the length of original list is $n$.
In the first **while** loop, we run $n-2$ times. If $head$ is negative, we run another **while** loop for the length of $nlist$, which is at most $n$. 
Therefore, the time complexity of worst-case is $2n-2 = O(n)$, so the time complexity of algorithm is $O(n)$.

The extra-space complexity is $O(1)$, since we only use several variables.
<br>
<p style="page-break-after:always"></p>

# Problem 3

## 1.
> Reference : all by myself

Assume human can simply distinguish the expression is converted or not.  

1. From left to right, find the smallest $()$, inside the $()$:
	**a.** From left to right, find $\times /$ , move number (or converted expression) on the right side to the left side, until there's no any $\times /$.
	**b.** From left to right, find $+-$ , move number (or converted expression) on the right side to the left side, until there's no any $+-$.
remove the $()$

2. Repeat step 1 until there's no any $()$

3. **a.** From left to right, find $\times /$ , move number (or converted expression) on the right side to the left side, until there's no any $\times /$.
	**b.** From left to right, find $+-$ , move number (or converted expression) on the right side to the left side, until there's no any $+-$.

4. Repeat step 3 until convert whole expression

Finish.

```ad-quote
title:
Steps : 
5 + 3 × 4 − (8 × (2 <mark style="background: #FFF3A3A6;">+</mark> 3) − (1 9 6 ×+) / 5)
5 + 3 × 4 − (8 × 2 3 + − (1 + 9 <mark style="background: #FFF3A3A6;">×</mark> 6) / 5)
5 + 3 × 4 − (8 × 2 3 + − (1 <mark style="background: #FFF3A3A6;">+</mark> 9 6 ×) / 5)
5 + 3 × 4 − (8 <mark style="background: #FFF3A3A6;">×</mark> 2 3 + − 1 9 6 × + / 5)
5 + 3 × 4 − (8 2 3 + × − 1 9 6 × + <mark style="background: #FFF3A3A6;">/</mark> 5)
5 + 3 × 4 − (8 2 3 + × <mark style="background: #FFF3A3A6;">−</mark> 1 9 6 × + 5 / )
5 + 3 <mark style="background: #FFF3A3A6;">×</mark> 4 − 8 2 3 + × 1 9 6 × + 5 / −
5 <mark style="background: #FFF3A3A6;">+</mark> 3 4 × − 8 2 3 + × 1 9 6 × + 5 / −
5 3 4 × + <mark style="background: #FFF3A3A6;">−</mark> 8 2 3 + × 1 9 6 × + 5 / −
5 3 4 × +  8 2 3 + × 1 9 6 × + 5 / − −
```

<br>
<p style="page-break-after:always"></p>

## 2.
> Reference : all by myself

Assume human can remember the whole expression.
(if can't, just take a note)

From left to right:
- If it's a number, stored it.
- If it's a symbol, calculate it with the latest two numbers, then stored the answer.

Finish.

```ad-quote
title:
Steps : 
5 2 7 × + 6 3 4 − × 4 8 6 × + 4 /  −  −
<mark style="background: #FFF3A3A6;">5 2 7</mark> → stored
<mark style="background: #FFF3A3A6;">×</mark> → 2 × 7 = 14 → stored
<mark style="background: #FFF3A3A6;">\+</mark> → 5 + 14 = 19 → stored
<mark style="background: #FFF3A3A6;">6 3 4</mark> → stored
<mark style="background: #FFF3A3A6;">\-</mark> → 3 - 4 = -1 → stored
<mark style="background: #FFF3A3A6;">×</mark> → 6 × -1 = -6 → stored
<mark style="background: #FFF3A3A6;">4 8 6</mark> → stored
<mark style="background: #FFF3A3A6;">×</mark> → 8 × 6 = 48 → stored
<mark style="background: #FFF3A3A6;">\+</mark> → 4 + 48 = 52 → stored
<mark style="background: #FFF3A3A6;">4</mark> → stored
<mark style="background: #FFF3A3A6;">/</mark> → 52 / 4 = 13 → stored
<mark style="background: #FFF3A3A6;">\-</mark> → 13 - (-6) = 19 → stored
<mark style="background: #FFF3A3A6;">\-</mark> → 19 - 19 = 0 → ans
```

<br>
<p style="page-break-after:always"></p>

## 3.
> Reference : all by myself

$S_0, S_1, S_2$ for stack 0, 1, 2

Steps :
1. pop "4" from $S_0$ and push it to $S_2$
2. pop "1" "2" from $S_0$ and push it to $S_1$
3. pop "3" from $S_0$ and push it to $S_2$
4. pop "2" "1" from $S_1$ and push it to $S_2$
<br>

## 4.
> Reference : all by myself

Steps :
1. Find the max value $M$ in $S_0$.
2. Pop every number on top of $M$ (exclude $M$) to $S_1$.
3. Check every number in $S_1$ is descending from top to bottom. If not, return FALSE.
4. Pop $M$ from $S_0$ and push it to $S_2$.
5. Repeat step 1~4 until $S_0$ is empty, then return TRUE.

Time Complexity : $O(n^2)$
<br>

## 5.
> Reference : all by myself

The first student should park zir bike at position $5$, since maximum distance between existing bike is $10$, and $0 + 10/2 = 5$.
Now the bikes' positions are
$[~0, ~5, ~10, ~15.5, ~17~]$

Similarly, the second student park at $10 + 5.5/2 = 12.75$.
$[~0, ~5, ~10, ~12.75, ~15.5, ~17~]$
And the third student park at $0 + 5/2 = 2.5$
$[~0, ~2.5, ~5, ~10, ~12.75, ~15.5, ~17~]$

So the distance between the third student’s bike to the nearest bike is $2.5$.
<br>
<p style="page-break-after:always"></p>

## 6.
> Reference : all by myself

```pseudo
\begin{algorithm}
\caption{Bike Parking}
\begin{algorithmic}
\COMMENT{array begin at [0]}
\PROCEDURE{Parking}{$A, n, m$}
	\STATE Set $Q[m]$
	\STATE $Q[0] = 0$
	\STATE Set $c = 0$
	\STATE Set $d, d_1, d_2$
	\STATE \\
	\FOR{$i=0$ \TO $i=m$}
		\IF{$c \le n$}
			\STATE $d_1 = A[c+1] - A[c]$
		\ELSE
			\STATE $d_1 = 0$
		\ENDIF
		\STATE $d_2 =$ Seek($Q.head$)
		\STATE \\
		\IF{$d_1 \ge d_2$}
			\STATE $c = c + 1$
			\STATE $d = {d_1 \over 2}$
		\ELIF{$d_1 \lt d_2$}
			\STATE Dequene($Q$)
			\STATE $d = {d_2 \over 2}$
		\ENDIF
		\STATE \\
		\STATE Enquene($Q, d$)
		\STATE Enquene($Q, d$)
		\STATE $i = i + 1$
	\ENDFOR
	\RETURN $d$
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
```
24 lines (without blanked)
<br>

### **Explanation**

We use a quene $Q$ to stored the distance between bikes after each student parked. For student should park zir bike in the largest distance, we find the largest distance as method below.

Assume the distance of initial bikes $D : [~a, ~b~, ...]$ with $a>b$.
At first step , because $Q.head = 0$, we enquene $d = {a \over 2}$ into $Q$.
$D : [~b~, ...]$
$Q : [~{a \over 2}~]$ 

The next step we will enquene $d = {b \over 2}$ or $d = {a \over 4}$ into $Q$, 
since ${a \over 2} > {b \over 2}$ and ${a \over 2} > {a \over 4}$, $Q.head$ is always the largest in $Q$
 
Therefore, we can use $d_1$ for the distance between $A[c]$ and $A[c+1]$ with $c = 0$ at begin, since the distances between initial bikes are descending, $d_1$ must be the largest in $A$.
And we use $d_2$ for the distance after parking a bike $Q.head$.
By comparing $d_1$ and $d_2$, we can find the largest distance.

For every turn, we compare $d_1$ and $d_2$, student should park zir bike in the larger one. Since the bike parked in the middle, we divide the larger one by 2 and assign it to $d$, and enquene $d$ to $Q$ for twice, so they can be use by another student. 

After $m$ turns, the $m$-th student park zis bike, and the $d$ is exactly the distance between the bike parked by the $m$-th student and the nearest bike.
<br>

### **Time & Space Complexity**
For time complexity, the **for** loop run $m$ times, so the time complexity is $m+c = O(m) = O(n+m)$ with some constant $c << m$

For every turn,  we enquene two elements into $Q$, in the end, $Q$ will contain $2m$ elements, and the original space of bikes location is $n$. So the total space complexity is $n+2m+c = O(m+n)$ with some constant $c << m$