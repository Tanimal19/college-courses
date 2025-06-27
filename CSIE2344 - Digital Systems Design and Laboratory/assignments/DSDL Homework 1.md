B11902038 資工一 鄭博允

---
# 1.

Assume base is $N$
then we can transform the operation as below
$$
\begin{aligned}
&(0\ N^2 + 2\ N^1 + 4\ N^0 )\\
&+ (0\ N^2 + 4\ N^1 + 3\ N^0 )\\
&+ (0\ N^2 + 1\ N^1 + 3\ N^0 )\\
&+ (0\ N^2 + 3\ N^1 + 3\ N^0 )\\
&= (2\ N^2 + 0\ N^1 + 1\ N^0 )
\end{aligned}
$$

and get 
$$
\begin{aligned}
10\ N + 13 &= 2\ N^2 + 1 \\
2\ N^2 - 10\ N - 12 &= 0 \\ 
\end{aligned}
$$

therefore **base $N$ is 6**

---

# 2.

### (1)
| Decimal | Code |
|:-------:|:----:|
|    0    | 0000 |
|    1    | 0111 |
|    2    | 0110 |
|    3    | 0101 |
|    4    | 0100 |
|    5    | 1011 |
|    6    | 1010 |
|    7    | 1001 |
|    8    | 1000 |
|    9    | 1111 |
<br>

### (2)

$9 - d$ in  this code can be transform to :
`1111` - `xxxx`   (where x represent 0 or 1)

then the answer will be :
`(1-x)(1-x)(1-x)(1-x)`

( literally doing **Xor** at every digit )

---

# 3.


F' = (AB'C + (A' + B + D)(ABD' + B'))'
= (AB'C)'((A' + B + D)(ABD' + B'))'
= (A' + B + C')((A' + B + D)' + (ABD'+B')')
= (A' + B + C')(A + B' + D'+ B(ABD')')
= **(A' + B + C')(A + B' + D' + B(A' + B' + D))**

---

# 4.

### (1)
**D( C( A'+B' ) + AC' )**
<br>

### (2)
This expression is equivalent to DCA'+DCB'+DAC'
![[School/Course Homeworks/DSDL/assets/circuit1.png|475]]
<br>
<br>
<br>
<br>

### (3)
D( CA' + CB' + AC' )
= D( CA' + CB' + AC' + AB' + AA' + CC' )
= D(A + C)(A' + B' + C')
![[School/Course Homeworks/DSDL/assets/circuit2.png|475]]

---
<br>

# 5.

(A + B + C + D)(A' + B' + C + D')(A' + C)(A + D)(B + C + D)
= (A + B + C + D)(A' + B' + C + D')(A' + C)(A + D)~~(B + C + D)~~ $\,$ `consensus` 
= (A + B + C + D)(A' + B' + C + D')(A' + C)(A + D) $\,$ `absorption` 
= (A' + C)(A + D)
= A'A + A'D + AC + CD
= **AC + A'D + CD**
<br>

---
<br>

# 6.

BCD + C′D′ + B′C′D + CD
= (B+1)CD + C′(D′ + B′D)
= CD + C′(D′ + B′)
= CD + C'D' + B'C'
= CD + C'D' + B'C' + B'D + CC' + DD'
= **(C' + D)(B' + C + D')**
<br>

---

# 7.

|  A  |  B  |  C  | output |
|:---:|:---:|:---:|:------:|
|  0  |  0  |  0  |   0    |
|  0  |  0  |  1  |   0    |
|  0  |  1  |  0  |   0    |
|  0  |  1  |  1  |   1    |
|  1  |  0  |  0  |   0    |
|  1  |  0  |  1  |   1    |
|  1  |  1  |  0  |   1    |
|  1  |  1  |  1  |   1    |

A'BC + AB'C + ABC' + ABC
= BC + AC + AB + ABC
= **AB + AC + BC**