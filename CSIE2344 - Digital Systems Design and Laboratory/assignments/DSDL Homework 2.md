B11902038 資工一 鄭博允

---

# 1.

**Truth Table**

| No. | A   | B   | C   | X   | Y   |
|:--- |:--- |:--- |:--- |:--- |:--- |
| 0   | 0   | 0   | 0   | 0   | 0   |
| 1   | 0   | 0   | 1   | 0   | 1   |
| 2   | 0   | 1   | 0   | 0   | 1   |
| 3   | 0   | 1   | 1   | 1   | 0   |
| 4   | 1   | 0   | 0   | 0   | 1   |
| 5   | 1   | 0   | 1   | 1   | 0   |
| 6   | 1   | 1   | 0   | 1   | 0   |
| 7   | 1   | 1   | 1   | 1   | 1   |

To find minterm we counting 1's 
and maxterm we counting 0's

$$
\begin{aligned}
X &= m_3 + m_5 + m_6 + m_7 = \sum{m(3,5,6,7)} \\\\
&=M_0M_1M_2M_4 = \prod{M(0,1,2,4)}
\end{aligned}
$$

$$
\begin{aligned}
Y &= m_1 + m_2 + m_4 + m_7 = \sum{m(1,2,4,7)} \\\\
&=M_0M_3M_5M_6 = \prod{M(0,3,5,6)}
\end{aligned}
$$
<br>

# 2.
## (1)
| ABCD | number | S   | T   | U   | V   | W   | X   | Y   | Z   |
|:---- |:------ | --- | --- | --- | --- | --- | --- | --- | --- |
| 0000 | 0      | 0   | 0   | 0   | 0   | 0   | 0   | 0   | 0   |
| 0001 | 1      | 0   | 0   | 0   | 0   | 0   | 1   | 0   | 1   |
| 0010 | 2      | 0   | 0   | 0   | 1   | 0   | 0   | 0   | 0   |
| 0011 | 3      | 0   | 0   | 0   | 1   | 0   | 1   | 0   | 1   |
| 0100 | 4      | 0   | 0   | 1   | 0   | 0   | 0   | 0   | 0   |
| 0101 | 5      | 0   | 0   | 1   | 0   | 0   | 1   | 0   | 1   |
| 0110 | 6      | 0   | 0   | 1   | 1   | 0   | 0   | 0   | 0   |
| 0111 | 7      | 0   | 0   | 1   | 1   | 0   | 1   | 0   | 1   |
| 1000 | 8      | 0   | 1   | 0   | 0   | 0   | 0   | 0   | 0   |
| 1001 | 9      | 0   | 1   | 0   | 0   | 0   | 1   | 0   | 1   |
| 1010 | 10     | 0   | 1   | 0   | 1   | 0   | 0   | 0   | 0   |
| 1011 | 11     | 0   | 1   | 0   | 1   | 0   | 1   | 0   | 1   |
| 1100 | 12     | 0   | 1   | 1   | 0   | 0   | 0   | 0   | 0   |
| 1101 | 13     | 0   | 1   | 1   | 0   | 0   | 1   | 0   | 1   |
| 1110 | 14     | 0   | 1   | 1   | 1   | 0   | 0   | 0   | 0   |
| 1111 | 15     | 0   | 1   | 1   | 1   | 0   | 1   | 0   | 1   |

## (2)

$S=0$ , $T=A$ , $U=B$ , $V=C$

$W=0$ , $X=D$ , $Y=0$ , $Z=D$
<br>

# 3.
![[School/Course Homeworks/DSDL/assets/hw2p3.png|400]]
$F=A'B' + AB + C'$
<br>

# 4.
## (1)
![[School/Course Homeworks/DSDL/assets/hw2p4-1.png|475]]
$F = A'B + AB'C' + C'D + A'D$


## (2)
![[School/Course Homeworks/DSDL/assets/hw2p4-2.png|475]]

$F' = AC + A'B'D' + ABD'$
By DeMorgan's law :
$F = (A'+C')(ABD)(A'B'D)$
<br>
<br>

# 5.
![[School/Course Homeworks/DSDL/assets/hw2p5.png|400]]
![[School/Course Homeworks/DSDL/assets/hw2p5-2.png|300]]

$F = B'D' + C'D + A'B' + BD$
<br>

# 6.
![[School/Course Homeworks/DSDL/assets/6.png|250]]

Draw $F$ as K-MAP , find all PIs of 1's and 0's 
and thus find out the 
minimum SOP $F = A'BC' + AB'C' + BC'D$
minimum POS $F = C'(A+B)(A'+B'+D)$
therefore these circuits are minimum.
![[School/Course Homeworks/DSDL/assets/sop.png]]

![[School/Course Homeworks/DSDL/assets/pos.png]]
<br>

# 7.

To minimize the cost , we need to minimize number of terms and literals.
![[School/Course Homeworks/DSDL/assets/7.png|300]]
Use K-Map to find minimum POS :
$F = AB' + A'B + BC'D' + A'CD'$ 
$\quad= AB' + A'B + (BC' + A'C)D'$  (not minimum)

find minimum SOP :
$F' = A'B'C' + A'B'D + ABD + ABC$
$F=(A+B+C)(A+B+D')(A'+B'+D')(A'+B'+C')$
$\quad=(A+B+CD)(A'+B'+C'D')$

![[School/Course Homeworks/DSDL/assets/7-2.png]]

**5 gates ; 12 gate inputs**