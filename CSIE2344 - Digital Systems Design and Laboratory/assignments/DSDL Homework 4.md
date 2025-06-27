B11902038 資工一 鄭博允

---
# 1 $\quad$ Counter Design 

### 1.
![[School/Course Homeworks/DSDL/assets/4-1-1.png|450]]
![[School/Course Homeworks/DSDL/assets/4-1-2.png|575]]
<div style="page-break-after:always;"></div>

### 2.
![[School/Course Homeworks/DSDL/assets/4-1-3.png|175]]
$D_C=CA+BA'$
<br>

### 3.
![[School/Course Homeworks/DSDL/assets/4-1-4.png|175]]
$T_C=B'A'+C'A'$
<div style="page-break-after:always;"></div>

### 4.
We derived truth table of $S_B, R_B$ from the excitation table
![[School/Course Homeworks/DSDL/assets/4-1-5.png|475]]
$S_B = C'$ , $R_B = CA$
<div style="page-break-after:always;"></div>

### 5.
We derived truth table of $J_A, K_A$ from the excitation table
![[School/Course Homeworks/DSDL/assets/4-1-6.png|475]]
$J_A = C$ , $K_A = C'B+CB'$
<div style="page-break-after:always;"></div>

# 2 $\quad$ Construction of State Table and State Graph
### 1.
- Derive input & output equations
$J_1 = X$ , $K_1 = (X Q_2')'$
$J_2 = X$ , $K_2 = (X Q_1)'$
$Z = Q_2'\ ⊕\ X$
<br>

-  Derive the next-state equation of each flip-flop
$Q_1^+ = XQ_1' + X Q_2'Q_1$
$Q_2^+ = XQ_2' + X Q_1 Q_2$
<br>

- Plot a next-state map for each flip-flop
![[School/Course Homeworks/DSDL/assets/4-2-1.png|500]]
<br>

- form the state table
![[School/Course Homeworks/DSDL/assets/4-2-2.png|425]]
<div style="page-break-after:always;"></div>

### 2.
S0 = 00, S1 = 01, S2 = 10, S3 = 11

![[School/Course Homeworks/DSDL/assets/4-2-3.png|375]]
<div style="page-break-after:always;"></div>

# 3 $\quad$ Derivation of State Tables

### 1.

We derive the state graph of the circuit :

![[School/Course Homeworks/DSDL/assets/4-3-1.png|300]]

- **<mark style="background: #D2B3FFA6;">State S0 (reset)</mark>** : 
When the preceding three inputs are $000$ or $100$. 
Whether receive $0$ or $1$ will form a BCD digit. 

- **<mark style="background: #D2B3FFA6;">State S1</mark>** : 
If receive $0$ will form a BCD digit ; receive $1$ will **not** form a BCD digit. 
Go to S0 if the next two inputs are both $0$. 

- **<mark style="background: #D2B3FFA6;">State S2</mark>** : 
If receive $0$ will form a BCD digit ; receive $1$ will **not** form a BCD digit. 
Go to S0 if the next input is $0$, otherwise go back to S0 .
<br>

State Table : 
![[School/Course Homeworks/DSDL/assets/4-3-2.png|450]]
<br>
<div style="page-break-after:always;"></div>

### 2.


<div style="page-break-after:always;"></div>

# 4 $\quad$ Lab 2: Part 1

### 1. 
```java
module mult_fast(
	output reg[7:0] P,  // product
	input[3:0] A, B,    // multiplicand and multiplier
	input clk		    // clock (posedge)
	);
	// stage 0 (input)
	reg[3:0] a_s0, b_s0;
	always @(posedge clk) begin
		a_s0 <= A;
		b_s0 <= B;
	end
	// stage 1
	wire[3:0] pp0 = a_s0 & {4{b_s0[0]}}; // ignore the delays of AND gates
	wire[4:1] pp1 = a_s0 & {4{b_s0[1]}}; // ignore the delays of AND gates
	wire[5:2] pp2 = a_s0 & {4{b_s0[2]}}; // ignore the delays of AND gates
	wire[6:3] pp3 = a_s0 & {4{b_s0[3]}}; // ignore the delays of AND gates
	reg[5:1] sum1;
	always @(pp0, pp1)
		sum1[5:1] <= #7 pp0[3:1] + pp1[4:1]; // delay of the 4-bit adder
	reg[7:3] sum3;
	always @(pp2, pp3)
		sum3[7:3] <= #7 pp2[5:3] + pp3[6:3]; // delay of the 4-bit adder
	reg[5:0] sum1_s1;
	reg[7:2] sum3_s1;
	always @(posedge clk) begin
		sum1_s1 <= {sum1, pp0[0]};
		sum3_s1 <= {sum3, pp2[2]};
	end
	// stage 2 (outout)
	reg[7:2] sum2;
	always @(sum1_s1, sum3_s1)
		sum2[7:2] <= #8 sum1_s1[5:2] + sum3_s1[7:2]; // delay of the 6-bit adder
	always @(posedge clk) begin
		P <= {sum2, sum1_s1[1:0]};
	end
endmodule
```
<br>

### 2.
![[School/Course Homeworks/DSDL/assets/4-1.png]]
<br>

### 3.
The latency is **21** ticks.
<div style="page-break-after:always;"></div>

# 5 $\quad$ Lab 2: Part 2

### 1.
minimum clock cycle = **8** ticks

### 2.
![[School/Course Homeworks/DSDL/assets/4-2.png]]