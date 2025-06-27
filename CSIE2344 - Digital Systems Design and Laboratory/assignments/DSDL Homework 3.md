B11902038 資工一 鄭博允

---

# 1 $\quad$ MUXes and Three-State Buffers
![[School/Course Homeworks/DSDL/assets/3-1-1.png|450]]
<br>

# 2 $\quad$ Latch I
![[School/Course Homeworks/DSDL/assets/3-2-1.png]]
<div style="page-break-after:always;"></div>

# 3 $\quad$ Latch II
### 1.
from latch we know : 
$P = (Q \cdot H)'$ , $Q = P' + R$

case $Q = 0$ :
$P = (0 \cdot H)' = 1$ for both $H=0,1$
$Q$ should remain $0$, so $P' + R = 0 + R = 0$ and thus $R = 0$

case $Q = 1$ :
we want $P = (1 \cdot H)' = 0$, therefore $H=1$
$Q = P' + R = 1 + R$ for both $R=0,1$

combine above conditions, we get $R = 0, H = 1$
<br>

### 2.
![[School/Course Homeworks/DSDL/assets/3-3-2.png|450]]
$\quad\ Q^+ = R + QH \quad\quad\quad\quad\quad\quad\quad\ P^+ = H' + PR'$
<div style="page-break-after:always;"></div>

# 4  $\quad$ Lab 1: Source Code
### 1.

```c
module rca_gl(
	output C3,       // carry output
	output[2:0] S,   // sum
	input[2:0] A, B, // operands
	input C0         // carry input
	);

	// TODO:: Implement gate-level RCA
	wire C1, C2; 
	FA fa0(C1, S[0], A[0], B[0], C0);
	FA fa1(C2, S[1], A[1], B[1], C1);
	FA fa2(C3, S[2], A[2], B[2], C2);
endmodule
```

### 2.

```c
module cla_gl(
	output C3,       // carry output
	output[2:0] S,   // sum
	input[2:0] A, B, // operands
	input C0         // carry input
	);

	// TODO:: Implement gate-level CLA
	wire[2:0] P, G;
	wire C1, C2;
	wire n;
	
	AND and0(G[0], A[0], B[0]);
	OR or0(P[0], A[0], B[0]);
	AND and1(G[1], A[1], B[1]);
	OR or1(P[1], A[1], B[1]);
	AND and2(G[2], A[2], B[2]);
	OR or2(P[2], A[2], B[2]);

	wire pc0;
	assign pc0 = P[0] & C0;
	assign C1 = G[0] | pc0;

	wire pc1, gp1;
	assign pc1 = pc0 & P[1];
	assign gp1 = G[0] & P[1];
	assign C2 = G[1] | pc1 | gp1;

	wire pc2, gp21, gp22;
	assign pc2 = pc1 & P[2];
	assign gp21 = gp1 & P[2];
	assign gp22 = G[1] & P[2];
	assign C3 = G[2] | gp22 | gp21 | pc2;

	FA fa0(n, S[0], A[0], B[0], C0);
	FA fa1(n, S[1], A[1], B[1], C1);
	FA fa2(n, S[2], A[2], B[2], C2);

endmodule
```
<div style="page-break-after:always;"></div>

# 5 $\quad$ Lab 1: Waveform
![[School/Course Homeworks/DSDL/assets/3-5-1.png|650]]
<br>

# 6 $\quad$ Lab 1: Propagation Delays
### 1.
![[School/Course Homeworks/DSDL/assets/3-6-1.png]]

### 2.
![[School/Course Homeworks/DSDL/assets/3-6-3.png]]
<br>

# 7 $\quad$ Lab 1: Some Derivation

The gate levels can be derive from the number of gate(s) passed in the longest path, which is from $C_0$ to $C_n$ for n-bit carry-lookahead adder.

For each $P_i$ and $G_i$ can be implement in 1 gate, and it's level is 1.

> $C_{i+1} = G_i + P_i \cdot C_i$

From $C_i$ to $C_{i+1}$ will passed 2 gates : 1 AND gate and 1 OR gate.

We can break down the path to many parts:
from $C_0$ to $C_1$ passed 2 gates
from $C_1$ to $C_2$ passed 2 gates
...
from $C_{n-1}$ to $C_n$ passed 2 gates

Therefore, we finally know that from $C_0$ to $C_n$ will passed $2n$ gates,
and the gate levels of n-bit carry-lookahead adder is $2n$.