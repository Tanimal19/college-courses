b11902038 資工三 鄭博允

---
# 5.

(1). By definition we have $$\tilde{\bf w}^* = \sum_{n=1}^N\alpha^*_ny_n{\bf z}_n \quad\text{and}\quad\tilde{\bf w}^*_0 = \sum_{n=1}^N\alpha^*_ny_n{\bf z}_0$$
Since ${\bf z}^T_n = [1, {\bf x}^T_n]$ and the constraint $$\sum_{n=1}^N\alpha_ny_n = 0$$
we have $\tilde{\bf w}^*_0 = 0$.

(2). The optimal solution $(b^*, \tilde{\bf w}^*, \alpha^*)$ satisfied
$$\begin{aligned}
&\min_{{\bf w}, b, \xi} &{1\over2}{\tilde{\bf w}^*}^T\tilde{\bf w}^{*} + C\sum_{n=1}^N\xi_n\\
&\text{subject to} &y_n({\tilde{\bf w}^*}^T{\bf z}_n + b^*) \ge 1 - \xi_n
\end{aligned}$$
which is equivalent to
$$\begin{aligned}
&\min_{{\bf w}, b, \xi} &{1\over2}{{\bf w}^*}^T{\bf w}^{*} + {1\over2}{\tilde{\bf w}^*_0}^2 + C\sum_{n=1}^N\xi_n\\
&\text{subject to} &y_n({{\bf w}^*}^T{\bf x}_n + {\tilde{\bf w}^*_0} + b^*) \ge 1 - \xi_n
\end{aligned}$$
which is equivalent to
$$\begin{aligned}
&\min_{{\bf w}, b, \xi} &{1\over2}{{\bf w}^*}^T{\bf w}^{*} + C\sum_{n=1}^N\xi_n\\
&\text{subject to} &y_n({{\bf w}^*}^T{\bf x}_n + b^*) \ge 1 - \xi_n
\end{aligned}$$
since $\tilde{\bf w}^*_0 = 0$.

Thus, $(b^*, {\bf w}^*, \alpha^*)$ is also an optimal solution of the original problem.

<div style="page-break-after:always;"></div>

# 6.
To derive the dual formulation of $P$:
(1). we include the constraint into the primal problem like this:

$$\begin{aligned}
L(b, {\bf w}, \xi, \alpha, \beta) = &{1\over2}{\bf w}^T{\bf w} + C\sum^{N}_{n=1}\xi_n\\
&+ \sum^{N}_{n=1}\alpha_n\left(1 - \xi_n - y_n({\bf w}^T\Phi({\bf x}_n)+b)\right)\\
&+ \sum^{N}_{n=1} \beta_n(-\xi_n)\\
&+ \gamma(1 - y_0({\bf w}^T\Phi({\bf x}_0)+b))
\end{aligned}$$
where $\alpha_n, \beta_n, \gamma_n \ge 0$ are the multipliers for each constraints.

(2). then we optimize $L$ over primal variables $b, {\bf w}, \xi_n$ by taking differentiation and set to 0:
- (a)$${\partial L \over \partial {\bf w}} = {\bf w}-\sum^{N}_{n=1}\alpha_n y_n \Phi({\bf x}_n)-\gamma y_0\Phi({\bf x}_0) = 0 \implies {\bf w} =\sum^{N}_{n=1}\alpha_n y_n \Phi({\bf x}_n)-\gamma y_0\Phi({\bf x}_0)$$
- (b)$${\partial L \over \partial b} = -\sum^{N}_{n=1}\alpha_n y_n -\gamma y_0 = 0 \implies \sum^{N}_{n=1}\alpha_n y_n +\gamma y_0 = 0$$
- (c)$${\partial L \over \partial \xi_n} = C-\alpha_n-\beta_n = 0 \implies \alpha_n = C - \beta_n$$
  since we have $\beta_n \ge 0$, we can change it into a constraint $0 \le \alpha_n \le C$ and removed $\beta_n$.

(3). then we can simplify $L$:
- first we re-order the terms in $L$ into $$\begin{aligned}
L(b, {\bf w}, \xi, \alpha, \beta) = &{1\over2}{\bf w}^T{\bf w} + \sum^{N}_{n=1}\alpha_n + \gamma\\
&- {\bf w}^T \left(\sum^{N}_{n=1}\alpha_n y_n \Phi({\bf x}_n) - \gamma y_0 \Phi({\bf x}_0)\right)\\
&- \left(\sum^{N}_{n=1}\alpha_n y_n + \gamma y_0\right)\cdot b\\
&+ \sum^{N}_{n=1} (C - \alpha_n - \beta_n)\cdot\xi_n
\end{aligned}$$
- by (a) we can substitute the term $${\bf w}^T \left(\sum^{N}_{n=1}\alpha_n y_n \Phi({\bf x}_n) - \gamma y_0 \Phi({\bf x}_0)\right)$$
  into $${\bf w}^T{\bf w}$$
- by (b) we can remove the term $$\left(\sum^{N}_{n=1}\alpha_n y_n + \gamma y_0\right)\cdot b$$
- by (c) we can remove the term $$\sum^{N}_{n=1} (C - \alpha_n - \beta_n)\cdot\xi_n$$
therefore our $L$ is now:
$$L(b, {\bf w}, \xi, \alpha, \beta) = -{1\over2}{\bf w}^T{\bf w} + \sum^{N}_{n=1}\alpha_n + \gamma$$


$$\sum^{N}_{n=1}\alpha_n y_n \Phi({\bf x}_n)-\gamma y_0\Phi({\bf x}_0)$$
$$\begin{aligned}
L(b, {\bf w}, \xi, \alpha, \beta) = 
&-{1\over2} \cdot \left(
||\sum^{N}_{n=1}\alpha_n y_n{\bf z}_n||^2 + 2 \gamma \sum^{N}_{n=1} \alpha_n y_n y_0 {\bf z}_n^T{\bf z}_0 + ||\gamma y_0{\bf z}_0||^2
\right)\\
&+ \sum^{N}_{n=1}\alpha_n + \gamma\\
= &-{1\over2}||\sum^{N}_{n=1}\alpha_n y_n{\bf z}_n||^2 -\gamma \sum^{N}_{n=1} \alpha_n y_n y_0 {\bf z}_n^T{\bf z}_0 - {1\over2}||\gamma y_0{\bf z}_0||^2\\
&+ \sum^{N}_{n=1}\alpha_n + \gamma\\

\end{aligned}$$


(4). then we can substitute ${\bf w}$ again by (a), and get the Lagrange dual being:
$$\max -{1\over2}||\sum^{N}_{n=1}\alpha_n y_n{\bf z}_n||^2 -\gamma \sum^{N}_{n=1} \alpha_n y_n y_0 {\bf z}_n^T{\bf z}_0 - {1\over2}||\gamma y_0{\bf z}_0||^2 + \sum^{N}_{n=1}\alpha_n + \gamma$$
then change it to standard dual:
$$\begin{aligned}
\min_{\alpha, \gamma}\quad
&{1\over2}||\sum^{N}_{n=1}\alpha_n y_n{\bf z}_n||^2 +\gamma \sum^{N}_{n=1} \alpha_n y_n y_0 {\bf z}_n^T{\bf z}_0 + {1\over2}||\gamma y_0{\bf z}_0||^2 - \left(\sum^{N}_{n=1}\alpha_n + \gamma\right)
\\
\text{subject to}\quad
&\sum^{N}_{n=1}\alpha_n y_n +\gamma y_0 = 0
\\
&0 \le \alpha_n \le C, \text{for}\ n=1, 2, \cdots, N
\\
\text{implicitly}\quad
&{\bf w} =\sum^{N}_{n=1}\alpha_n y_n \Phi({\bf x}_n)-\gamma y_0\Phi({\bf x}_0)
\\
&\beta_n = C - \alpha_n
\end{aligned}$$

(5). finally, we express the dual in standard convex quadratic programming form:
- combining $\alpha_n$ and $\gamma$ into new $\alpha'$ vector where $$\alpha' = \begin{bmatrix}\alpha_1\\\alpha_2\\\vdots\\\alpha_N\\\gamma\end{bmatrix}$$
- change $${1\over2}||\sum^{N}_{n=1}\alpha_n y_n{\bf z}_n||^2 +\gamma \sum^{N}_{n=1} \alpha_n y_n y_0 {\bf z}_n^T{\bf z}_0 + {1\over2}||\gamma y_0{\bf z}_0||^2 $$
  into
  $${1\over2}\alpha'^TQ\alpha'$$
  where
  $$Q = \begin{bmatrix}
  H &G\\
  G^T &{\bf z}_0^T{\bf z}_0
  \end{bmatrix}$$
  and $H$ is an $N \times N$ matrix where $H_{nm} = y_ny_m{\bf z}_n^T{\bf z}_m$,
  $G$ is an $N \times 1$ matrix where $G_n = -y_ny_0{\bf z}_n^T{\bf z}_0$.
- change $$\sum^{N}_{n=1}\alpha_n + \gamma$$
  into $$-{\bf p}^T\alpha'$$
  where $${\bf p} = -1_{N+1}$$
- the constraint $$\sum^{N}_{n=1}\alpha_n y_n +\gamma y_0 = 0$$
  become $$A\alpha' = 0$$
  where $$A = \begin{bmatrix}y_1 &y_2 &\cdots &y_N &y_0\end{bmatrix}$$
- the constraint $$0 \le \alpha_n \le C$$
  become $$0 \le \alpha' \le c$$
  where $$c = \begin{bmatrix}C\\C\\\vdots\\C\\\infty\end{bmatrix}$$
therefore we have the final dual as:
$$\begin{aligned}
\min_{\alpha'}\quad
&{1\over2}\alpha'^TQ\alpha' + {\bf p}^T\alpha'
\\
\text{subject to}\quad
&A\alpha' = 0
\\
&0 \le \alpha' \le c
\end{aligned}$$

<div style="page-break-after:always;"></div>

# 7.
(1). first, we have
$$\begin{aligned}
\gamma &\gt {\ln(N-1) \over \epsilon^2}\\
\gamma\cdot\epsilon^2 &\gt \ln(N-1)\\
\gamma\cdot||{\bf x}_n - {\bf x}_k||^2 &\gt \ln(N-1)\\
-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2 &\lt -\ln(N-1)\\
\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2) &\lt {1 \over N-1}\\
(N-1)\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2) &\lt 1
\end{aligned}$$
for any $n, k = 1, \cdots, N$ and $n \neq k$.

(2). then we expand Ein
$$\begin{aligned}
E_{\text{in}}(\hat{h}) &= \sum_{k=1}^N y_k - \hat{h}({\bf x}_k)\\
&= \sum_{k=1}^N y_k - \text{sign}\left(\sum_{n=1}^N y_n\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2)\right)\\
&= \sum_{k=1}^N y_k - \text{sign}\left(y_k\cdot\exp(-\gamma\cdot||{\bf x}_k - {\bf x}_k||^2) +\sum_{n=1,n\neq k}^N y_n\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2)\right)\\
&= \sum_{k=1}^N y_k - \text{sign}\left(y_k +\sum_{n=1,n\neq k}^N y_n\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2)\right)
\end{aligned}$$


(3). by (1) we have
$$-1 \lt \sum_{n=1,n\neq k}^N y_n\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2) \lt 1$$
since $y_n = 1, -1$.

Because $|y_k| = 1$, thus
$$\text{sign}\left(y_k +\sum_{n=1,n\neq k}^N y_n\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2)\right) = y_k$$
for any $k = 1, \cdots, N$.

(4). combine (2) and (3) we can derive
$$\begin{aligned}
E_{\text{in}}(\hat{h}) &= \sum_{k=1}^N y_k - \text{sign}\left(y_k +\sum_{n=1,n\neq k}^N y_n\cdot\exp(-\gamma\cdot||{\bf x}_n - {\bf x}_k||^2)\right)\\
&= \sum_{k=1}^N y_k - y_k\\
&= 0
\end{aligned}$$

<div style="page-break-after:always;"></div>

# 8.

To prove $K(x, x') = \exp(2 \cos(x-x') - 2)$ is a valid kernel, we can first prove $K'(x, x') = \cos(x, x')$ is a valid kernel. If $K'$ is a valid kernel, then $K$ is also a valid kernel, since both exponentiation and linear transformation maintain symmetry and positive semi-definite propertys.

## Proving $K'(x, x') = \cos(x, x')$ is a valid kernel
(1). Represents special similarity
If we decompose $K'$ to
$$\cos(x-x') = \cos x \cos x' + \sin x \sin x'$$
which is actually the dot product of $(\cos x, \sin x)$ and $(\cos x', \sin x')$.
Therefore $K'$ can be represent as
$$K'(x, x') = \Phi(x)^T\Phi(x')$$
where $\Phi(x) = (\cos x, \sin x)$.

(2). Symmetry
$$\cos(x-x') = \cos x \cos x' + \sin x \sin x' = \cos x' \cos x + \sin x' \sin x = \cos(x' - x)$$

(3). Positive Semi-Definite
For any symmetric matrix $K$, if it's quadratic form
$$z^TKz = \sum_{i=1}^N\sum_{j=1}^Nz_iz_jK_{ij} \ge 0\quad \text{for all possible}\ z$$
then $K$ is positive semi-definite.

Now consider a matrix $K'$ where $K'_{ij} = K'(x, x')$, and it''s quadratic form
$$\begin{aligned}
z^TK'z &= \sum_{i=1}^N\sum_{j=1}^Nz_iz_jK'(x_i, x_j)\\
&= \sum_{i=1}^N\sum_{j=1}^Nz_iz_j(\cos x_i \cos x_j + \sin x_i \sin x_j)\\
&= \left(\sum_{i=1}^Nz_i\cos x_i\right)^2 + \left(\sum_{i=1}^Nz_i\sin x_i\right)^2
\end{aligned}$$
which implies $z^TK'z \ge 0$ for all possible $z$. Thus $K'$ is positive semi-definite.

Therefore, $K'$ is a valid kernel and $K$ is also a valid kernel.

<div style="page-break-after:always;"></div>

# 9.

<div style="page-break-after:always;"></div>

# 10.
## Result & Findings

| $C$ | $Q$ | \#SV  |
| --- | --- | ----- |
| 0.1 | 2   | 505  |
| 0.1 | 3   | 547 |
| 0.1 | 4   | 575 |
| 1   | 2   | 505  |
| 1   | 3   | 547  |
| 1   | 4   | 575 |
| 10  | 2   | 505  |
| 10  | 3   | 547  |
| 10  | 4   | 575  |

For $C$, the regularization parameter, that is how much do we care about margin violation:
- however, $C$ seems no effect on the number of SV

For $Q$, the degree of the polynomial kernel:
- Small $Q$ has smaller \#SV, large $Q$ has larger \#SV.
- When $Q$ is small, which means the model is less complex, so the boundary might be simple and can be "describe" by less SV.
- When $Q$ is large, which means the model can be much more complex, and we need more SV to describe the boundary.

<div style="page-break-after:always;"></div>

## Code Snapshot
![|500](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241129223438.png)

<div style="page-break-after:always;"></div>

# 11.
## Result & Findings

| $C$ | $\gamma$ | margin (round down) |
| --- | -------- | ------------------- |
| 0.1 | 0.1      | 0.0022761           |
| 0.1 | 1        | 0.0018938           |
| 0.1 | 10       | 0.0018940           |
| 1   | 0.1      | 0.0005171           |
| 1   | 1        | 0.0001893           |
| 1   | 10       | 0.0001894           |
| 10  | 0.1      | 0.0004995           |
| 10  | 1        | 0.0001874           |
| 10  | 10       | 0.0001873           |

For $C$, the regularization parameter, that is how much do we care about margin violation:
- Small $C$ has larger margin, large $C$ has smaller margin.
- When $C$ is small, which means we don't care about margin violation much, so the model will try to maximize the margin.
- When $C$ is large, which means we **do** care about margin violation, so the margin might be smaller to minimize classification errors (margin violations).

For $\gamma$, the gaussian kernel parameter in $$K({\bf x}_n, {\bf x}_m) = \exp(-\gamma||{\bf x}_n - {\bf x}_m||^2)$$
- Small $\gamma$ has larger margin, large $\gamma$ has smaller margin.
- When $\gamma$ is small, which means the distance has less influence on the similarity of points, even the distance between points are far, they might still considered similar. As result, the boundary become smoother and wider, which make it more general, but might be underfitting.
- When $\gamma$ is large, the boundary become sharper and narrower, which make it sensitive the data points, might overfitting.

For the combination of $C$ and $\gamma$:
- If both are small, the margin might be too big, the model can't seperate the data well, underfitting
- If both are large, the margin might be too small, the model might be too sensitve, overfitting

<div style="page-break-after:always;"></div>

## Code Snapshot
![|450](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241129223500.png)

<div style="page-break-after:always;"></div>

# 12.
## Result & Findings
![C:\Users\bobch\Documents\VScode\HTML\hw6\p12.png|375](file:///c%3A/Users/bobch/Documents/VScode/HTML/hw6/p12.png)

It seems like the best $\gamma$ is always 0.01, this indicates that small $\gamma$ can have better performance on validation data, which means the model is more general rather than being overfitting.

## Code Snapshot
![|400](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241129223555.png)

<div style="page-break-after:always;"></div>

# 13.