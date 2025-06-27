b11902038 資工三 鄭博允

---

# 5.
First,
$$
\begin{aligned}
\sum_{k=1}^{K}({\bf w}^T\tilde{x}_k - \tilde{y}_k)^2 &= \sum_{k=1}^{K}({\bf w}^T\tilde{x}_k)^2\\
&= \sum_{k=1}^{K}(w_{k-1}\sqrt{\lambda\alpha_{k-1}})^2\\
&= \sum_{k=1}^{K}\lambda\cdot\alpha_{k-1}w_{k-1}^2
\end{aligned}
$$

if we use $i = k - 1$ to replace $k$ and $d+1$ to replace $K$, then we have
$$
\sum_{k=1}^{K}\lambda\cdot\alpha_{k-1}w_{k-1}^2 = \sum_{i=0}^{d}\lambda\cdot\alpha_{i}w_{i}^2
$$
Therefore $P_V$ becomes
$$\min {1\over N+d+1} \left(\sum_{n=1}^{N}({\bf w}^T{x}_n - {y}_n)^2+\lambda\sum_{i=0}^{d}\alpha_{i}w_{i}^2\right)$$

For $P_R$, we can also change it to
$$\min {1\over N} \left(\sum_{n=1}^{N}({\bf w}^T{x}_n - {y}_n)^2+\lambda\sum_{i=0}^{d}\alpha_{i}w_{i}^2\right)$$

Since the multiplier (${1\over N + d + 1}$ and $1 \over N$) doesn't affect the optimal solution, thus both solving $P_V$ and $P_R$ can be seen as solving
$$\min \left(\sum_{n=1}^{N}({\bf w}^T{x}_n - {y}_n)^2+\lambda\sum_{i=0}^{d}\alpha_{i}w_{i}^2\right)$$
and their optimal ${\bf w}^*$ is the same.

<div style="page-break-after:always;"></div>

# 6.

## substituting $\tilde{E}_{\text{in}}({\bf w})$ in $\tilde{E}_{\text{aug}}({\bf w})$: $$\tilde{E}_{\text{aug}}({\bf w}) = {E}_{\text{in}}({\bf w}^*) + {1\over2}({\bf w}-{\bf w}^*)^TH({\bf w}-{\bf w}^*) + {\lambda\over N}||{\bf w}||^2$$
## differentiate it with respect to ${\bf w}$
1. $$\nabla_{\bf w}{E}_{\text{in}}({\bf w}^*) = 0$$
2. $$\begin{aligned}
&\nabla_{\bf w}{1\over2}({\bf w}-{\bf w}^*)^TH({\bf w}-{\bf w}^*)\\
&=\nabla_{\bf w}({1\over2}{\bf w}^TH{\bf w}-{\bf w}^TH{\bf w}^*+{1\over2}({\bf w}^*)^TH{\bf w}^*)\\
&=H{\bf w} - H{\bf w}^* + 0
\end{aligned}$$
3. $$\nabla_{\bf w}{\lambda\over N}||{\bf w}||^2 = {2\lambda\over N}{\bf w}$$
then we have
$$\nabla_{\bf w}\tilde{E}_{\text{aug}}({\bf w}) = H({\bf w} - {\bf w}^*) + {2\lambda\over N}{\bf w}$$

## solve $\nabla_{\bf w}\tilde{E}_{\text{aug}}({\bf w}) = 0$
   $$\begin{aligned}
   H({\bf w} - {\bf w}^*) + {2\lambda\over N}{\bf w} &= 0\\
   H({\bf w} - {\bf w}^*) &= -{2\lambda\over N}{\bf w}\\
   H{\bf w} + {2\lambda\over N}{\bf w} &= H{\bf w}^*\\
   (H+{2\lambda\over N}I){\bf w} &= H{\bf w}^*\\
   {\bf w} &= (H-{2\lambda\over N}I)^{-1}H{\bf w}^*\\
   \end{aligned}$$

therefore we have the minimizer being
$${\bf w} = (H-{2\lambda\over N}I)^{-1}H{\bf w}^*$$

<div style="page-break-after:always;"></div>

# 7.
$$\begin{aligned}
\mathbb{E}\left({1\over K}\sum_{n=N=K+1}^N (y_n - \bar{y})^2\right)
&= {1\over K}\sum_{n=N=K+1}^N\mathbb{E}\left((y_n - \bar{y})^2\right)\\
&= {1\over K}\sum_{n=N=K+1}^N\mathbb{E}\left(y_n^2 - 2y_n\bar{y} + \bar{y}^2\right)\\
&= {1\over K}\sum_{n=N=K+1}^N\mathbb{E}(y_n^2) - 2\mathbb{E}(y_n\bar{y}) + \mathbb{E}(\bar{y}^2)\\
&= \mathbb{E}(y_n^2) - 2\mathbb{E}(y_n\bar{y}) + \mathbb{E}(\bar{y}^2)
\end{aligned}$$

## For $\mathbb{E}(y_n^2)$:
since $y_n$ generated i.i.d from $N(0, \sigma^2)$, we have
$$\mathbb{E}(y_n^2) = \sigma^2$$

## For $\mathbb{E}(y_n\bar{y})$:
$$\mathbb{E}(y_n\bar{y}) = \mathbb{E}(y_n)\cdot\mathbb{E}(\bar{y})$$
since $y_n$ generated i.i.d from $N(0, \sigma^2)$, we have
$$\mathbb{E}(y_n) = 0$$
and $\bar{y}$ is the mean of $y_1, \cdots, y_{N-K}$, which are also generated i.i.d from $N(0, \sigma^2)$, therefore we have
$$\mathbb{E}(\bar{y}) = 0$$
then we get
$$\mathbb{E}(y_n\bar{y}) = 0$$

## For $\mathbb{E}(\bar{y}^2)$:
$$\mathbb{E}(\bar{y}^2) = \text{Var}(\bar{y}) + \mathbb{E}(\bar{y})^2$$
first, we calculate the $\text{Var}(\bar{y})$:
$$\begin{aligned}
\text{Var}(\bar{y}) &= \text{Var}\left({1\over N-K}\sum_{n=1}^{N-K} y_n\right)\\
&= {1\over (N-K)^2}\sum_{n=1}^{N-K}\text{Var}\left(y_n\right)\\
&= {1\over (N-K)^2}\sum_{n=1}^{N-K}\sigma^2\\
&= {\sigma^2\over N-K}
\end{aligned}$$
then we have
$$\mathbb{E}(\bar{y}^2) = {\sigma^2\over N-K}$$

## Combine each term
Finally, we combine each term and get
$$\mathbb{E}\left({1\over K}\sum_{n=N=K+1}^N (y_n - \bar{y})^2\right) = \sigma^2 + 0 + {\sigma^2\over N-K} = (1 + {1 \over N-K})\sigma^2$$

<div style="page-break-after:always;"></div>

# 8.

<div style="page-break-after:always;"></div>

# 9.
By the test distribution, we have
$$E_{\text{out}}(g) = (1-p)\epsilon_{+} + p\epsilon_{-}$$
we want to find such $\epsilon_{+}$ and $\epsilon_{-}$ such that
$$E_{\text{out}}(g) = E_{\text{out}}(g_c) = p$$
therefore we have
$$\begin{aligned}
(1-p)\epsilon_{+} + p\epsilon_{-} &= p\\
\epsilon_{+} - p\epsilon_{+} + p\epsilon_{-} &= p\\
p(1+\epsilon_{+}-\epsilon_{-}) &= \epsilon_{+}\\
p &= {\epsilon_{+}\over1+\epsilon_{+}-\epsilon_{-}}
\end{aligned}$$

<div style="page-break-after:always;"></div>

# 10.
## Result
![|400](School/Course%20Homeworks/HTML/assets/problem10_eout_hist.png)
![|400](School/Course%20Homeworks/HTML/assets/problem10_non_zeros_hist.png)

<div style="page-break-after:always;"></div>

## Code Snapshot
![|625](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241118145342.png)

<div style="page-break-after:always;"></div>

# 11.
## Result & Findings
![|400](School/Course%20Homeworks/HTML/assets/problem11_eout_hist.png)
![|400](School/Course%20Homeworks/HTML/assets/10_11_eout_hist.png)

The differences in the algorithms for Problem 10 and Problem 11 are as follows:

|                 | problem10          | problem11           |
| --------------- | ------------------ | ------------------- |
| training set    | N=11876            | N=8000              |
| validation set  |                    | N=3876              |
| testing set     | N=1990             | N=1990              |
| choosing lambda | by $E_{\text{in}}$ | by $E_{\text{val}}$ |

Compared to the results of Problem 10, the results of Problem 11 appear slightly smaller and more concentrated. This indicates that using a validation set to select λ provides some improvement (while it's not significant).

However, the key difference lies in the training time. In each experiment iteration, Problem 10 requires training on a dataset of N=11,876 and recalculating $E_{\text{in}}$ on the same data. In contrast, Problem 11 uses a training set reduced by one-third, with only N=3,876 samples allocated for calculating $E_{\text{val}}$. The total data processed is almost halved compared to Problem 10.

Consequently, completing 1,126 experiment iterations took approximately 5 hours for Problem 10, while Problem 11 only required 3 hours. This demonstrates that splitting the data into training and validation sets can reduce training time without substantially impacting $E_{\text{out}}$.

<div style="page-break-after:always;"></div>

## Code Snapshot
![|625](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241118145419.png)

<div style="page-break-after:always;"></div>

# 12.
## Result & Findings
![|400](School/Course%20Homeworks/HTML/assets/problem12_eout_hist.png)
![|400](School/Course%20Homeworks/HTML/assets/11_12_eout_hist.png)

The differences in the algorithms for Problem 11 and Problem 12 are as follows:

|                 | problem11           | problem12          |
| --------------- | ------------------- | ------------------ |
| training set    | N=8000              | N=7918             |
| validation set  | N=3876              | N=3958             |
| testing set     | N=1990              | N=1990             |
| choosing lambda | by $E_{\text{val}}$ | by $E_{\text{cv}}$ |

Although the sizes of the training and validation sets are nearly identical in Problems 11 and 12, the algorithm in Problem 12 calculates $E_{\text{cv}}$ by treating all three partitions as the validation set, selecting the λ corresponding to the smallest $E_{\text{val}}$. This approach theoretically increases the likelihood of choosing a λ that minimizes the validation error.

However, the results show that the additional effort in Problem 12 did not improve $E_{\text{out}}$ as expected. In fact, the $E_{\text{out}}$ for Problem 12 was even higher than that for Problem 11.

<div style="page-break-after:always;"></div>

## Code Snapshot
![|625](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241118145504.png)

<div style="page-break-after:always;"></div>

# 13.