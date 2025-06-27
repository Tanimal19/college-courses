b11902038 資工三 鄭博允

---

# 5.
From
$$E_{\text{in}}({\bf w}) = {1 \over N}\sum\ln(1+\exp(-y_n {\bf w}^T {\bf x}_n))$$
we derive it's first derivative:
$$\begin{aligned}
{\partial \over \partial {\bf w}_j}E_{\text{in}}({\bf w}) &= {1 \over N}\sum -y_n {\bf x}_{n,j} \cdot {\exp(-y_n {\bf w}^T {\bf x}_n) \over 1+\exp(-y_n {\bf w}^T {\bf x}_n)}\\
&= {1 \over N}\sum -y_n {\bf x}_{n,j} \cdot (1-\theta(y_n {\bf w}^T {\bf x}_n))
\end{aligned}$$
and then the second derivative:
$$\begin{aligned}
{\partial^2 \over \partial{\bf w}_j\partial{\bf w}_k}E_{\text{in}}({\bf w}) &= {1 \over N}\sum y_n^2 {\bf x}_{n,j} {\bf x}_{n,k} \cdot {\theta(y_n {\bf w}^T {\bf x}_n) \over 1-\theta(y_n {\bf w}^T {\bf x}_n)}\\
&= {1 \over N}\sum {\bf x}_{n,j} {\bf x}_{n,k} \cdot {h_t({\bf x}_n) \over 1-h_t({\bf x}_n)}
\end{aligned}$$
since $y_n$ always be $1$ or $-1$.

Therefore, each element $(j, k)$ in $A_E({\bf w}_t)$ is
$$A_E({\bf w}_t)_{j,k} = {1 \over N}\sum {\bf x}_{n,j} {\bf x}_{n,k} \cdot {h_t({\bf x}n) \over 1-h_t({\bf x}_n)}$$

By $A_E({\bf w}) = X^TDX$, we have each diagonal element in $D$ being
$$D_{n, n} = h_t({\bf x}_n)(1 - h_t({\bf x}_n))$$
while other elements being $0$.

That is,
$$D = \begin{bmatrix}
&h_t({\bf x}_1)(1 - h_t({\bf x}_1)) &0 &\cdots &0\\
&0 &h_t({\bf x}_2)(1 - h_t({\bf x}_2)) &\cdots &0\\
&\vdots &\vdots &\ddots &\vdots\\
&0 &0 &\cdots &h_t({\bf x}_d)(1 - h_t({\bf x}_d))
\end{bmatrix}$$

<div style="page-break-after:always;"></div>

# 6.

<div style="page-break-after:always;"></div>

# 7.

<div style="page-break-after:always;"></div>

# 8.
Since we have only two point in the training set, and $h(x)$ is a line, therefore, we can certainly find $g(x)$ which passed $(x_1, f(x_1))$ and $(x_2, f(x_2))$.

We can solve below equations to find $w^*_0, w^*_1$:
$$\begin{aligned}
w^*_0 + w^*_1 x_1 = 1 - 2 x_1^2\\
w^*_0 + w^*_1 x_2 = 1 - 2 x_2^2
\end{aligned}$$
and get
$$\begin{aligned}
w^*_1 &= 2\cdot{(x_2^2 - x^2_1)\over x_1-x_2}\\
w^*_0 &= 1 - 2x_1^2 - w^*_1x_1
\end{aligned}$$

Since $g(x)$ passed $(x_1, f(x_1))$ and $(x_2, f(x_2))$, we have $E_{\text{in}}(g) = 0$.
Therefore, the expected value will be $\mathbb{E}_{\mathcal{D}}(|E_{\text{in}}(g) - E_{\text{out}}(g)|) = \mathbb{E}_{\mathcal{D}}(E_{\text{out}}(g))$.

Unfortunately, I don't know how to calculate $E_{\text{out}}$.

<div style="page-break-after:always;"></div>

# 9.

<div style="page-break-after:always;"></div>

# 10.
## Result & Findings
![|425](School/Course%20Homeworks/HTML/assets/hw4-p10-plot.png)

It looks like the $E_{\text{in}}({\bf w}_{\text{LIN}})$ and $E_{\text{out}}({\bf w}_{\text{LIN}})$ are the lower bound and upper bound of square errors.
For SGD, $E_{\text{in}}({\bf w}_{\text{t}})$ seems to decrease with each iteration, but the rate of decrease becomes more gradual as the iterations increase. On the other hand, $E_{\text{out}}({\bf w}_{\text{t}})$ initially decreases rapidly but suddenly rebounds at a certain point, which aligns closely with the point where the $E_{\text{in}}({\bf w}_{\text{t}})$ curve starts to flatten out. Therefore, it can be inferred that there exists a threshold in the iterations of SGD that allows $E_{\text{out}}({\bf w}_{\text{t}})$ to reach it's minimum points while also make $E_{\text{in}}({\bf w}_{\text{t}})$ relatively small. However, once this threshold is exceeded, $E_{\text{out}}({\bf w}_{\text{t}})$ starts to increase again.

## Code Snapshot
![|325](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241031215349.png)

<div style="page-break-after:always;"></div>

# 11.
## Result & Findings
![|425](School/Course%20Homeworks/HTML/assets/hw4-p11-plot%201.png)

The average $E_{\text{in}}$ difference ($E_{\text{in}}({\bf w}_{\text{LIN}}) - E_{\text{in}}({\bf w}_{\text{poly}})$) is 33.48223579870386.
In general, $E_{\text{in}}$ is decrease when using polynomial transform rather than original data.

## Code Snapshot
![|325](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241031215422.png)

<div style="page-break-after:always;"></div>

# 12.
## Result & Findings
![|425](School/Course%20Homeworks/HTML/assets/hw4-p12-plot%201.png)

The average $E_{\text{out}}$ difference ($E_{\text{out}}({\bf w}_{\text{LIN}}) - E_{\text{out}}({\bf w}_{\text{poly}})$) is -2850004899014561.5.
In general, $E_{\text{out}}$ is significantly increase when using polynomial transform rather than original data. 

## Code Snapshot
(it's the same code of problem 11)
![|375](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241031215506.png)

<div style="page-break-after:always;"></div>

# 13.