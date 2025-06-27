b11902038 資工三 鄭博允

---

# 5.

Assume
$\mathcal{H}_1$ has only one hypothesis $h_1: h_1({\bf x}) = 1$ and
$\mathcal{H}_2$ also has only one hypothesis $h_2: h_2({\bf x}) = -1$.

It's obvious that $d_{\text{vc}}(\mathcal{H}_1) = d_{\text{vc}}(\mathcal{H}_2) = 0$,
since both of them can't shatter input with $N=1$.

However, their set-union $\mathcal{H}_1 \cup \mathcal{H}_2$ can shatter input with $N = 1$,
since we have $h_1({\bf x}_1) = 1$ and $h_2({\bf x}_1) = -1$.

Therefore
$$d_{\text{vc}}(\mathcal{H}_1 \cup \mathcal{H}_2) \ge 1 \gt d_{\text{vc}}(\mathcal{H}_1) + d_{\text{vc}}(\mathcal{H}_2) = 0$$
and thus disprove the problem's statement.

<div style="page-break-after:always;"></div>

# 6.

Since false negative is 10 times more important than false positive, to balance between them (that is, we want the probability of false negative to be 10 times less than false positive), we derive:

```math
\begin{aligned}
P(\text{false negative}) &= {1 \over 10}P(\text{false positive})\\\\
P(y=+1|x) &= {1 \over 10}P(y=-1|x)\\\\
10\cdot P(y=+1|x) &= 1 - P(y=+1|x)\\\\
P(y=+1|x) &= {1\over11}
\end{aligned}
```

This value also represent the "threshold" of deciding $y=+1$ or $y=-1$, which is namely $\alpha$.

Therefore we set $\alpha = 1$.

<div style="page-break-after:always;"></div>

# 7.

To simplify notation, we use $A$ to represent ${\bf x}\sim P({\bf x})$, and $B$ to represent $y\sim P(y|{\bf x})$.

First, we derive

$$
\begin{aligned}
&E^{(2)}(h)\\
&= \mathbb{E}_{A,B}\left[h(x) \ne f(x)\ \text{and}\ f(x) = y\right] + \mathbb{E}_{A,B}\left[h(x) = f(x)\ \text{and}\ f(x) \ne y\right]\\
&= \mathbb{E}_{A}[h(x) \ne f(x)]\cdot\mathbb{E}_{A,B}[f(x) = y] + \mathbb{E}_{A}[h(x) = f(x)]\cdot\mathbb{E}_{A,B}[f(x) \ne y]\\
&= E^{(1)}(h)\cdot\mathbb{E}_{A,B}[f(x) = y] + \mathbb{E}_{A}[h(x) = f(x)]\cdot E^{(2)}(f)\\
&= E^{(1)}(h)\cdot(1-E^{(2)}(f)) + (1-E^{(1)}(h))\cdot E^{(2)}(f)
\end{aligned}
$$

Consider there's no error, we have $E^{(2)}(f) = 0$, and thus
$$E^{(2)}(h) = E^{(1)}(h) + 0 = E^{(1)}(h) + E^{(2)}(f)$$

Consider there's some error, we have

$$
\begin{aligned}
E^{(2)}(h) &= E^{(1)}(h) + E^{(2)}(f) - 2\cdot E^{(1)}(h)\cdot E^{(2)}(f)\\
&\lt E^{(1)}(h) + E^{(2)}(f)
\end{aligned}
$$

since both $E^{(1)}(h) \gt 0$ and $E^{(2)}(f) \gt 0$.

Therefore, we prove that
$$E^{(2)}(h) \le E^{(1)}(h) + E^{(2)}(f)$$

<div style="page-break-after:always;"></div>

# 8.

From the lecture we know that ${\bf w}_{\text{LIN}} = ({\bf X}^T{\bf X})^{-1}{\bf X}^T{\bf y}$ where

$$
{\bf X} = \begin{bmatrix}
1 &x_{1,1} &\cdots &x_{1,d}\\
1 &x_{2,1} &\cdots &x_{2,d}\\
\vdots\ &\vdots &\ddots &\vdots\\
1 &x_{N,1} &\cdots &x_{N,d}\\
\end{bmatrix},
{\bf y} = \begin{bmatrix}y_1\\y_2\\\vdots\\y_N\\\end{bmatrix}
$$

Let the new input matrix be

$$
{\bf X}'  = \begin{bmatrix}
1126 &x_{1,1} &\cdots &x_{1,d}\\
1126 &x_{2,1} &\cdots &x_{2,d}\\
\vdots\ &\vdots &\ddots &\vdots\\
1126 &x_{N,1} &\cdots &x_{N,d}\\
\end{bmatrix}
$$

Note that ${\bf X}'$ can be derived from ${\bf X}' = {\bf X}\cdot{\bf Z}$ with diagonal matrix

$$
{\bf Z} = \begin{bmatrix}
1126 &0 &\cdots &0\\
0 &1 &\cdots &0\\
\vdots\ &\vdots &\ddots &\vdots\\
0 &0 &\cdots &1\\
\end{bmatrix}
$$

Now we calculate ${\bf w}_{\text{LUCKY}}$ by

$$
\begin{aligned}
{\bf w}_{\text{LUCKY}} &= ({\bf X}'^T{\bf X}')^{-1}{\bf X}'{\bf y}\\
&= (({\bf X}{\bf Z})^T({\bf X}{\bf Z}))^{-1}({\bf X}{\bf Z})^T{\bf y}\\
&= ({\bf Z}^T{\bf X}^T{\bf X}{\bf Z})^{-1}{\bf Z}^T{\bf X}^T{\bf y}\\
&= ({\bf Z}{\bf X}^T{\bf X}{\bf Z})^{-1}{\bf Z}{\bf X}^T{\bf y}\\
&= {\bf Z}^{-1}({\bf X}^T{\bf X})^{-1}{\bf Z}^{-1}{\bf Z}{\bf X}^T{\bf y}\\
&= {\bf Z}^{-1}({\bf X}^T{\bf X})^{-1}{\bf X}^T{\bf y}\\
&= {\bf Z}^{-1}{\bf w}_{\text{LIN}}
\end{aligned}
$$

since ${\bf Z}$ is symmetric and thus ${\bf Z} = {\bf Z}^T$.

Therefore, we have ${\bf w}_{\text{LIN}} = {\bf D}{\bf w}_{\text{LUCKY}}$ where
$${\bf D} = {\bf Z}^{-1}$$

<div style="page-break-after:always;"></div>

# 9.

$$
\begin{aligned}
&\max_{\bf w}\quad\prod_{n=1}^{N}\tilde{h}(y_n{\bf x}_n)\\
&= \max_{\bf w}\quad\prod_{n=1}^{N}{1\over2}\left({y_n{\bf w}^T{\bf x}_n\over\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}+1\right)\\
&= \max_{\bf w}\quad\prod_{n=1}^{N}\left({y_n{\bf w}^T{\bf x}_n\over\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}+1\right)\\
&= \max_{\bf w}\quad\exp\left[\sum_{n=1}^{N}\ln\left({y_n{\bf w}^T{\bf x}_n\over\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}+1\right)\right]\\
&= \max_{\bf w}\quad\sum_{n=1}^{N}\ln\left({y_n{\bf w}^T{\bf x}_n\over\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}+1\right)\\
&= \min_{\bf w}\quad\sum_{n=1}^{N}-\ln\left({y_n{\bf w}^T{\bf x}_n\over\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}+1\right)\\
&= \min_{\bf w}\quad{1\over N}\sum_{n=1}^{N}-\ln\left({y_n{\bf w}^T{\bf x}_n\over\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}+1\right)\\
&= \min_{\bf w}\quad{1\over N}\sum_{n=1}^{N}\ln\left({\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}\over y_n{\bf w}^T{\bf x}_n+\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}\right)\\
\end{aligned}
$$

We have
$$\tilde{E}_{\text{in}}({\bf w}) = {1\over N}\sum_{n=1}^{N}\ln\left({\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}\over y_n{\bf w}^T{\bf x}_n+\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}\right)$$

We first denote
$$A = {\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}\over y_n{\bf w}^T{\bf x}_n+\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}, B = y_n{\bf w}^T{\bf x}_n$$
and calculate

$$
\begin{aligned}
&{\partial\tilde{E}_{\text{in}}({\bf w})\over\partial{\bf w}_i}\\
&= {1\over N}\sum_{n=1}^{N}
{\color{red}{\partial\ln{A}\over\partial{A}}}
{\partial((\sqrt{1+B^2})({B+\sqrt{1+B^2}})^{-1})\over\partial{B}}
{\partial(y_n{\bf w}^T{\bf x}_n)\over\partial{\bf w}_i}\\
&= {1\over N}\sum_{n=1}^{N}
{\color{red}{1\over A}}
{\color{blue}{\partial((\sqrt{1+B^2})({B+\sqrt{1+B^2}})^{-1})\over\partial{B}}}
{\partial(y_n{\bf w}^T{\bf x}_n)\over\partial{\bf w}_i}\\
&= {1\over N}\sum_{n=1}^{N}
{1\over A}
{\color{blue}{{B-\sqrt{1+B^2}}\over{B\sqrt{B^2+1}+1+B^2}}}{\color{orange}{\partial(y_n{\bf w}^T{\bf x}_n)\over\partial{\bf w}_i}}\\
&= {1\over N}\sum_{n=1}^{N}
{1\over A}
{{B-\sqrt{1+B^2}}\over{B\sqrt{B^2+1}+1+B^2}}
{\color{orange}{y_nx_{n,i}}}\\
&= {1\over N}\sum_{n=1}^{N}
{y_n{\bf w}^T{\bf x}_n+\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}\over{\sqrt{1+(y_n{\bf w}^T{\bf x}_n)^2}}}
{{{y_n{\bf w}^T{\bf x}_n}-\sqrt{1+({y_n{\bf w}^T{\bf x}_n})^2}}\over{{y_n{\bf w}^T{\bf x}_n}\sqrt{({y_n{\bf w}^T{\bf x}_n})^2+1}+({y_n{\bf w}^T{\bf x}_n})^2+1}}
{y_nx_{n,i}}\\
&= {1\over N}\sum_{n=1}^{N}
2\tilde{h}(y_n{\bf x}_n)\cdot
{{{y_n{\bf w}^T{\bf x}_n}-\sqrt{1+({y_n{\bf w}^T{\bf x}_n})^2}}\over{{y_n{\bf w}^T{\bf x}_n}\sqrt{({y_n{\bf w}^T{\bf x}_n})^2+1}+({y_n{\bf w}^T{\bf x}_n})^2+1}}
{y_nx_{n,i}}\\
\end{aligned}
$$

therefore we have

$$
\nabla\tilde{E}_{\text{in}}({\bf w}) = {1\over N}\sum_{n=1}^{N}
2\tilde{h}(y_n{\bf x}_n)\cdot
{{{y_n{\bf w}^T{\bf x}_n}-\sqrt{1+({y_n{\bf w}^T{\bf x}_n})^2}}\over{{y_n{\bf w}^T{\bf x}_n}\sqrt{({y_n{\bf w}^T{\bf x}_n})^2+1}+({y_n{\bf w}^T{\bf x}_n})^2+1}}
{y_n{\bf x}_n}
$$

(seriously, what is this... I dont think this is correct)

<div style="page-break-after:always;"></div>

# 10.

## Result & Finding

![|500](School/Course%20Homeworks/HTML/assets/p10-scatter.png)

Most of the training results are concentrated in the lower left corner, that is, when $E_{\text{in}}$ is smaller, $E_{\text{out}}$ will also be smaller. However, judging from the axis scale, the value of $E_{\text{out}}$ is still much larger than $E_{\text{in}}$. Therefore, using the method given, even if the performance on training data is not bad, the performance on real data still needs to be improved.

## Code Snapshot

![|400](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241016152851.png)

<div style="page-break-after:always;"></div>

# 11.

## Result & Finding

![|500](School/Course%20Homeworks/HTML/assets/p11-plot.png)

When the size of training data ($N$) increases, $E_{\text{out}}$ drops significantly, but there is no significant improvement when $N \gt 500$; while $E_{\text{in}}$ becomes slightly larger as $N$ increases, but it is not very obvious. In conclusion, from this figure, we can observe that the size of the training data is not necessarily the bigger the better; as long as it reaches a certain amount, $E_{\text{out}}$​ and $E_{\text{in}}$​ will become quite close.

## Code Snapshot

![|400](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241016152928.png)

<div style="page-break-after:always;"></div>

# 12.

## Result & Finding

![|500](School/Course%20Homeworks/HTML/assets/p12-plot.png)

Looking at the results of question 12 alone, the error appears to be decreasing, and the trend between $N$ and the error is very similar to that of question 11. However, if we plot the results of question 11 on the same graph, we can observe some interesting findings.

![|500](School/Course%20Homeworks/HTML/assets/p11-12-mix-plot.png)

We can see that although both converge at $N=500$, the error (whether $E_{\text{in}}$ or $E_{\text{out}}$​) with only two features is actually larger than that with twelve features. In other words, reducing the number of features worsens the training results.

## Code Snapshot

![|400](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241016152959.png)

<div style="page-break-after:always;"></div>

# 13.
