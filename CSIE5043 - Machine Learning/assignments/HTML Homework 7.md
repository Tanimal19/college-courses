b11902038 資工三 鄭博允

---

# 5.
For each $g_t$, we can see it as a independent Bernoulli trial, that is $g_t({\bf x})$ has only two result:
- correct ($p=1-e_t$)
- incorrect ($p=e_t$)

Therefore $G({\bf x})$ is actually doing independent Bernoulli trail for $2M+1$ times, the pmf is
$$P(X = k) = \dbinom{2M+1}{k} e_t^k(1-e_t)^{2M+1-k}$$
where $X$ is the number of $g_t$ that make incorrect prediction. 

And $E_{\text{out}}(G)$ is actually the probability of more than half of the $g_t$ predict incorrectly, which is
$$P(X > M) = \sum^{2M+1}_{k=M+1} \dbinom{2M+1}{k} e_t^k(1-e_t)^{2M+1-k}$$

Using Hoeffding’s inequality,
$$P(X - \mathbb{E}[X] \ge t) \le \exp\left(-{2t^2 \over n}\right)$$
The expect value $\mathbb{E}[X] = (2M+1)e_t$, therefore we can set the threshold $t$ as $t=M + {1\over2} -\mathbb{E}[X]$ to fit our requirement.

Thus we have
$$\begin{aligned}
P(X \ge M + {1\over2}) = P(X \gt M) &\le \exp\left(-{2(M + {1\over2} - (2M+1)e^t)^2 \over 2M+1}\right)\\
& \le \exp\left(-2(2M+1)({1\over2} -e_t)^2\right)
\end{aligned}$$

In the end, we have the upper bound being
$$E_{\text{out}}(G) \le \exp\left(-2(2M+1)({1\over2} -e_t)^2\right)$$

<div style="page-break-after:always;"></div>


# 6.

Since $$\begin{aligned}
U_{t+1} &= \sum^N_{n=1} u_n^{(t+1)}\\
&= U_t \cdot \epsilon_t \cdot \sqrt{1-\epsilon_t \over \epsilon_t} + U_t \cdot (1-\epsilon_t) \cdot \sqrt{\epsilon_t \over 1-\epsilon_t}\\
&= U_t \cdot 2 \sqrt{\epsilon_t(1-\epsilon_t)}
\end{aligned}$$
We have
$${U_{t+1} \over U_t} = 2 \sqrt{\epsilon_t(1-\epsilon_t)}$$

<div style="page-break-after:always;"></div>

# 7.
Assume using the squared error loss function. Then the residual for each iteration is
$$r_t = - \nabla_{f_t} \left((y - f_t({\bf x}))^2\right)$$

First we initialize the model with $f_0({\bf x})$. Then the residual after first iteration is
$$r_1 = 2(y - g_0({\bf x}))$$

<div style="page-break-after:always;"></div>

# 8.
(1) After update all $s_n$ by
$$s_n \leftarrow s_n + \alpha_t g_t({\bf x_n})$$
we have
$$\begin{aligned}
&\sum_{n=1}^N(y_n - s_n) g_t({\bf x}_n) = \sum_{n=1}^N(y_n - s_n - \alpha_t g_t({\bf x_n})) g_t({\bf x}_n) =\\
&\sum_{n=1}^N(y_n - s_n) g_t({\bf x}_n) -\alpha_t \sum_{n=1}^N g_t({\bf x}_n)^2
\end{aligned}$$


(2) The (squared) loss we want to minimize is
$$\sum_{n=1}^N (y_n - s_n - g_t({\bf x}_n))^2$$
To optimize it, we can set it's derivative to 0:
$${\partial \over \partial g_t({\bf x}_n)}\sum_{i=1}^N (y_i - s_i - g_t({\bf x}_i))^2 = -2(y_n - s_n - g_t({\bf x}_n)) = 0$$
Then we have $g_t({\bf x}_n) = y_n - s_n$.

<div style="page-break-after:always;"></div>

# 9.

<div style="page-break-after:always;"></div>

# 10.
## Result
![|475](School/Course%20Homeworks/HTML/assets/hw7-p10.png)

Every $g_t$ generate by decision stump has different Ein, while most of them around 0.46 ~ 0.48. 

The $\epsilon_t$ is slightly smaller and smoother than original $E_{\text{in}}(g_t)$ because it's normalized by weight.

Both $E_{\text{in}}(g_t)$ and $\epsilon_t$ has no obvious relationship with $t$, that is the error of decision stump is independent from round.

## Code Snapshot
![|350](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241216143600.png)

<div style="page-break-after:always;"></div>

# 11.
## Result
![|475](School/Course%20Homeworks/HTML/assets/hw7-p11.png)

While we see individual $g_t$ has bad performance in problem 10, problem 11 shows that the meta $G_t$ has better $E_{\text{in}}$ after "aggregate" more $g_t$, this describe the effect of adaboost is significant.

However, the $E_{\text{out}}(G_t)$ has no improvement even aggregate all $g_t$, this indicate that adaboost is easy to be overfitting.

<div style="page-break-after:always;"></div>

# 12.

<div style="page-break-after:always;"></div>

# 13.

<div style="page-break-after:always;"></div>
