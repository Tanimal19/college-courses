b11902038 資工三 鄭博允

---

# 5.
What ChatGPT say make no sense. 

For the "Given" part, ChatGPT is perfectly understand the question.

However, for the "Process" part, ChatGPT is wrong since we can't use the first $N-1$ term to find the coefficients of the polynomial $P(x)$. For example, assuming $N=2$, that means $P(x) = a_2x^2 + a_1x^1 + a_0$ while we only know the first term of the integer sequence $P(1) = y_1$. Therefore it's obvious that we can't solve $P(x)$. 

Moreover, ChatGPT takes a wrong example in the "Example" part, where it assume the polynomial be degree 2, but have the first 3 terms of the integer sequence, which is not conform to our question.

<div style="page-break-after:always;"></div>

# 6.
First, for each number, the probability of "being green" is:
$${1\over2}$$
Then, for each number, the probability of "being green on all five tickets" is:
$$\left({1\over2}\right)^5 = {1\over32}$$
Therefore, the probability of "each number is not green on at least one tickets", which is the complement of the given statement "some number is green on all five tickets", is:
$$\left(1-{1\over32}\right)^{16} = \left({31\over32}\right)^{16}$$
In the end, we have the probability of "some number is green on all five tickets" being:
$$1- \left({31\over32}\right)^{16}$$

<div style="page-break-after:always;"></div>

# 7.
From Problem 6, for number 5, the probability of "being green on all five tickets" is:
$${1\over32}$$

<div style="page-break-after:always;"></div>

# 8.
Since the original equation is for one machine, we first use union bound to extend it to all machine $m \in \{1, 2, \cdots, M\}$:
$$P\left(\mu_m \gt {c_m\over N_m} + \sqrt{\ln{t} − {1\over2}\ln{\delta}\over N_m}\right) \le \sum^{M}_{m=1}\delta t^{−2} = M\delta t^{−2}$$

Then, we use union bound again to extend it to $t \in \{M+1, M+2, \cdots\}$:
$$P\left(\mu_m \gt {c_m\over N_m} + \sqrt{\ln{t} − {1\over2}\ln{\delta}\over N_m}\right) \le \sum^{\infty}_{t=M+1}M\delta t^{−2} \le M^2\delta$$
by
$$\sum^{\infty}_{t=M+1}M\delta t^{−2} = M\delta\sum^{\infty}_{t=M+1} t^{−2} \le M\delta\sum^{\infty}_{t=1} t^{−2} \le M\delta{\pi^2\over6} \le M^2\delta$$
(since $M \ge 2 \ge {\pi^2\over6}$)

After that, we use a new $\delta' = M^2\delta$ to replace old $\delta$:
$$P\left(\mu_m \gt {c_m\over N_m} + \sqrt{\ln{t} + \ln{M} − {1\over2}\ln{\delta'}\over N_m}\right) \le \delta'$$
by
$${1\over2}\ln{\delta'} = {1\over2}\ln{M^2\delta} = \ln{M} + {1\over2}\ln{\delta}$$

Then we can derive the complement of this probability, that is:
$$P\left(\mu_m \le {c_m\over N_m} + \sqrt{\ln{t} + \ln{M} − {1\over2}\ln{\delta'}\over N_m}\right) \ge 1-\delta'$$
for all $t \in \{M+1, M+2, \cdots\}$ and  $m \in \{1, 2, \cdots, M\}$.

<div style="page-break-after:always;"></div>

# 9.
Notation: $h({\bf x}) = h(\{x_1, x_2, \cdots, x_k\})$

For a symmetric boolean function $h$ with $k$ input fields, since each $x_i$ can only be $+1$ or $-1$, the domain oo $h$ is upper-bounded by $2^k$, that is, $N = 1 \dots 2^k$.

Also, since $h({\bf x})$ depends on the number of $+1$'s (or $-1$'s), $h$ has only $k+1$ possible outputs. For each output, it can be either $+1$ or $-1$, therefore there are $2^{k+1}$ possible dichotomies.

When $N = k+1$, $h$ can shatter certain input combination:
${\bf x}_1$ has zero $+1$, ${\bf x}_2$ has one $+1$, $\cdots$, ${\bf x}_N$ has $k$ $+1$
which has $2^{k+1}$ possible output combinations.

However, when $N \gt k+1$, $h$ can't shatter any input combination, since there must be some ${\bf x}_i$ and ${\bf x}_j$ has same number of $+1$'s, and thus $h({\bf x}_i) = h({\bf x}_j)$. In other words, we can't produce some dichotomies that has $h({\bf x}_i) \ne h({\bf x}_j)$.

Therefore, the break point of $h$ is $N = k+2$, and the vc dimension is $k+1$.

<div style="page-break-after:always;"></div>

# 10.
First, we can split the domain into three regions:
(a) $0$ to $\theta$,
(b) $\theta$ to $\large{\theta\over|\theta|}$
(c) $\theta$ to $-\large{\theta\over|\theta|}$.

For example, if $\theta \ge 0$, the three regions will be:
![|274](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241002175122.png)

Let's start from region (c). The probability of getting error (that is, $h_{s,\theta}(x) = -y$) is determined by $s$:
- If $s = 1$, then $s\cdot\text{sign}(x - \theta) = \text{sign}(x) = y$ for all $x$ when $y$ is not flipped, thus the **error only occur when $y$ is flipped**, and the probability is $p$.
- If $s = -1$, the situation is opposite to the above, where the **error occur when $y$ is not flipped**, thus the prbability is $1 - p$.
We can use the equation below to describe the probability:
$${1\over2}\ {\color{red}-}\ s({1\over2}-p)$$

For region (b), we can easily find that it has the same characteristics of region (c), thus the probability of getting error is also
$${1\over2}\ {\color{red}-}\ s({1\over2}-p)$$

For region (a), the probability of getting error is also determined by $s$, but slightly different from above:
- If $s = -1$, then $s\cdot\text{sign}(x - \theta) = \text{sign}(x) = y$ for all $x$ when $y$ is not flipped, thus the **error only occur when $y$ is flipped**, and the probability is $p$.
- If $s = 1$, the situation is opposite to the above, where the **error occur when $y$ is not flipped**, thus the prbability is $1 - p$.
Therefore, the equation becomes:
$${1\over2}\ {\color{blue}+}\ s({1\over2}-p)$$

$E_{\text{out}}(h_{s,\theta})$ means the probability of getting error in all regions ($[-1,1]$), we can calculate it by multiplying the probability of $x$ falling in each region by the probability of getting error in that region and adding them together:
$$\begin{aligned}
E_{\text{out}}(h_{s,\theta}) &= [{1\over2}+s({1\over2}-p)]\cdot{|\theta|\over2} + [{1\over2}-s({1\over2}-p)]\cdot{(2-|\theta|)\over2}\\\\
&= [{1\over2}+v]\cdot{|\theta|\over2} + [{1\over2}-v]\cdot{(2-|\theta|)\over2}\\\\
&= {1\over2} -v + v\cdot|\theta|
\end{aligned}$$

Then, we can substitute ${1\over2}-v$ with $u$, and we get:
$$E_{\text{out}}(h_{s,\theta}) = u + v\cdot|\theta|$$

While we use $\theta \ge 0$ for example, the proof is also valid when $\theta \le 0$.

<div style="page-break-after:always;"></div>

# 11.
## Result
![|500](School/Course%20Homeworks/HTML/assets/p11-scatter.png)

median of $E_{out} - E_{in}$: 0.2306835251071872

## Code Snapshot
![|500](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241003110406.png)

<div style="page-break-after:always;"></div>

# 12.
## Result
![|500](School/Course%20Homeworks/HTML/assets/p12-scatter.png)

median of $E_{out} - E_{in}$: -0.007453724959534683

## Findings
For simplicity, we use $A$ to represent a regular decision stump learning algorithm and $A_r$ to represent a random one.

The figure below is a mixture of results from p11 and p12, we can find that $E_in$ of $A$ is generally smaller than $A_r$, this indicates $A$ indeed finds a better $g$ than $A_r$, within sample data.

However, when it comes to real-world data, the $E_{out}$ of both $A$ and $A_r$ are very similar, and the median of $E_{out} - E_{in}$ is much larger in $A$ compared to $A_r$, which means the $g$ found by $A$ didn't has the same good performance in real-world data as in sample data.

![|500](School/Course%20Homeworks/HTML/assets/mix-scatter.png)

## Code Snapshot
![|500](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020241003112257.png)

<div style="page-break-after:always;"></div>

# 13.
