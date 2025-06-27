b11902038 資工三 鄭博允

---

# 5.

## Response from Gemini
A Possible Application of Active Learning: Medical Image Analysis
Active learning can be a powerful tool in medical image analysis due to the following reasons:
1. Labeling Costs: Manually labeling large datasets of medical images is time-consuming and expensive. Active learning can significantly reduce the required labeling effort by intelligently selecting the most informative images for human annotation.  
2. Expert Expertise: Medical experts may have limited time or resources to label all images. Active learning can prioritize images that are most likely to be challenging or require expert judgment.  
3. Model Improvement: By iteratively selecting and labeling informative images, active learning can help models learn faster and more accurately, improving their performance on medical image tasks such as classification, segmentation, and detection.  

Specific Use Case: Breast Cancer Detection
In breast cancer detection, active learning could be used to:
- Prioritize images that are most likely to contain abnormalities or be difficult to classify.
- Reduce the workload for radiologists by focusing their attention on the most challenging cases.  
- Improve the accuracy of breast cancer detection models by training them on a more diverse and representative dataset.

By carefully selecting which images to label, active learning can help optimize the training process for medical image analysis models, leading to more accurate and efficient diagnostic tools.  

Would you like to explore another application of active learning?

## Argument
I think the example of Medical Image Analysis proposed by Gemini is very suitable for active learning scenarios.

First, I recently read a paper [2] on a similar topic. The paper described using AI to assist pathologists in locating tumor positions, which is quite similar to the active learning process as I understand it. In the paper, the AI would use specific strategies to select potentially pathological slide areas, which would then be confirmed by pathologists and returned to the AI for more detailed screening, continuously cycling through these interactions. Therefore, similar application scenarios have already appeared in fields such as medical image analysis.

Next, because medical images often show differences in very small areas, which might make it difficult for the model to distinguish, it's best to have domain experts assist with classification, allowing the model to classify more accurately.

Finally, although it's feasible to use active learning to train a model for Medical Image Analysis, I believe that relying entirely on the model for judgment is still not completely reliable. In the current situation, the model is best used as a supportive tool to maximize its effectiveness.

## Reference:
[1] (Gemini) https://g.co/gemini/share/26d958ad5c09
[2] https://arxiv.org/abs/2302.07309

<div style="page-break-after:always;"></div>

# 6.
## Response from Gemini
Yes, machine learning can be used to predict earthquakes. While it's not perfect and doesn't guarantee 100% accuracy, it has shown promise in identifying areas at higher risk and potentially providing early warnings.  

Here's how it works:

1. Data Collection: Researchers gather vast amounts of data related to earthquakes, such as historical seismic activity, ground deformation, electromagnetic signals, and environmental factors.
2. Feature Engineering: This involves selecting and preparing relevant features from the data that might be indicative of an earthquake.  
3. Model Training: Machine learning algorithms, like neural networks, support vector machines, or random forests, are trained on this data. The models learn to identify patterns and relationships between these features and earthquake occurrences.  
4. Prediction: Once trained, the model can be used to analyze new data and predict the likelihood of an earthquake in a specific region.  

While machine learning has shown potential, it's important to note that predicting earthquakes remains a complex challenge due to the chaotic nature of the Earth's interior. However, ongoing research and advancements in technology are continually improving our ability to forecast seismic events and potentially save lives.  

Would you like to know more about specific techniques or research being done in this field?

## Argument
I think Gemini's points are valid. Although we currently can't fully predict earthquakes, machine learning can indeed be used to analyze earthquake data and attempt to identify patterns.

While we haven't yet discovered clear patterns in earthquake occurrence, I believe this is due to the numerous factors within the Earth's interior that have made it difficult for past analytical methods to identify regularities. However, by using unsupervised learning techniques, which excel at discovering unknown data patterns, we may actually uncover patterns in earthquake occurrence in the future.

## Reference:
[1] (Gemini) https://g.co/gemini/share/dcbb86b74d11

<div style="page-break-after:always;"></div>

> For problem 7, 8
> to reduce the use of notation, ${\bf w}$ will be auto transpose if needed,
> that is, ${\bf w}{\bf x} = {\bf w}^T{\bf x}$  

# 7.
We update whenever $\text{sign}(0)$ happened.
Assume ${\bf w}_0$ is a zero vector.

Consider an example:
${\bf x}^{\text{orig}} \in \mathbb{R}^2$
${\bf x}^{\text{orig}}_1 = (1,1)$ with $y_1 = 1$
${\bf x}^{\text{orig}}_2 = (-1, -1)$ with $y_2 = -1$

For PLA on $\{({\bf x}_n, y_n)\}^N_{n=1}$
we have
$${\bf w}_{\text{PLA}} =
\begin{bmatrix}
1\\
1\\
1\\
\end{bmatrix}$$
(calculation at appendix 7-a)

For PLA on $\{({\bf x}'_n, y_n)\}^N_{n=1}$
we have
$${\bf w}'_{\text{PLA}} =
\begin{bmatrix}
0\\
2\\
2\\
\end{bmatrix}$$
(calculation at appendix 7-b)

Now applied both ${\bf w}_{\text{PLA}}$ and ${\bf w}'_{\text{PLA}}$ on ${\bf x}^{\text{orig}}_3 = (-0.2, -0.2)$,
$${\bf x}_3 =
\begin{bmatrix}
1\\
-0.2\\
-0.2\\
\end{bmatrix},
{\bf x}'_3 =
\begin{bmatrix}
2\\
-0.2\\
-0.2\\
\end{bmatrix}$$
we have
$$
\begin{aligned}
&\text{sign}({\bf w}_{\text{PLA}}{\bf x}_3) = \text{sign}\left(
\begin{bmatrix}
1,1,1
\end{bmatrix}
\begin{bmatrix}
1\\
-0.2\\
-0.2\\
\end{bmatrix}
\right)
= 1\\
\ne\ &\text{sign}({\bf w'}_{\text{PLA}}{\bf x}'_3) = \text{sign}\left(
\begin{bmatrix}
0, 2, 2
\end{bmatrix}
\begin{bmatrix}
2\\
-0.2\\
-0.2\\
\end{bmatrix}
\right)
= -1
\end{aligned}$$

Therefore, ${\bf w}_{\text{PLA}}$ and ${\bf w}'_{\text{PLA}}$ are not equivalent.

## appendix 7-a
For PLA on $\{({\bf x}_n, y_n)\}^N_{n=1}$
the 1st update happened at ${\bf x}_1, y_1$ since
$$\text{sign}({\bf w}_0{\bf x}_1) = \text{sign}(0)$$
we then have
$${\bf w}_1 =  {\bf w}_0 + y_1{\bf x}_1 =
\begin{bmatrix}
0\\
0\\
0\\
\end{bmatrix}+
\begin{bmatrix}
1\\
1\\
1\\
\end{bmatrix}=
\begin{bmatrix}
1\\
1\\
1\\
\end{bmatrix}$$
then the PLA stopped since
$$\begin{aligned}
\text{sign}({\bf w}_1{\bf x}_1) &= \text{sign}(3) = 1 = y_1\\
\text{sign}({\bf w}_1{\bf x}_2) &= \text{sign}(-1) = -1 = y_2
\end{aligned}
$$
thus, we have
$${\bf w}_{\text{PLA}} =
\begin{bmatrix}
1\\
1\\
1\\
\end{bmatrix}$$


## appendix 7-b
For PLA on $\{({\bf x}'_n, y_n)\}^N_{n=1}$
the 1st update happened at ${\bf x}'_1, y_1$ since
$$\text{sign}({\bf w}_0{\bf x}'_1) = \text{sign}(0)$$
we then have
$${\bf w}'_1 =  {\bf w}_0 + y_1{\bf x}'_1 =
\begin{bmatrix}
0\\
0\\
0\\
\end{bmatrix}+
\begin{bmatrix}
2\\
1\\
1\\
\end{bmatrix}=
\begin{bmatrix}
2\\
1\\
1\\
\end{bmatrix}$$
the 2nd update happened at ${\bf x}'_2, y_2$ since
$$\text{sign}({\bf w'}_1{\bf x}'_2) = \text{sign}(2) = 1 \ne y_2 = -1$$
we then have
$${\bf w}'_2 =  {\bf w}'_1 + y_2{\bf x}'_2 =
\begin{bmatrix}
2\\
1\\
1\\
\end{bmatrix}+
\begin{bmatrix}
-2\\
1\\
1\\
\end{bmatrix}=
\begin{bmatrix}
0\\
2\\
2\\
\end{bmatrix}$$
then the PLA stopped since
$$\begin{aligned}
\text{sign}({\bf w'}_1{\bf x}'_1) &= \text{sign}(4) = 1 = y_1\\
\text{sign}({\bf w'}_1{\bf x}'_2) &= \text{sign}(-4) = -1 = y_2
\end{aligned}
$$
thus, we have
$${\bf w}'_{\text{PLA}} =
\begin{bmatrix}
0\\
2\\
2\\
\end{bmatrix}$$
<div style="page-break-after:always;"></div>

# 8.
We update whenever $\text{sign}(0)$ happened.
Assume ${\bf w}_0$ is a zero vector.

Let $P$ denote the PLA on $\{({\bf x}_n, y_n)\}^N_{n=1}$ and $P'$ denote the PLA on $\{({\bf x}'_n, y_n)\}^N_{n=1}$

(1) For each update of $P$ at ${\bf x}_t, y_t$, we have
$${\bf w}_{t+1} = {\bf w}_t + y_t{\bf x}_t =
\begin{bmatrix}w_{t,0} + y_t\cdot 1\\w_{t,1} + y_t\cdot x^{\text{orig}}_{t,1}\\\vdots\\\end{bmatrix}$$
Therefore, start from ${\bf w}_0$, after $T$ times update where $t = c_1, c_2, \cdots, c_T$ we have
$${\bf w}_T =
\begin{bmatrix}{\large\displaystyle\sum^{c_T}_{t=c_1}}(y_t)\\{\large\displaystyle\sum^{c_T}_{t=c_1}}(y_t\cdot x^{\text{orig}}_{t,1})\\\vdots\\\end{bmatrix}$$

<br>

(2) For each update of $P'$ at ${\bf x}'_t, y_t$, we have
$${\bf w}'_{t+1} = {\bf w}'_t + y_t{\bf x}'_t =
\begin{bmatrix}w'_{t,0} + y_t\cdot 3\\w'_{t,1} + y_t\cdot 3x^{\text{orig}}_{t,1}\\\vdots\\\end{bmatrix}$$
Therefore, start from ${\bf w}_0$, after $T$ times update where $t = c'_1, c'_2, \cdots, c'_T$ we have
$${\bf w}'_{T} =
\begin{bmatrix}3{\large\displaystyle\sum^{c'_T}_{t=c'_1}}(y_t)\\3{\large\displaystyle\sum^{c'_T}_{t=c'_1}}(y_t\cdot x^{\text{orig}}_{t,1})\\\vdots\\\end{bmatrix}$$

<br>

(3) By (1) and (2), we can deduce that if the path till the $T$-th update of $P$ and $P'$ are the same, that is $c_i = c'_i$ for $i = 1, \cdots, T$, then
$${\bf w}'_T = 3{\bf w}_T$$
and thus
$$\text{sign}({\bf w}'_T{\bf x}_n) = \text{sign}(3{\bf w}_T\cdot{\bf x}_n) = \text{sign}({\bf w}_T{\bf x}_n)$$
for $n = 1, \cdots, N$

<br>

Since ${\bf w}_0$ is a zero vector and we update PLA whenver $\text{sign}(0)$ happened, the 1-st update of $P$ and $P'$ are at the same point, thus $c_1 = c'_1$.
By (3), we have
$$\text{sign}({\bf w}_1{\bf x}_n) = \text{sign}({\bf w}'_1{\bf x}_n)$$
for $n = 1, \cdots, N$
and therefore the next update of $P$ and $P'$ will be at the same point, and so on.

In the end, the final update path of $P$ and $P'$ will be totally same, and thus we have
$$\text{sign}({\bf w}_{\text{PLA}}{\bf x}_n) = \text{sign}({\bf w}'_{\text{PLA}}{\bf x}_n)$$
for $n = 1, \cdots, N$

Therefore, ${\bf w}_{\text{PLA}}$ and ${\bf w}'_{\text{PLA}}$ are equivalent.
<div style="page-break-after:always;"></div>

# 9.
N/A
<div style="page-break-after:always;"></div>

# 10.
## Result
![|450](School/Course%20Homeworks/HTML/assets/pla_distribution.png)

The average is 100.909 and the median is 102. The mean and median are almost the same, which shows that the distribution of PLA results is very even when randomly selected samples.
The minimum value is 70, and the number of updates appears to have an upper bound because the maximum value, which is 113, are closer to the median.

## Code Snapshot
![|450](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020240926110906.png)
<div style="page-break-after:always;"></div>

# 11.
## Result
![|450](School/Course%20Homeworks/HTML/assets/pla_wt_plot.png)

Overall, the norm of ${\bf w}_t$ seems to get bigger with updates, but will suddenly drop off every few times.
Also, the result curves of each experiment are similar.

## Code Snapshot
![|450](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020240926111203.png)
<div style="page-break-after:always;"></div>

# 12.
## Result
![|450](School/Course%20Homeworks/HTML/assets/m_pla_distribution.png)

Surprisingly, the average is 101.29 and the median is 103, which is very simliar to the result of problem 10. Also, I've run the original PLA and the modified PLA on same random seeds, it turns out they have the exactly same result.
Thus, it seems that the new weight vector ${\bf w}_{t+1}$ will always classify $({\bf x}_{n(t)}, y_{n(t)})$ correctly. In other words, when the modified PLA meets an incorrect point, one update is enough.

## Code Snapshot
![|450](School/Course%20Homeworks/HTML/assets/Pasted%20image%2020240926111029.png)
<div style="page-break-after:always;"></div>

# 13.
N/A