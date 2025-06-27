b11902038 資工三 鄭博允

---

# Question 1

Python use IEEE 754 to store floating point in 64-bit binary format, thus the actual value of `0.1` is `0.1000000000000000055511151231257827021181583404541015625`,
and `0,2` is `0.200000000000000011102230246251565404236316680908203125`.

And the sum of them is not equal to `0.3`, thus the code return false.

> Reference: https://www.h-schmidt.net/FloatConverter/IEEE754.html

---
# Question 2

## 1.
convert hexadecimal to binary value (IEEE 754)
- `0xC3336000` -> `11000011001100110110000000000000` -> $-1 \times 2^7 \times 1.0110011011_2$
- `0x42E27400` -> `01000010111000100111010000000000` -> $1 \times 2^6 \times 1.1100010011101_2$

get formula:
$$(-1 \times 2^7 \times 1.0110011011_2) + (1 \times 2^6 \times 1.1100010011101_2)$$

shift number with smaller expoent
$$(-1 \times 2^7 \times 1.0110011011_2) + (1 \times 2^7 \times 0.11100010011101_2)$$

add significands
$$(-1 \times 2^7 \times 1.0110011011_2) + (1 \times 2^7 \times 0.1100010011101_2) = -0.10000100010011 \times 2^7$$

normalize
$$-0.10000100010011_2 \times 2^7 = -1.0000100010011_2 \times 2^6$$

convert back to hexadecimal
$-1.010000111011_2 \times 2^6$ -> `11000010100001000100110000000000` -> `0xC2844C00`

## 2.
convert hexadecimal to binary
- `0xC3336000` -> `11000011001100110110000000000000` 
- `0x42E27400` -> `01000010111000100111010000000000`

convert binary to decimal by 2's complement
- `11000011001100110110000000000000` -> - `00111100110011001010000000000000`
- `01000010111000100111010000000000` -> unchange

calculate formula
`-00111100110011001010000000000000` + `01000010111000100111010000000000`
= `00000110000101011101010000000000`

convert to hexadecimal
`00000110000101011101010000000000` -> `0x0615D400`

---
# Question 3

First, let's think about how to convert the value to binary using IEEE 754.

We can ignore sign bit right now, because it's should always be 1 bit.

Since$$1.3 = (1 + \text{frac}) \times 2^{(\text{expo} - \text{bias})}$$we can calculate fraction and exponent by$$(1 + \text{frac}) = 1.3 / 2^{(\text{expo} - \text{bias})}$$noted that the value $(1 + \text{frac})$ is between 1 and 2. 

Therefore, we can iterate all possible value of $(\text{expo} - \text{bias})$ and find a value that is between 1 and 2, which is the value we want. Then record the corresponding $\text{expo}$ and $\text{frac}$.

After getting the binary, we can calculate it's real value $X$ using IEEE 754, then calculate the error between original value 1.3 and the calculated value $X$.

We can iterate above operation from $S = 1, E=1, F=8$ to $S=1,E=8,F=1$, and find the combination that has smallest error.

It's difficult to calculate this by hand, so I implement it as Python program, where part of the result is shown at below figure.
![|400](School/Course%20Homeworks/Computer%20Architecture/assets/Pasted%20image%2020241107163309.png)

Therefore, the combination that has smallest error is $S=1, E=2, F=7$ or $S=1, E=3, F=6$.
And the calculated value is $1.296875$.

---
# Question 4
Because multiplication can be done with multiple adders, which can be pipelined, make it much faster than just using single multiplier. However, division can't be done in parallel, since substraction is depend on the sign of remainder.

Also, division need additional error handling method to insure correct result, which cause more time than multiplication, which doesn't need error handling.