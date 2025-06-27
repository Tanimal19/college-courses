b11902038 資工三 鄭博允

---

# 1.
## (a).
For pipelined processor, the clock cycle time is determined by slowest stage latency, since each stage need to complete in one clock cycle.
Therefore, the clock cycle time is **360ps**.

For single-cycle processor, all stages complete in one clock cycle, thus the clock cycle time is determined by the sum of all stage latency, which is 250 + 360 + 150 + 280 + 200 = **1240ps**.

## (b).
An `sw` (store word) instruction involves the following stages: IF, ID, EX, MEM.
However, stage WB will still be executed.

Assume we only execute one `sw` instruction, then the latency of pipelined processor is 360ps x 5 stages = **1800ps**. To be noted, pipelining improve instruction throughput, rather than latency of individual  instruction.

For single-cylce processor, all instruction need exactly one clock cycle, thus the latency is same as clock cycle time = **1240ps**.

## (c).
For pipelined processor, if the pipeline is filled (since we execute a large program, this might be true), then each instruction only use one clock cycle, that is **360ps**.

For single-cycle processor, one instruction still use one clock cycle, that is **1240ps**.

Therefore, for a single instruction, the performance of pipelined processor is 1240/360 = **3.44** faster than single-cycle processor.

<br><br>

## (d).
Since the clock cycle time is determined by the slowest stage latency, splitting the slowest stage (ID) can achieve the best improvement.

The clock cycle time after splitting ID stage is $\max(250, 180, 150, 280, 200)$ = **280ps**. Therefore the performance improve by 360/280 = **1.29**.

## (e).
**Data harzard**, happened when an instruction depends on previous instruction's result, can cause bubbles in the pipeline.
For example, `sub x2, x19, x3` depends on `add x19, x0, x1`.

1. To address this, we can use **forwarding**, which the instruction can directly retrieve data from internal buffers (after EX), instead of waiting it to be stored into memory.
2. Or, we can reorder the instructions (by compiler), execute some independent instructions between two instructions that cause hazard.

<div style="page-break-after:always;"></div>

# 2.
## (a).
- **RAW**: occurs when a register is written in one instruction, but then read by another before the written is complete.
- **WAW**: occurs when a register is written by one instruction and then overwritten by another before the old value being read. (should not occur in in-order pipeline)
- **WAR**: occurs when a register is read by one instruction but then overwritten by another before the read is complete.

```
RAW:
x8 in line 2 & 5
x12 in line 5 & 6
x12 in line 6 & 7
x13 in line 7 & 8
x12 in line 11 & 12
x12 in line 12 & 13
x13 in line 13 & 14
x12 in line 17 & 18
x12 in line 18 & 19
x13 in line 19 & 20
x12 in line 20 & 21
x10 in line 21 & 24

WAW:
x12 in line 5 & 6
x12 in line 11 & 12
x12 in line 17 & 18

WAR:
x12 in line 19 & 20
```

## (b).
With forwarding, we can avoid all RAW hazards, thus every instruction can execute without stalls in one cycles. The total cycles is **15 (for 15 instructions) + 4 (the remaining stages for the last instruction) = 19 cycles**.

Without forwarding, we need two cycle stalls for each RAW hazards, thus we have **2 x 12 = 24** more cycles, that is 19 + 24 = **43 cycles**.

<br>

## (c).
```python
add x8, x6, x7

# precompute memory addresses for tool[a], tool[b], tool[c]
slli x11, x6, 2
slli x12, x7, 2
slli x13, x8, 2
add x11, x5, x11    # x11 = address of tool[a]
add x12, x5, x12    # x12 = address of tool[b]
add x13, x5, x13    # x13 = address of tool[c]

# compute results
lw x9, 0(x11)       # d = tool[a]
lw x10, 0(x12)
add x10, x10, x6    # e = tool[b] + a
lw x8, 0(x13)       # c = tool[c]

add x6, x10, x0
```

1. Register Renaming: use three temporary registers for three different array address, remove the dependency.
	- `x11` for `tool[a]`
	- `x12` for `tool[b]`
	- `x13` for `tool[c]`
2. Reordering: make all `slli` execute together, avoiding operate with same register in two subsequent instruction.

The cycles needed for this code is **12 (for 12 instructions) + 4 = 16 cycles**, since there's no hazards. And the performance is improve by **43/16 = 2.69 times**.